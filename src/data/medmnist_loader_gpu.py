"""
MedMNIST loader - GPU-resident pipeline.

WHY THIS IS NOT A TORCH DataLoader
----------------------------------
Measured on the project hardware (bloodmnist, 800 training images, batch 32):

    data loading only      28.03 s/epoch
    loading + fwd/bwd      32.58 s/epoch
    GPU compute alone       0.89 s/epoch-equivalent

97% of wall-clock was CPU data preparation. Scaled to the full sweep that is
~150 days of compute to do ~5 days of arithmetic.

The cause: MedMNIST stores 28x28 images, ResNet-18 wants 224x224, and the old
pipeline performed that upsample on a single CPU thread for every image, every
epoch, every run - the identical interpolation repeated millions of times.

The fix: raw images are tiny (a 900-image scarcity subset is 2.1 MB as uint8;
PathMNIST's full test split is 17 MB). Load the entire split onto the GPU once
as uint8 at native resolution, then do augment -> upsample -> normalize on the
GPU per batch. DataLoader leaves the hot path completely.

Adding num_workers would have parallelised work that should not happen at all.

EXPERIMENTAL PROPERTIES PRESERVED FROM THE PREVIOUS VERSION
-----------------------------------------------------------
- Stratified shots-per-class sampling (a class is never dropped)
- Validation subsampled to match training scarcity, capped by availability
- Seed-reproducible subsets, shuffling, and augmentation
- No final batch of size 1 (BatchNorm1d raises on it)
"""
import numpy as np
import torch
import torch.nn.functional as F
import medmnist

import config


# ------------------------------------------------------------------ helpers
def _labels_of(ds) -> np.ndarray:
    return np.asarray(ds.labels).ravel()


def _raw_images(ds) -> torch.Tensor:
    """Native-resolution uint8 tensor [N, C, H, W]. No resize, no normalize."""
    imgs = np.asarray(ds.imgs)
    if imgs.ndim == 3:                      # [N,H,W] grayscale
        imgs = imgs[:, :, :, None]
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


def _stratified(labels, n_per_class, num_classes, rng):
    idx, realised = [], []
    for c in range(num_classes):
        cls = np.where(labels == c)[0]
        take = min(n_per_class, len(cls))
        if take:
            idx.extend(rng.choice(cls, size=take, replace=False).tolist())
        realised.append(int(take))
    return np.array(sorted(idx)), realised


def _proportional(labels, cap, num_classes, rng):
    n = len(labels)
    if cap is None or cap >= n:
        return np.arange(n), np.bincount(labels, minlength=num_classes).tolist()
    idx, realised = [], []
    for c in range(num_classes):
        cls = np.where(labels == c)[0]
        take = min(max(1, int(round(len(cls) * cap / n))), len(cls))
        idx.extend(rng.choice(cls, size=take, replace=False).tolist())
        realised.append(int(take))
    return np.array(sorted(idx)), realised


def _drop_singleton(idx, batch_size):
    return idx[:-1] if (len(idx) > 2 and len(idx) % batch_size == 1) else idx


# ------------------------------------------------------------------ batching
class GPUBatches:
    """
    Iterable yielding (x, y) with x already normalized at 224x224 on device.

    Augmentation is per-sample (not per-batch) and runs as a single batched
    affine warp, so it costs microseconds instead of a Python loop.
    """

    def __init__(self, images_u8, labels, batch_size, shuffle, augment,
                 device, seed, n_channels):
        self.images = images_u8.to(device, non_blocking=True)
        self.labels = labels.to(device, non_blocking=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.device = device
        self.n_channels = n_channels
        self.epoch = 0
        self._seed = seed
        self._mean = torch.tensor(config.NORM_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(config.NORM_STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self):
        return len(self.labels)

    def _augment(self, x, g):
        """Random horizontal flip + rotation, applied at native resolution."""
        b = x.size(0)
        flip = torch.rand(b, generator=g).to(self.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

        deg = (torch.rand(b, generator=g) * 2 - 1) * 10.0     # +/- 10 degrees
        th = (deg * torch.pi / 180.0).to(self.device)
        cos, sin = torch.cos(th), torch.sin(th)
        mat = torch.zeros(b, 2, 3, device=self.device, dtype=x.dtype)
        mat[:, 0, 0], mat[:, 0, 1] = cos, -sin
        mat[:, 1, 0], mat[:, 1, 1] = sin, cos
        grid = F.affine_grid(mat, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    def _prepare(self, imgs_u8, g):
        x = imgs_u8.float().div_(255.0)
        if self.augment:
            x = self._augment(x, g)
        # Upsample on GPU. This is the operation that cost 28 s/epoch on CPU.
        x = F.interpolate(x, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                          mode="bilinear", align_corners=False)
        if self.n_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self._mean) / self._std

    def __iter__(self):
        # Seeded per epoch: reproducible, and different shuffling each epoch.
        g = torch.Generator().manual_seed(self._seed * 100_003 + self.epoch)
        n = len(self.labels)
        order = (torch.randperm(n, generator=g).to(self.device)
                 if self.shuffle else torch.arange(n, device=self.device))
        for i in range(0, n, self.batch_size):
            sel = order[i:i + self.batch_size]
            yield self._prepare(self.images[sel], g), self.labels[sel]
        self.epoch += 1


# ------------------------------------------------------------------ public
def get_loaders(dataset_name: str,
                n_per_class: int = None,
                seed: int = 42,
                batch_size: int = None,
                augment: bool = False,
                full_data: bool = False,
                device=None):
    """Returns (train, val, test, meta). Signature unchanged from the CPU version."""
    batch_size = batch_size or config.BATCH_SIZE
    device = device or config.DEVICE

    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])
    n_channels = info["n_channels"]

    splits = {s: DataClass(split=s, download=True, root=config.DATA_CACHE)
              for s in ("train", "val", "test")}
    y = {s: _labels_of(d) for s, d in splits.items()}
    x = {s: _raw_images(d) for s, d in splits.items()}

    rng = np.random.default_rng(seed)

    if full_data:
        cap = config.FULL_DATA_CAP.get(dataset_name)
        tr_idx, tr_counts = _proportional(y["train"], cap, num_classes, rng)
        va_idx = np.arange(len(y["val"]))
        va_counts = np.bincount(y["val"], minlength=num_classes).tolist()
        regime = f"full(cap={cap})" if cap else "full"
    else:
        if n_per_class is None:
            raise ValueError("n_per_class required when full_data=False")
        tr_idx, tr_counts = _stratified(y["train"], n_per_class, num_classes, rng)
        va_idx, va_counts = _stratified(
            y["val"], config.VAL_MULTIPLIER * n_per_class, num_classes, rng)
        regime = f"{n_per_class}/class"

    tr_idx = _drop_singleton(tr_idx, batch_size)

    def make(split, idx, shuffle, aug):
        return GPUBatches(x[split][idx],
                          torch.from_numpy(y[split][idx]).long(),
                          batch_size, shuffle, aug, device, seed, n_channels)

    train = make("train", tr_idx, True, augment)
    val = make("val", va_idx, False, False)
    test = make("test", np.arange(len(y["test"])), False, False)

    meta = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "regime": regime,
        "seed": seed,
        "augment": augment,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(y["test"])),
        "train_class_counts": tr_counts,
        "val_class_counts": va_counts,
        # >1 means model selection sees more labels than training does.
        # Reported, not hidden.
        "val_train_ratio": round(len(va_idx) / max(len(tr_idx), 1), 3),
    }
    return train, val, test, meta


def num_classes_of(dataset_name: str) -> int:
    return len(medmnist.INFO[dataset_name]["label"])
