"""
MedMNIST loader.

WHAT CHANGED FROM THE CONFERENCE VERSION, AND WHY
-------------------------------------------------
1. Scarcity is expressed in ABSOLUTE SHOTS PER CLASS, not fractions.
   1% meant 5 images on BreastMNIST and ~900 on PathMNIST, so a single row of
   a scaling-law figure mixed two unrelated experiments. Absolute counts put
   all four datasets on one comparable axis and match the convention used in
   the medical few-shot literature.

2. Sampling is STRATIFIED. np.random.choice over the flat index range could
   omit whole classes on 8- and 9-class datasets, varying by seed. A model
   cannot learn a class it never saw, and the resulting variance is not a
   property of the architecture.

3. Validation is SUBSAMPLED to match. The old code trained on 54 images and
   selected the best of 100 epochs on 78 validation images - model selection
   consumed more labels than training did, so the claimed regime was not the
   actual regime. Val is capped by availability (breastmnist minority val
   class has only 21), so at large n_per_class the ratio inverts; that is
   reported rather than hidden.

4. Transform order is fixed: augment at 28x28 BEFORE upsampling to 224.
   The old order repeated the grayscale channel before resizing, paying 3x
   the interpolation cost for no benefit.

NOTE ON CLASS BALANCE
---------------------
Equal-n-per-class sampling makes the scarcity grid balanced by construction,
so inverse-frequency class weighting is a no-op there. It only does work in
the full-data reference row, which uses the natural imbalanced split. Both
paths exist deliberately; the methodology says so.
"""
import os
import numpy as np
import torch
import medmnist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import config


def _build_transform(n_channels: int, augment: bool):
    t = [transforms.ToTensor()]
    if augment:
        # Applied at native 28x28: cheaper than at 224 and identical in effect.
        t += [transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomRotation(degrees=10)]
    t += [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True)]
    if n_channels == 1:
        t += [transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
    t += [transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)]
    return transforms.Compose(t)


def _labels_of(dataset) -> np.ndarray:
    return np.asarray(dataset.labels).ravel()


def _stratified_indices(labels, n_per_class, num_classes, rng):
    """Take up to n_per_class from every class. Returns indices and realised counts."""
    idx, realised = [], []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        take = min(n_per_class, len(cls_idx))
        if take > 0:
            idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        realised.append(int(take))
    return np.array(sorted(idx)), realised


def _proportional_indices(labels, cap, num_classes, rng):
    """Cap a full split while preserving its natural class proportions."""
    n = len(labels)
    if cap is None or cap >= n:
        return np.arange(n), np.bincount(labels, minlength=num_classes).tolist()
    idx, realised = [], []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        take = max(1, int(round(len(cls_idx) * cap / n)))
        take = min(take, len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        realised.append(int(take))
    return np.array(sorted(idx)), realised


def _avoid_singleton_batch(indices, batch_size):
    """
    A final batch of exactly one sample makes BatchNorm1d raise. Drop one index
    rather than setting drop_last=True, which would discard scarce data.
    """
    if len(indices) > 2 and len(indices) % batch_size == 1:
        return indices[:-1]
    return indices


def get_loaders(dataset_name: str,
                n_per_class: int = None,
                seed: int = 42,
                batch_size: int = None,
                augment: bool = False,
                full_data: bool = False):
    """
    Returns (train_loader, val_loader, test_loader, meta).

    full_data=True  -> natural imbalanced split, capped per config.FULL_DATA_CAP
    full_data=False -> n_per_class stratified shots; val capped at
                       VAL_MULTIPLIER * n_per_class per class
    """
    batch_size = batch_size or config.BATCH_SIZE
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])
    n_channels = info["n_channels"]
    os.makedirs(config.DATA_CACHE, exist_ok=True)   # medmnist will not create it

    train_tf = _build_transform(n_channels, augment=augment)
    eval_tf = _build_transform(n_channels, augment=False)

    train_ds = DataClass(split="train", transform=train_tf, download=True, root=config.DATA_CACHE)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, root=config.DATA_CACHE)
    test_ds = DataClass(split="test", transform=eval_tf, download=True, root=config.DATA_CACHE)

    rng = np.random.default_rng(seed)
    y_train, y_val = _labels_of(train_ds), _labels_of(val_ds)

    if full_data:
        cap = config.FULL_DATA_CAP.get(dataset_name)
        tr_idx, tr_counts = _proportional_indices(y_train, cap, num_classes, rng)
        va_idx, va_counts = np.arange(len(y_val)), np.bincount(y_val, minlength=num_classes).tolist()
        regime = f"full(cap={cap})" if cap else "full"
    else:
        if n_per_class is None:
            raise ValueError("n_per_class required when full_data=False")
        tr_idx, tr_counts = _stratified_indices(y_train, n_per_class, num_classes, rng)
        va_idx, va_counts = _stratified_indices(
            y_val, config.VAL_MULTIPLIER * n_per_class, num_classes, rng)
        regime = f"{n_per_class}/class"

    tr_idx = _avoid_singleton_batch(tr_idx, batch_size)
    train_ds, val_ds = Subset(train_ds, tr_idx), Subset(val_ds, va_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    meta = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "regime": regime,
        "seed": seed,
        "augment": augment,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_ds)),
        "train_class_counts": tr_counts,
        "val_class_counts": va_counts,
        # val_train_ratio > 1 means model selection sees more labels than
        # training does. Logged so it can be reported honestly.
        "val_train_ratio": round(len(va_idx) / max(len(tr_idx), 1), 3),
    }
    return train_loader, val_loader, test_loader, meta


def num_classes_of(dataset_name: str) -> int:
    return len(medmnist.INFO[dataset_name]["label"])
