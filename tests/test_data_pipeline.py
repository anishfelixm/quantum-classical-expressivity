"""
Data pipeline tests.

These guard the three loader properties the scarcity claim depends on:
every class present, validation subsampled, sampling reproducible from seed.
Plus the noise round-trip, where an ordering mistake in the conference version
made sigma=0 non-identity.

Run:  python -m pytest tests/test_data_pipeline.py -v -s
"""
import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config                                          # noqa: E402
from data.medmnist_loader import get_loaders           # noqa: E402
from data.noise import add_gaussian_noise, seed_for_sigma  # noqa: E402


# breastmnist is 2-class and tiny; bloodmnist is 8-class - the case where
# non-stratified sampling used to drop classes entirely.
@pytest.mark.parametrize("dataset", ["breastmnist", "bloodmnist"])
@pytest.mark.parametrize("n_per_class", [5, 20])
def test_every_class_present(dataset, n_per_class):
    _, _, _, meta = get_loaders(dataset, n_per_class=n_per_class, seed=42)
    counts = meta["train_class_counts"]
    print(f"\n  {dataset} n={n_per_class}: train={counts} val={meta['val_class_counts']} "
          f"val/train={meta['val_train_ratio']}")
    assert all(c > 0 for c in counts), f"a class was dropped: {counts}"
    assert all(c == n_per_class for c in counts), "class counts are not balanced"


def test_validation_is_subsampled():
    """The bug this guards: selecting on the full val split while training on 10 images."""
    _, _, _, meta = get_loaders("pneumoniamnist", n_per_class=5, seed=42)
    full_val = sum(config.VAL_CLASS_COUNTS["pneumoniamnist"])
    print(f"\n  n_val={meta['n_val']} (full split would be {full_val})")
    assert meta["n_val"] < full_val, "validation was not subsampled"
    assert meta["n_val"] <= 2 * config.VAL_MULTIPLIER * 5


def test_val_cap_by_availability():
    """breastmnist minority val class is 21, so 2*100 is not reachable - must cap, not crash."""
    _, _, _, meta = get_loaders("breastmnist", n_per_class=100, seed=42)
    print(f"\n  breastmnist n=100: val counts={meta['val_class_counts']} "
          f"(available {config.VAL_CLASS_COUNTS['breastmnist']})")
    assert meta["val_class_counts"][0] == 21


def test_sampling_is_seed_reproducible():
    a = get_loaders("breastmnist", n_per_class=10, seed=7)[3]
    b = get_loaders("breastmnist", n_per_class=10, seed=7)[3]
    c = get_loaders("breastmnist", n_per_class=10, seed=8)[3]
    assert a["train_class_counts"] == b["train_class_counts"]
    assert a["n_train"] == b["n_train"] == c["n_train"]


def test_no_singleton_final_batch():
    for n in config.N_PER_CLASS:
        _, _, _, meta = get_loaders("bloodmnist", n_per_class=n, seed=42)
        assert meta["n_train"] % config.BATCH_SIZE != 1, \
            f"n_train={meta['n_train']} leaves a batch of 1; BatchNorm1d will raise"


def test_image_shape_and_channels():
    tr, _, _, _ = get_loaders("breastmnist", n_per_class=5, seed=42)
    x, y = next(iter(tr))
    assert x.shape[1:] == (3, config.IMAGE_SIZE, config.IMAGE_SIZE), x.shape
    assert x.dtype == torch.float32


def test_noise_zero_sigma_is_identity():
    """sigma=0 must be a bit-exact no-op, or the clean baseline is not clean."""
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(add_gaussian_noise(x, 0.0), x)


def test_noise_respects_physical_bounds():
    """
    After the four-step round trip, de-normalized pixels must lie in [0,1].
    The conference version clamped in NORMALIZED space, where [0,1] is not a
    physical bound - that clamp destroyed signal instead of modelling a sensor.
    """
    mean = torch.tensor(config.NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.NORM_STD).view(1, 3, 1, 1)
    x_real = torch.rand(8, 3, 32, 32)
    x = (x_real - mean) / std

    out_real = add_gaussian_noise(x, 0.5) * std + mean
    print(f"\n  de-normalized range after sigma=0.5: "
          f"[{out_real.min():.4f}, {out_real.max():.4f}]")
    assert out_real.min() >= -1e-5 and out_real.max() <= 1 + 1e-5


def test_noise_is_reproducible_across_models():
    """RNG parity: every arm must face bit-identical corrupted tensors."""
    x = torch.randn(4, 3, 32, 32)
    for sigma in (0.03, 0.10):
        torch.manual_seed(seed_for_sigma(42, sigma))
        a = add_gaussian_noise(x, sigma)
        torch.manual_seed(seed_for_sigma(42, sigma))
        b = add_gaussian_noise(x, sigma)
        assert torch.equal(a, b), f"noise not reproducible at sigma={sigma}"


def test_sigma_seeds_are_distinct():
    """int() truncation used to collide 0.03 -> 29 and 0.030000001 -> 30."""
    seeds = [seed_for_sigma(42, s) for s in config.NOISE_LEVELS]
    assert len(set(seeds)) == len(seeds), f"colliding sigma seeds: {seeds}"


if __name__ == "__main__":
    test_every_class_present("bloodmnist", 5)
    test_validation_is_subsampled()
    test_val_cap_by_availability()
    test_noise_zero_sigma_is_identity()
    test_noise_respects_physical_bounds()
    print("\nData pipeline checks passed.")
