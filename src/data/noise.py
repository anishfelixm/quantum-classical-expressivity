"""
Analog sensor noise (AWGN).

THE ORDERING IS THE WHOLE POINT
-------------------------------
Noise must be injected in PHYSICAL PIXEL SPACE, not in ImageNet-normalized
space. The conference version clamped normalized tensors to [0,1], which is
not a physical bound - normalized pixels live roughly in [-2.1, 2.6] - so the
clamp silently destroyed signal rather than modelling a sensor floor/ceiling.
Numbers produced under that protocol are not comparable to these.

    1. inverse-normalize   x*sigma_ImageNet + mu_ImageNet  ->  [0,1]
    2. inject              x + eps,  eps ~ N(0, sigma^2)
    3. clamp               [0,1]   (a sensor cannot record negative light)
    4. re-normalize        back into the backbone's input distribution

RNG PARITY
----------
The seed is a deterministic function of sigma, so every architecture faces
bit-identical corrupted tensors. round() rather than int() truncation avoids
float-representation collisions (0.03 -> 29 vs 30).
"""
import torch
import config

_MEAN = torch.tensor(config.NORM_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.NORM_STD).view(1, 3, 1, 1)


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """images: [B,3,H,W] normalized. Returns normalized, noise-corrupted tensor."""
    if sigma == 0.0:
        return images
    mean = _MEAN.to(images.device, images.dtype)
    std = _STD.to(images.device, images.dtype)

    real = images * std + mean                                  # 1
    noisy = real + torch.randn_like(real) * sigma               # 2
    noisy = torch.clamp(noisy, 0.0, 1.0)                        # 3
    return (noisy - mean) / std                                 # 4


def seed_for_sigma(base_seed: int, sigma: float) -> int:
    """Deterministic per-sigma seed, shared across all architectures."""
    return base_seed + int(round(sigma * 1000))
