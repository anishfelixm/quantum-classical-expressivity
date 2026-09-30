"""
Verify the v37 changes BEFORE any long run. Every check must print PASS.

    python scripts/verify_v37.py

WHY THIS EXISTS. The changes touch model construction (registry, quantum_vqc),
the data path every experiment uses (medmnist_loader), and parity - the
property the whole paper rests on. A silent change to any of them would
invalidate a multi-day sweep, so each is checked directly.

    1. Parity, the project rule: quantum_vqc, matched_param_fullrank and
       low_rank(rank=2) must all report 24 head parameters at d=4.
    2. Parity for the ansatz check: quantum_basic must match quantum_vqc
       exactly (24/48/96) and share its 3^d spectrum.
    3. The loader refactor is BIT-IDENTICAL: _prepare must equal the old
       implementation exactly, or every existing result changes meaning.
    4. Native-resolution noise: sigma=0 must equal the clean path exactly;
       the same seed must give identical corruption (every arm sees the same
       images); grayscale noise must be identical across the replicated
       channels (one draw per pixel, as a single-channel sensor produces).
    5. Config constants that the amendments depend on.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config                                                   # noqa: E402
from data.medmnist_loader import GPUBatches                     # noqa: E402
from models.registry import build_arm                           # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def head_params(arm, d, **kw):
    m = build_arm(arm, d=d, num_classes=2, seed=42, build_backbone=False, **kw)
    return sum(p.numel() for p in m.head.parameters() if p.requires_grad)


# ------------------------------------------------------------------ 1-2 parity
print("\n1. Parity (project rule)")
for arm, kw in (("quantum_vqc", {}), ("matched_param_fullrank", {}),
                ("low_rank", {"head_rank": 2})):
    n = head_params(arm, 4, **kw)
    check(f"{arm:24s} d=4 -> 24", n == 24, f"got {n}")

print("\n2. Parity for the ansatz check")
for d, want in ((4, 24), (8, 48), (16, 96)):
    a, b = head_params("quantum_vqc", d), head_params("quantum_basic", d)
    check(f"quantum_basic == quantum_vqc at d={d} ({want})",
          a == b == want, f"vqc={a} basic={b}")
x = build_arm("quantum_basic", d=4, num_classes=2, seed=42, build_backbone=False).head.describe()
y = build_arm("quantum_vqc", d=4, num_classes=2, seed=42, build_backbone=False).head.describe()
check("quantum_basic: 6 circuit layers, same spectrum as quantum_vqc",
      x["circuit_layers"] == 6 and x["spectrum_size"] == y["spectrum_size"] == 81,
      f"layers={x['circuit_layers']} spectrum={x['spectrum_size']}")

# ------------------------------------------------------------------ 3-4 loader
print("\n3-4. Loader: refactor is bit-identical; noise is native-resolution")
dev = config.DEVICE
g = torch.Generator().manual_seed(0)
mean = torch.tensor(config.NORM_MEAN, device=dev).view(1, 3, 1, 1)
std = torch.tensor(config.NORM_STD, device=dev).view(1, 3, 1, 1)


def old_prepare(imgs_u8, n_channels):
    """Exact copy of the pre-v36 _prepare, augment off."""
    x = imgs_u8.float().div_(255.0)
    x = F.interpolate(x, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                      mode="bilinear", align_corners=False)
    if n_channels == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x - mean) / std


for C in (1, 3):
    imgs = torch.randint(0, 256, (40, C, 28, 28), generator=g, dtype=torch.uint8)
    labels = torch.randint(0, 2, (40,), generator=g)
    b = GPUBatches(imgs, labels, batch_size=16, shuffle=False, augment=False,
                   device=dev, seed=1, n_channels=C)

    new = torch.cat([x for x, _ in b])
    ref = old_prepare(imgs.to(dev), C)
    check(f"C={C}: _prepare bit-identical to the pre-v36 code",
          torch.equal(new, ref), f"max|diff|={float((new - ref).abs().max()):.1e}")

    clean0 = torch.cat([x for x, _ in b.iter_with_sensor_noise(0.0, 123)])
    check(f"C={C}: sigma=0 equals the clean path exactly", torch.equal(clean0, ref))

    n1 = torch.cat([x for x, _ in b.iter_with_sensor_noise(0.1, 123)])
    n2 = torch.cat([x for x, _ in b.iter_with_sensor_noise(0.1, 123)])
    n3 = torch.cat([x for x, _ in b.iter_with_sensor_noise(0.1, 124)])
    check(f"C={C}: same seed -> identical corruption (RNG parity across arms)",
          torch.equal(n1, n2))
    check(f"C={C}: different seed -> different corruption", not torch.equal(n1, n3))

    if C == 1:
        phys = n1 * std + mean          # back to [0,1] physical space
        spread = float((phys - phys[:, :1]).abs().max())
        check("C=1: one noise draw per pixel, shared by the 3 replicated channels",
              spread < 1e-5, f"max channel difference {spread:.1e}")

    # Native-resolution noise, once upsampled, is spatially smooth: neighbouring
    # 224-px pixels inside one native pixel are strongly correlated. Noise added
    # after upsampling (v1) would make them independent.
    diff = (n1 - ref)[:, 0]
    corr = torch.corrcoef(torch.stack([diff[:, :, :-1].flatten(),
                                       diff[:, :, 1:].flatten()]))[0, 1]
    check(f"C={C}: noise field is spatially correlated (injected at 28x28)",
          float(corr) > 0.8, f"neighbour correlation {float(corr):.3f}")

# ------------------------------------------------------------------ 5 config
print("\n5. Config")
check("DECLARED_FAMILY_SIZE == 23", config.DECLARED_FAMILY_SIZE == 23,
      str(config.DECLARED_FAMILY_SIZE))
need = {"quantum_reupload", "quantum_rich", "quantum_rich_padded",
        "quantum_basic", "low_rank"}
check("LR_INHERITANCE covers every untuned arm", need <= set(config.LR_INHERITANCE),
      str(sorted(need - set(config.LR_INHERITANCE))))
check("quantum_basic registered as a quantum arm", "quantum_basic" in config.QUANTUM_ARMS)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
