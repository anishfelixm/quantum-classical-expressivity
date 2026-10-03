"""
Verify the v40 changes BEFORE the follow-up runs. Every line must print PASS.

    python scripts/verify_v40.py

    1. Parity, the project rule: quantum_vqc, matched_param_fullrank and
       low_rank(rank=2) report 24 head parameters at d=4.
    2. The EXISTING fourier_rff arm is bit-identical to the pre-v40 code: same
       frequencies, same count, at d = 4, 8, 16 and several seeds. Its shards
       already exist in the confirmatory namespace; a silent change would make
       H-S2 refer to a different model than the one that produced them.
    3. fourier_rff_r2 has the intended shape: all 312 canonical frequencies of
       {-2..2}^4, 624 features, 2,500 parameters.
    4. fourier_rff_r2 genuinely spans the re-uploading circuit's function class:
       the circuit's output, at random circuit parameters, is fitted by the
       {-2..2}^d features to float precision - and NOT by the {-1,0,1}^d
       features of the original arm, so the test cannot pass vacuously.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config                                                   # noqa: E402
from models.classical_fourier import FourierRFFHead             # noqa: E402
from models.registry import build_arm                           # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def head_params(arm, d, **kw):
    m = build_arm(arm, d=d, num_classes=2, seed=42, build_backbone=False, **kw)
    return sum(p.numel() for p in m.head.parameters() if p.requires_grad)


# ------------------------------------------------------------------ 1
print("\n1. Parity (project rule)")
for arm, kw in (("quantum_vqc", {}), ("matched_param_fullrank", {}),
                ("low_rank", {"head_rank": 2})):
    n = head_params(arm, 4, **kw)
    check(f"{arm:24s} d=4 -> 24", n == 24, f"got {n}")

# ------------------------------------------------------------------ 2
print("\n2. Existing fourier_rff arm unchanged")


def _canon(w):
    for x in w:
        if x > 0:
            return True
        if x < 0:
            return False
    return False


def reference_omega(d, seed, max_features=2048):
    """Exact copy of the pre-v40 frequency construction."""
    n_canonical = (3 ** d - 1) // 2
    m = max(1, min(n_canonical, max_features // 2))
    g = torch.Generator().manual_seed(seed)
    if d <= 8:
        allf = torch.tensor([w for w in itertools.product([-1, 0, 1], repeat=d)
                             if _canon(w)], dtype=torch.float32)
        return allf[torch.randperm(allf.shape[0], generator=g)[:m]]
    seen, out = set(), []
    while len(out) < m:
        for row in torch.randint(-1, 2, (4 * m, d), generator=g):
            if len(out) >= m:
                break
            w = tuple(int(x) for x in row)
            if not _canon(w):
                w = tuple(-x for x in w)
                if not _canon(w):
                    continue
            if w not in seen:
                seen.add(w)
                out.append(w)
    return torch.tensor(out, dtype=torch.float32)


for d in (4, 8, 16):
    same = all(torch.equal(FourierRFFHead(d, seed=s).omega, reference_omega(d, s))
               for s in (42, 123, 2026))
    check(f"d={d}: frequencies bit-identical to pre-v40 code (3 seeds)", same)

# ------------------------------------------------------------------ 3
print("\n3. fourier_rff_r2 shape")
h = build_arm("fourier_rff_r2", d=4, num_classes=2, seed=42, build_backbone=False).head
n_par = sum(p.numel() for p in h.parameters() if p.requires_grad)
check("312 canonical frequencies of {-2..2}^4", h.omega.shape == (312, 4),
      str(tuple(h.omega.shape)))
check("624 features, 2,500 head parameters", h.n_features == 624 and n_par == 2500,
      f"features={h.n_features} params={n_par}")
check("frequencies lie in {-2..2} and include |w|=2",
      float(h.omega.abs().max()) == 2.0 and float(h.omega.abs().min()) == 0.0)

# ------------------------------------------------------------------ 4
print("\n4. fourier_rff_r2 spans the re-uploading circuit's output")
torch.manual_seed(0)
qh = build_arm("quantum_reupload", d=4, num_classes=2, seed=42, build_backbone=False).head
with torch.no_grad():
    for p in qh.parameters():
        p.uniform_(-np.pi, np.pi)          # a generic circuit, not near-identity
# 3,000 points against 625 basis functions: far more rows than columns, so a
# small residual is evidence of membership in the span rather than an
# underdetermined fit (with fewer points than columns, ANY function fits).
z = (torch.rand(3000, 4) * 2 - 1) * (np.pi / 2)
with torch.no_grad():
    v = qh(z).double().numpy()                                     # [3000, 4]


def residual(omega):
    P = z.double().numpy() @ omega.double().numpy().T
    B = np.hstack([np.ones((len(z), 1)), np.cos(P), np.sin(P)])
    coef, *_ = np.linalg.lstsq(B, v, rcond=None)
    return float(np.abs(B @ coef - v).max())


r2 = residual(h.omega)
r1 = residual(FourierRFFHead(4, seed=42).omega)
check("re-upload output lies in the {-2..2}^d span", r2 < 1e-4, f"max residual {r2:.2e}")
check("and NOT in the {-1,0,1}^d span (test is not vacuous)", r1 > 100 * max(r2, 1e-7),
      f"max residual {r1:.2e}")

# ------------------------------------------------------------------ 5
print("\n5. Config")
check("fourier_rff_r2 registered", "fourier_rff_r2" in config.ARMS)
check("fourier_rff_r2 inherits fourier_rff's rate",
      config.LR_INHERITANCE.get("fourier_rff_r2") == "fourier_rff")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
