"""
Print, rather than only assert, the numbers that tests/test_independent_theory.py
checks, so the manuscript can cite a log instead of a threshold.

    python scripts/print_independent_theory.py | tee logs_independent_theory.txt

Numpy only. Uses the test module's own circuit, data and weights (seeded), so
these are exactly the quantities the tests assert on.

Added 15 October 2026. The 24-of-1,770 reordering figure quoted in several
documents could not be traced to any log, and a recomputation of the test's own
quantity on another machine gave a different count. This prints it. It also
reports the same count over 1,000 random linear readouts, because one readout
vector is an arbitrary choice and the count depends on it.
"""
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import test_independent_theory as T   # noqa: E402

print(f"numpy {np.__version__}; d={T.D}, L={T.L}, {len(T._Z)} inputs, weights ~ N(0,1) seed 0")

print(f"1. single-encoding <X_i> in the 3^d span: max residual {T.resid(T.basis(T._Z, 1), T._V):.3e}")
bad = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [3, 0, 0, 0], [2, 2, 0, 0]], float)
wrong = np.concatenate([np.cos(T._Z @ bad.T), np.sin(T._Z @ bad.T)], 1)
print(f"   wrong-frequency basis: max residual {T.resid(wrong, T._V):.3f} (must be large)")

Vp = np.array([T.run(z, T._W, T.D, T.L, pairs=True) for z in T._Z[:200]])
print(f"2. two-qubit readout in the 3^d span: max residual {T.resid(T.basis(T._Z[:200], 1), Vp):.3e}")

Vr = np.array([T.run(z, T._W, T.D, T.L, R=2) for z in T._Z])
print(f"3. re-uploading in the 5^d span: max residual {T.resid(T.basis(T._Z, 2), Vr):.3e}; "
      f"in the 3^d span: {T.resid(T.basis(T._Z, 1), Vr):.3f} (must be large)")

p = 0.05
Ve = np.array([T.run(z, T._W, T.D, T.L, p=p, where="end") for z in T._Z[:40]])
print(f"4. end-of-circuit depolarising, p={p}: max |noisy - (1-4p/3) x clean| = "
      f"{np.abs(Ve - (1 - 4 * p / 3) * T._V[:40]).max():.3e}  (1-4p/3 = {1 - 4 * p / 3:.4f})")

Vl = np.array([T.run(z, T._W, T.D, T.L, p=p, where="layer") for z in T._Z[:60]])
ratio = Vl / T._V[:60]
pairs = list(combinations(range(60), 2))


def flips(w):
    s0, s1 = T._V[:60] @ w, Vl @ w
    return sum(1 for i, j in pairs if (s0[i] - s0[j]) * (s1[i] - s1[j]) < 0)


w_test = np.random.default_rng(1).normal(size=T.D)
print(f"5. depolarising after every layer, p={p}: per-output contraction ratio "
      f"{np.nanmin(ratio):.3f} to {np.nanmax(ratio):.3f} over 60 inputs")
print(f"   reordered pairs for the test's readout vector (seed 1): {flips(w_test)} of {len(pairs)}")
rng = np.random.default_rng(2026)
counts = np.array([flips(rng.normal(size=T.D)) for _ in range(1000)])
print(f"   over 1,000 random readout vectors (seed 2026): median {int(np.median(counts))}, "
      f"5th-95th percentile {int(np.percentile(counts, 5))}-{int(np.percentile(counts, 95))}, "
      f"max {counts.max()}, zero in {int((counts == 0).sum())} of 1000")
