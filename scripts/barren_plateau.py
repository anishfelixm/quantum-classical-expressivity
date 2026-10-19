"""
Amendment 24 (written 17 October 2026, before the run): gradient variance of
the VQC head at initialisation - the barren-plateau diagnostic.

    python scripts/barren_plateau.py            # ~10-20 min on CPU
    python scripts/barren_plateau.py --quick    # d = 4 only, 20 draws

WHY
---
docs/MATH_VERIFICATION.md section 7 printed a table of "quantum-weight gradient
variance" for d in {4, 8, 16} and L in {1, 2, 4}. No script in the repository
produced it, and the quantity the training loop logs (quantum_grad_var) is the
variance ACROSS a circuit's parameters within one step, which is not the
barren-plateau quantity. That table is withdrawn and replaced by this
measurement.

WHAT IS MEASURED
----------------
The barren-plateau quantity (McClean et al. 2018): the variance, over random
initialisations, of the partial derivatives of a cost with respect to the
circuit parameters.
- Head: the project's own VQCHead (src/models/quantum_vqc.py, current commit),
  single encoding, StronglyEntanglingLayers, <X_i> readout, default.qubit,
  backpropagation.
- Grid: d in {4, 8, 16} qubits x L in {1, 2, 4} layers.
- Initialisations, 200 draws each (torch seeds 0..199):
    "project"  N(0, 0.1^2) - the initialisation every reported run used;
    "uniform"  U[0, 2*pi) - the standard setting of the barren-plateau literature.
- Inputs: one fixed batch of 32 latents z ~ U[-pi/2, pi/2]^d (numpy seed 12345),
  the head's operating range (tanh * pi/2).
- Cost: C = mean over the batch of the head's first output, <X_0>.
- Reported per (d, L, init): the variance over draws of each parameter's
  derivative, averaged over the 3Ld parameters; and the variance of the
  derivative with respect to one fixed parameter (first layer, wire 0, second
  rotation angle).

PREDICTIONS
-----------
 P1. Under uniform initialisation, the mean variance at d = 16 is below that at
     d = 4, at L = 4.
 P2. At every (d, L), the project's initialisation gives a larger mean variance
     than uniform initialisation.
Exploratory. Interpretive use only: whether d = 16 results could be limited by
trainability.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeat                                          # noqa: E402

LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--draws", type=int, default=200)
    a = p.parse_args()
    codeat.setup(None)
    import numpy as np
    import torch
    import config
    from models.quantum_vqc import VQCHead

    dims, layers, draws = ([4], [1, 2, 4], 20) if a.quick else ([4, 8, 16], [1, 2, 4], a.draws)
    torch.set_default_dtype(torch.float32)
    res = {}
    out(f"Amendment 24 - gradient variance at initialisation | sha={config.git_sha()[:8]} | "
        f"draws={draws} | CPU")
    out(f"{'d':>3s} {'L':>3s} {'init':8s} {'mean Var dC/dtheta':>20s} {'Var dC/dtheta[0,0,1]':>22s} "
        f"{'seconds':>8s}")
    for d in dims:
        z = torch.as_tensor((np.random.default_rng(12345).random((32, d)) * 2 - 1) * (math.pi / 2),
                            dtype=torch.float32)
        for L in layers:
            for init in ("project", "uniform"):
                t0 = time.time()
                grads = []
                for s in range(draws):
                    torch.manual_seed(s)
                    head = VQCHead(d, n_layers=L, n_uploads=1)       # N(0, 0.1^2) at construction
                    w = head.q_layer.weights
                    if init == "uniform":
                        with torch.no_grad():
                            w.uniform_(0.0, 2 * math.pi, generator=torch.Generator().manual_seed(s))
                    head.zero_grad()
                    c = head(z)[:, 0].mean()
                    c.backward()
                    grads.append(w.grad.detach().double().reshape(-1).numpy().copy())
                G = np.stack(grads)                                   # [draws, 3Ld]
                var = G.var(axis=0, ddof=1)
                one = float(G[:, 1].var(ddof=1))
                res[f"d{d}_L{L}_{init}"] = {"mean_var": float(var.mean()), "var_param_0_0_1": one,
                                            "n_params": int(G.shape[1]), "draws": draws}
                out(f"{d:3d} {L:3d} {init:8s} {var.mean():20.3e} {one:22.3e} {time.time() - t0:8.1f}")
    if not a.quick:
        p1 = res["d16_L4_uniform"]["mean_var"] < res["d4_L4_uniform"]["mean_var"]
        p2 = all(res[f"d{d}_L{L}_project"]["mean_var"] > res[f"d{d}_L{L}_uniform"]["mean_var"]
                 for d in dims for L in layers)
        out(f"\nP1 {'HOLDS' if p1 else 'FAILS'}: uniform init, L=4: d=16 "
            f"{res['d16_L4_uniform']['mean_var']:.3e} vs d=4 {res['d4_L4_uniform']['mean_var']:.3e}")
        out(f"P2 {'HOLDS' if p2 else 'FAILS'}: project init above uniform at every (d, L)")
    path = os.path.join(config.ARTIFACT_ROOT, "barren_plateau.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    dst = os.path.join(os.getcwd(), "logs_barren_plateau.txt" if not a.quick else "logs_barren_plateau_quick.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst} and {path}")


if __name__ == "__main__":
    main()
