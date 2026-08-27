"""
EXPERIMENT 8 - EMPIRICAL LIPSCHITZ CONSTANTS.

WHY THIS EXISTS
---------------
The project's second hypothesis is that the quantum head degrades more
gracefully under input noise. That belief has a concrete mathematical basis,
unlike the expressivity argument:

    v_i(z) = <psi(z)| U^dag X_i U |psi(z)>

U is unitary, so it preserves norms, and the output is bounded in [-1, 1] for
every input and every parameter setting. The map is therefore Lipschitz-bounded
BY CONSTRUCTION. A classical MLP has no such guarantee - its weights can grow
without bound, so an arbitrarily small input perturbation can produce an
arbitrarily large output change.

Measuring the constant turns "the VQC is more robust" from an observation on a
noise curve into a MECHANISM that predicts the curve. That distinction is what
separates a description from an explanation, and reviewers notice it.

WHAT IS MEASURED
----------------
For each head f and many random inputs z in the operating range:

    L_local(z) = max over random unit directions u  of  ||f(z + eps*u) - f(z)|| / eps
    L_hat      = max over sampled z of L_local(z)

Reported: the max (an estimate of the true Lipschitz constant), the mean (typical
sensitivity), and the p95. Also the output range, since a head whose outputs are
bounded cannot produce large swings regardless of its local slope.

TWO REGIMES
-----------
    --at-init   (default) architecture-level property, averaged over parameter
                draws. This is the honest measurement for "bounded BY
                CONSTRUCTION" - it is a claim about the architecture, not about
                one trained model.
    --trained   loads trained heads and measures the realised constants. Slower,
                and reported alongside rather than instead: a reviewer will want
                both.

Cost: forward passes only. Minutes.

USAGE
-----
    python src/08_lipschitz.py
    python src/08_lipschitz.py --dims 4 8 --n-draws 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                      # noqa: E402
import shards                                      # noqa: E402
from models.registry import build_arm              # noqa: E402

EXPERIMENT = "08_lipschitz"

ARMS = ["linear", "mlp", "matched_param", "matched_param_fullrank",
        "low_rank", "fourier_rff",
        "quantum_vqc", "quantum_reupload", "quantum_rich", "quantum_rich_padded"]


@torch.no_grad()
def lipschitz_estimate(head, d, angle_scale, n_points=512, n_directions=32,
                       eps=1e-3, device=None, generator=None):
    """
    Finite-difference estimate of the local Lipschitz constant.

    eps=1e-3 is small enough that the finite difference approximates the
    directional derivative, and large enough to stay well clear of float32
    cancellation. Directions are drawn on the unit sphere rather than along
    coordinate axes: axis-aligned probing would miss the worst direction for
    any head whose Jacobian is not diagonal, which is all of them.
    """
    device = device or config.DEVICE
    head = head.to(device).eval()

    # Inputs span the actual operating range: z_tilde = tanh(z) * angle_scale
    # lives in [-angle_scale, angle_scale]^d.
    z = (torch.rand(n_points, d, generator=generator, device=device) * 2 - 1) * angle_scale

    f0 = head(z)                                            # [N, out]
    out_dim = f0.shape[1]
    worst = torch.zeros(n_points, device=device)

    for _ in range(n_directions):
        u = torch.randn(n_points, d, generator=generator, device=device)
        u = u / u.norm(dim=1, keepdim=True)
        ratio = (head(z + eps * u) - f0).norm(dim=1) / eps
        worst = torch.maximum(worst, ratio)

    # NORMALISED BY OUTPUT WIDTH. ||f(z+eu) - f(z)|| is an L2 norm over out_dim
    # components, so a head that emits 10 numbers has a mechanically larger
    # constant than one emitting 4 even when every component is equally
    # sensitive. quantum_rich emits 10 and quantum_vqc emits 4, so the raw
    # constants are not comparable between them; dividing by sqrt(out_dim)
    # gives per-component sensitivity, which is.
    #
    # Both are reported: the raw value governs how much the LOGITS move (the
    # classifier sees the full vector), the normalised value is the fair
    # architecture-level comparison.
    root_dim = float(out_dim) ** 0.5
    return {
        "lipschitz_max": float(worst.max()),
        "lipschitz_mean": float(worst.mean()),
        "lipschitz_p95": float(worst.quantile(0.95)),
        "lipschitz_max_per_dim": float(worst.max()) / root_dim,
        "lipschitz_mean_per_dim": float(worst.mean()) / root_dim,
        "output_min": float(f0.min()),
        "output_max": float(f0.max()),
        "output_absmax": float(f0.abs().max()),
        "out_dim": int(out_dim),
    }


def run(dims, n_draws, angle_scale, force=False):
    rows = []
    for d in dims:
        for arm in ARMS:
            if arm == "fourier_exact" and d > config.FOURIER_EXACT_MAX_DIM:
                continue
            keys = dict(dim=d, arm=arm, scale=f"{angle_scale:.4f}")
            if not force and shards.exists(EXPERIMENT, **keys):
                continue

            draws = []
            for i in range(n_draws):
                config.set_determinism(1000 + i)
                g = torch.Generator(device=config.DEVICE).manual_seed(1000 + i)
                model = build_arm(arm, d=d, num_classes=2, seed=1000 + i,
                                  build_backbone=False, angle_scale=angle_scale)
                draws.append(lipschitz_estimate(model.head, d, angle_scale,
                                                device=config.DEVICE, generator=g))
                del model
                torch.cuda.empty_cache()

            summary = {k: float(np.mean([x[k] for x in draws])) for k in draws[0]}
            summary["lipschitz_max_over_draws"] = float(
                np.max([x["lipschitz_max"] for x in draws]))
            summary["lipschitz_max_sd"] = float(
                np.std([x["lipschitz_max"] for x in draws]))
            summary["n_draws"] = n_draws

            shards.write(EXPERIMENT, {"metrics": summary, "raw": draws}, **keys)
            rows.append((d, arm, summary))
            print(f"  d={d:2d} {arm:24s} L_max={summary['lipschitz_max']:8.3f} "
                  f"L_mean={summary['lipschitz_mean']:8.3f} "
                  f"|out|max={summary['output_absmax']:.3f}", flush=True)
    return rows


def summarise():
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    tbl = {}
    for r in rows:
        k = r["keys"]
        tbl[(k["dim"], k["arm"])] = r["metrics"]

    print(f"\n=== Empirical Lipschitz constants (architecture-level) ===")
    print("Bounded output + small L => graceful degradation under input noise.")
    print("L/sqrt(out_dim) is the comparable column: heads emit different")
    print("numbers of observables, and an L2 norm over more components is")
    print("mechanically larger even at equal per-component sensitivity.")
    print(f"\n{'d':>3s} {'arm':24s} {'out':>4s} {'L_max':>10s} {'L/sqrt(dim)':>12s} "
          f"{'L_mean':>10s} {'|out|max':>9s}")
    print("-" * 86)
    for (d, arm) in sorted(tbl):
        m = tbl[(d, arm)]
        per = m.get("lipschitz_max_per_dim", float("nan"))
        print(f"{d:3d} {arm:24s} {m.get('out_dim', 0):>4d} {m['lipschitz_max']:10.3f} "
              f"{per:12.3f} {m['lipschitz_mean']:10.3f} {m['output_absmax']:9.3f}")

    # The comparison the paper actually makes
    print(f"\n=== Ratio to quantum_vqc (>1 means the arm is more sensitive) ===")
    for d in sorted({k[0] for k in tbl}):
        base = tbl.get((d, "quantum_vqc"))
        if not base:
            continue
        print(f"\n  d={d}  (quantum_vqc L_max = {base['lipschitz_max']:.3f})")
        for arm in ARMS:
            m = tbl.get((d, arm))
            if not m or arm == "quantum_vqc":
                continue
            raw = m["lipschitz_max"] / base["lipschitz_max"]
            pd = (m.get("lipschitz_max_per_dim", float("nan"))
                  / base.get("lipschitz_max_per_dim", float("nan")))
            print(f"    {arm:24s} raw {raw:6.2f}x   per-dim {pd:6.2f}x")

    print("\nNOTE: measured at initialisation - this is a claim about the")
    print("ARCHITECTURE, not about any trained model. Trained constants are")
    print("recorded per-run by 03_robustness_evaluation.py and reported alongside.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", nargs="+", type=int, default=[4])
    p.add_argument("--n-draws", type=int, default=20)
    p.add_argument("--angle-scale", type=float, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if args.summary_only:
        summarise()
        return

    scale = args.angle_scale if args.angle_scale is not None else float(config.ANGLE_SCALE)
    print(f"Lipschitz measurement | dims={args.dims} draws={args.n_draws} "
          f"angle_scale={scale:.4f} | sha={config.git_sha()[:8]}")
    run(args.dims, args.n_draws, scale, force=args.force)
    summarise()


if __name__ == "__main__":
    main()
