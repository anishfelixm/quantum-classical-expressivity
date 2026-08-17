"""
EXPERIMENT 3 (Q5) - ROBUSTNESS UNDER ANALOG SENSOR NOISE.

This is the SECOND pillar of the original hypothesis, and until now it had zero
runs. The first pillar - "a quantum circuit accesses more states, so it should
be more expressive" - is closed: the measured output lies in an explicitly
constructible 3^d classical trigonometric span, and with 3Ld parameters the
circuit reaches only a small manifold inside it.

The noise pillar is different, and it has real mathematics behind it:

    v_i(z) = <psi(z)| U^dag X_i U |psi(z)>

U is unitary, so norms are preserved and the output is bounded in [-1, 1] for
every input and every parameter setting. The map is Lipschitz-bounded BY
CONSTRUCTION. A classical MLP is not: its weights can grow without bound.

So there is a principled reason to expect the quantum head to degrade more
gracefully - and this script tests it.

WHY IT RETRAINS RATHER THAN LOADING CHECKPOINTS
------------------------------------------------
Experiment 1 discards `best_state` after training, so no checkpoints exist on
disk. Rather than re-run Experiment 1 to produce them, this script trains the
head itself. That is cheap in the frozen regime, because training uses CACHED
256-d features (seconds for classical arms, ~2.5 min for quantum).

Noise, however, must be injected on IMAGES, before the backbone - corrupting
cached features would model a different and physically meaningless process. So
after training on cached features the head is transplanted into a full model
with the backbone attached, and the noise sweep runs end to end. The backbone is
frozen and bit-exactly deterministic (verified), so the transplant is exact, not
an approximation.

WHAT IS REPORTED AT EVERY NOISE LEVEL
-------------------------------------
AUC, Macro-F1, balanced accuracy, ECE and predicted-probability spread.

Reporting all of them is the point. The conference paper's "Zombie State" - AUC
0.6118 with probability standard deviation 0.0057 - is the signature of a
CALIBRATION failure, not a decision-boundary failure. If AUC holds while F1
collapses, the model still ranks cases correctly and only the threshold has
drifted. That is a narrower and much more accurate claim than "the quantum model
shatters", and it is invisible if you only plot F1.

Per-sample probabilities are saved as .npy so the confirmatory analysis can run
the nested bootstrap the pre-registration specifies. Bootstrapping over test
indices is impossible from scalar metrics alone.

RNG PARITY
----------
The seed is a deterministic function of sigma, so every architecture faces
bit-identical corrupted tensors. Without this, arm differences would be confounded
with noise draws.

USAGE
-----
    python src/03_robustness_evaluation.py --quick     # 2 datasets, n in {5,100}
    python src/03_robustness_evaluation.py             # full
    python src/03_robustness_evaluation.py --summary-only
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import get_loaders, num_classes_of    # noqa: E402
from data.noise import add_gaussian_noise, seed_for_sigma       # noqa: E402
from models.registry import build_arm                           # noqa: E402
from train.loop import train_model                              # noqa: E402
from train.metrics import compute_metrics                       # noqa: E402

# 01 exposes the feature cache; reuse it rather than re-extracting.
_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features
FROZEN = _exp1.FROZEN

EXPERIMENT = "03_robustness"

ARMS = ["linear", "matched_param_fullrank", "fourier_rff", "quantum_vqc"]
PRED_DIR = os.path.join(config.ARTIFACT_ROOT, "predictions", EXPERIMENT)
os.makedirs(PRED_DIR, exist_ok=True)


@torch.no_grad()
def lipschitz_of_trained(head, d, angle_scale, device, n_points=256,
                         n_directions=16, eps=1e-3):
    """Realised Lipschitz constant of the TRAINED head. Pairs with 08_lipschitz."""
    head = head.eval()
    z = (torch.rand(n_points, d, device=device) * 2 - 1) * angle_scale
    f0 = head(z)
    worst = torch.zeros(n_points, device=device)
    for _ in range(n_directions):
        u = torch.randn(n_points, d, device=device)
        u = u / u.norm(dim=1, keepdim=True)
        worst = torch.maximum(worst, (head(z + eps * u) - f0).norm(dim=1) / eps)
    return {"lipschitz_max": float(worst.max()),
            "lipschitz_mean": float(worst.mean())}


@torch.no_grad()
def sweep_noise(model, test_batches, num_classes, seed, device):
    """
    Evaluate across NOISE_LEVELS. Returns per-sigma metrics plus the per-sample
    probabilities at every sigma (needed for the nested bootstrap).
    """
    model.eval()
    curve, preds_by_sigma = {}, {}

    for sigma in config.NOISE_LEVELS:
        torch.manual_seed(seed_for_sigma(seed, sigma))
        probs, preds, labels = [], [], []

        for x, y in test_batches:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)
            p = torch.softmax(model(add_gaussian_noise(x, sigma)), dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())

        probs = np.concatenate(probs)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        curve[f"{sigma:.2f}"] = compute_metrics(labels, preds, probs, num_classes)
        preds_by_sigma[f"{sigma:.2f}"] = probs.astype(np.float16)   # half is plenty

    return curve, preds_by_sigma, labels


def run_cell(dataset, n_per_class, seed, arm, dim=4, force=False):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed, arm=arm)
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    device = config.DEVICE
    C = num_classes_of(dataset)
    t0 = time.time()

    # --- train the head on cached CLEAN features (fast) --------------------
    loaders, _ = get_cached_features(dataset, str(n_per_class), seed)
    head_model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                           seed=seed, build_backbone=False)
    _, _, best_state = train_model(
        head_model, loaders["train"], loaders["val"], loaders["test"],
        num_classes=C, use_features=True,
        is_quantum=arm.startswith("quantum"), verbose=False)
    if best_state is not None:
        head_model.load_state_dict(best_state, strict=False)

    lip = lipschitz_of_trained(head_model.head, dim, head_model.angle_scale, device)

    # --- transplant into a full model and sweep noise on IMAGES ------------
    full = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                     seed=seed, freeze_policy=FROZEN, build_backbone=True).to(device)
    full.load_state_dict(head_model.state_dict(), strict=False)

    _, _, test_batches, meta = get_loaders(dataset, n_per_class=n_per_class,
                                           seed=seed, augment=False)
    curve, preds_by_sigma, labels = sweep_noise(full, test_batches, C, seed, device)

    tag = f"{dataset}__n{n_per_class}__d{dim}__s{seed}__{arm}"
    np.savez_compressed(os.path.join(PRED_DIR, tag + ".npz"),
                        labels=labels.astype(np.int16), **preds_by_sigma)

    clean = curve["0.00"]
    payload = {
        "curve": curve,
        "lipschitz_trained": lip,
        "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test", "regime")},
        "predictions_file": tag + ".npz",
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del head_model, full
    torch.cuda.empty_cache()
    return clean, curve, lip


# ------------------------------------------------------------------ summary
def _paired(a, b):
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common
                  if a[s] is not None and b[s] is not None])
    if len(d) < 2:
        return None
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, m - 1.96 * se, m + 1.96 * se, len(d)


def summarise(metric="auc"):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    tbl, lip = {}, {}
    for r in rows:
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]))
        for s, m in r["curve"].items():
            tbl.setdefault((cell, s), {}).setdefault(k["arm"], {})[k["seed"]] = m.get(metric)
        lip.setdefault(k["arm"], []).append(r["lipschitz_trained"]["lipschitz_max"])

    sigmas = [f"{s:.2f}" for s in config.NOISE_LEVELS]
    cells = sorted({c for (c, _) in tbl})

    print(f"\n=== {metric.upper()} vs noise (mean over seeds) ===")
    for cell in cells:
        print(f"\n  {cell[0]}  n={cell[1]}/class")
        print(f"    {'arm':24s} " + " ".join(f"{s:>7s}" for s in sigmas))
        for arm in ARMS:
            vals = []
            for s in sigmas:
                v = [x for x in tbl.get((cell, s), {}).get(arm, {}).values() if x is not None]
                vals.append(f"{np.mean(v):7.4f}" if v else "      -")
            print(f"    {arm:24s} " + " ".join(vals))

    # --- retention: the actual robustness question ------------------------
    print(f"\n=== Retention at sigma=0.20 ({metric} at 0.20 / {metric} at 0.00) ===")
    print("Higher = degrades more gracefully. This, not raw score, is the claim.")
    for cell in cells:
        line = []
        for arm in ARMS:
            c0 = tbl.get((cell, "0.00"), {}).get(arm, {})
            c2 = tbl.get((cell, "0.20"), {}).get(arm, {})
            common = sorted(set(c0) & set(c2))
            r = [c2[s] / c0[s] for s in common
                 if c0.get(s) and c2.get(s) and c0[s] > 0]
            line.append(f"{arm}={np.mean(r):.3f}" if r else f"{arm}=-")
        print(f"  {cell[0]:16s} n={cell[1]:>4s}  " + "  ".join(line))

    # --- calibration vs boundary failure ----------------------------------
    print(f"\n=== Is F1 collapse a CALIBRATION failure? ===")
    print("relative loss = 1 - metric(0.20)/metric(0.00). If F1 loss >> AUC loss,")
    print("ranking survives and only the decision threshold drifted.")
    f1_tbl = {}
    for r in rows:
        k = r["keys"]
        for s, m in r["curve"].items():
            f1_tbl.setdefault(((k["dataset"], str(k["regime"])), s), {}) \
                  .setdefault(k["arm"], {})[k["seed"]] = m.get("macro_f1")
    for cell in cells:
        for arm in ARMS:
            a0 = [v for v in tbl.get((cell, "0.00"), {}).get(arm, {}).values() if v]
            a2 = [v for v in tbl.get((cell, "0.20"), {}).get(arm, {}).values() if v]
            f0 = [v for v in f1_tbl.get((cell, "0.00"), {}).get(arm, {}).values() if v]
            f2 = [v for v in f1_tbl.get((cell, "0.20"), {}).get(arm, {}).values() if v]
            if not (a0 and a2 and f0 and f2):
                continue
            la = 1 - np.mean(a2) / np.mean(a0)
            lf = 1 - np.mean(f2) / np.mean(f0)
            flag = "  <-- calibration" if lf > 2 * la and la < 0.15 else ""
            print(f"  {cell[0]:16s} n={cell[1]:>4s} {arm:24s} "
                  f"AUC loss {la:6.3f} | F1 loss {lf:6.3f}{flag}")

    print(f"\n=== Trained Lipschitz constants (max over runs) ===")
    for arm in ARMS:
        if arm in lip:
            print(f"  {arm:24s} L_max mean={np.mean(lip[arm]):8.3f} "
                  f"max={np.max(lip[arm]):8.3f}")

    print("\nEXPLORATORY. Confirmatory statistics come from 04_statistical_analysis.py")
    print("using the saved per-sample predictions.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", type=int, default=[5, 20, 100])
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--quick", action="store_true",
                   help="2 datasets x n in {5,100} x 5 seeds")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--metric", default="auc")
    args = p.parse_args()

    if args.summary_only:
        summarise(args.metric)
        return

    if args.quick:
        args.datasets = ["breastmnist", "pneumoniamnist"]
        args.regimes = [5, 100]
        args.seeds = config.ALL_SEEDS[:5]

    total = (len(args.datasets) * len(args.regimes)
             * len(args.arms) * len(args.seeds))
    print(f"Q5 robustness | {total} cells | {len(config.NOISE_LEVELS)} noise levels "
          f"| device={config.DEVICE} | sha={config.git_sha()[:8]}")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for n in args.regimes:
            for seed in args.seeds:
                for arm in args.arms:
                    done += 1
                    out = run_cell(ds, n, seed, arm, dim=args.dim, force=args.force)
                    if out is None:
                        continue
                    clean, curve, lip = out
                    eta = (time.time() - t0) / done * (total - done) / 3600
                    print(f"[{done}/{total}] {ds} n={n} s={seed} {arm:24s} "
                          f"auc {clean['auc']:.4f}->{curve['0.20']['auc']:.4f} "
                          f"f1 {clean['macro_f1']:.4f}->{curve['0.20']['macro_f1']:.4f} "
                          f"L={lip['lipschitz_max']:.2f} ETA {eta:.1f}h", flush=True)

    summarise(args.metric)


if __name__ == "__main__":
    main()
