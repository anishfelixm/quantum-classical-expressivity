"""
EXPERIMENT 3 (Q5) - ROBUSTNESS UNDER ANALOG SENSOR NOISE.

This is the SECOND pillar of the original hypothesis. The first pillar - "a
quantum circuit accesses more states, so it should be more expressive" - is
closed: the measured output lies in an explicitly constructible 3^d classical
trigonometric span, and with 3Ld parameters the circuit reaches only a small
manifold inside it.

The noise pillar is different, and it has real mathematics behind it:

    v_i(z) = <psi(z)| U^dag X_i U |psi(z)>

U is unitary, so norms are preserved and the output is bounded in [-1, 1] for
every input and every parameter setting. The map is Lipschitz-bounded BY
CONSTRUCTION. A classical MLP is not: its weights can grow without bound.

WHY IT RETRAINS RATHER THAN LOADING CHECKPOINTS
------------------------------------------------
Experiment 1 does not persist weights to disk, so this script trains the head
itself. That is cheap in the frozen regime: training uses CACHED 256-d features.

Noise, however, must be injected on IMAGES, before the backbone - corrupting
cached features would model a different and physically meaningless process. So
after training on cached features the head is transplanted into a full model
with the backbone attached, and the noise sweep runs end to end. The backbone is
frozen and bit-exactly deterministic (verified), so the transplant is exact.

WHAT IS REPORTED AT EVERY NOISE LEVEL
-------------------------------------
AUC, Macro-F1, balanced accuracy, average precision, sensitivity, specificity,
ECE and predicted-probability spread.

Reporting all of them is the point. The conference paper's "Zombie State" - AUC
0.6118 with probability standard deviation 0.0057 - is the signature of a
CALIBRATION failure, not a decision-boundary failure. If AUC holds while F1
collapses, the model still ranks cases correctly and only the threshold has
drifted. That is a narrower and much more accurate claim than "the quantum model
shatters", and it is invisible if you only plot F1.

PREDICTION FILES
----------------
Written through shards.save_predictions() and read back by 04 through
shards.load_predictions(), keyed identically. This script previously built its
own filename convention that 04's reader did not match, so the nested bootstrap
silently fell back to seed-level resampling. One writer, one reader, one naming
function.

Multi-sigma runs store one array per noise level under the `condition` argument,
so 04 can bootstrap at any sigma.

RNG PARITY
----------
The noise seed is a deterministic function of sigma, so every architecture faces
bit-identical corrupted tensors. Without this, arm differences would be
confounded with noise draws.

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
from models.registry import build_arm, QUANTUM_ARMS             # noqa: E402
from train.loop import train_model                              # noqa: E402
from train.metrics import compute_metrics                       # noqa: E402

# 01 exposes the feature cache; reuse it rather than re-extracting.
_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features
FROZEN = _exp1.FROZEN

EXPERIMENT = "03_robustness"

ARMS = ["linear", "matched_param_fullrank", "fourier_rff", "quantum_vqc"]


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
    """Evaluate across NOISE_LEVELS. Returns metrics and per-sigma probabilities."""
    model.eval()
    curve, probs_by_sigma, labels = {}, {}, None

    for sigma in config.NOISE_LEVELS:
        torch.manual_seed(seed_for_sigma(seed, sigma))
        probs, preds, ys = [], [], []

        for x, y in test_batches:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)
            p = torch.softmax(model(add_gaussian_noise(x, sigma)), dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
            ys.append(y.cpu().numpy())

        probs = np.concatenate(probs)
        preds = np.concatenate(preds)
        labels = np.concatenate(ys)

        curve[f"{sigma:.2f}"] = compute_metrics(labels, preds, probs, num_classes)
        probs_by_sigma[f"{sigma:.2f}"] = probs

    return curve, probs_by_sigma, labels


def run_cell(dataset, n_per_class, seed, arm, dim=4, force=False,
             lr_head=None, lr_quantum=None):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed, arm=arm)
    if lr_head is not None:
        keys["lrh"] = f"{float(lr_head):.0e}"
    if lr_quantum is not None and arm in QUANTUM_ARMS:
        keys["lrq"] = f"{float(lr_quantum):.0e}"

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
        is_quantum=(arm in QUANTUM_ARMS), verbose=False,
        lr_head=lr_head, lr_quantum=lr_quantum)
    if best_state is not None:
        head_model.load_state_dict(best_state, strict=False)

    lip = lipschitz_of_trained(head_model.head, dim, head_model.angle_scale, device)

    # --- transplant into a full model and sweep noise on IMAGES ------------
    full = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                     seed=seed, freeze_policy=FROZEN, build_backbone=True).to(device)
    full.load_state_dict(head_model.state_dict(), strict=False)

    _, _, test_batches, meta = get_loaders(dataset, n_per_class=n_per_class,
                                           seed=seed, augment=False)
    curve, probs_by_sigma, labels = sweep_noise(full, test_batches, C, seed, device)

    # One array per sigma, under the shared naming function so 04 can find them.
    pred_file = shards.save_predictions(EXPERIMENT, labels, probs_by_sigma, **keys)

    payload = {
        "curve": curve,
        "lipschitz_trained": lip,
        "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test", "regime")},
        "predictions_file": pred_file,
        "lr": {"head": lr_head, "quantum": lr_quantum},
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del head_model, full
    torch.cuda.empty_cache()
    return curve["0.00"], curve, lip


# ------------------------------------------------------------------ summary
def summarise(metric="auc"):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    tbl, f1_tbl, lip = {}, {}, {}
    for r in rows:
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]))
        for s, m in r["curve"].items():
            tbl.setdefault((cell, s), {}).setdefault(k["arm"], {})[k["seed"]] = m.get(metric)
            f1_tbl.setdefault((cell, s), {}).setdefault(k["arm"], {})[k["seed"]] = m.get("macro_f1")
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

    print(f"\n=== Trained Lipschitz constants ===")
    print("Pairs with 08_lipschitz.py, which measures the same quantity at init.")
    for arm in ARMS:
        if arm in lip:
            print(f"  {arm:24s} L_max mean={np.mean(lip[arm]):8.3f} "
                  f"max={np.max(lip[arm]):8.3f}")

    n_pred = sum(1 for r in rows if r.get("predictions_file"))
    print(f"\n{n_pred}/{len(rows)} runs recorded per-sample predictions.")
    print("EXPLORATORY. Confirmatory statistics come from 04_statistical_analysis.py")
    print("using those predictions.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", type=int, default=[5, 20, 100])
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--use-tuned-lr", action="store_true",
                   help="load per-arm LRs selected by 09_lr_selection.py")
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

    tuned = {}
    if args.use_tuned_lr:
        import json
        path = os.path.join(config.ARTIFACT_ROOT, "lr_selection.json")
        if not os.path.exists(path):
            print(f"--use-tuned-lr: {path} not found; run 09_lr_selection.py first")
            return
        with open(path) as f:
            tuned = json.load(f)["selected"]
        print(f"using tuned LRs: {tuned}")

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
                    lr = tuned.get(arm)
                    out = run_cell(ds, n, seed, arm, dim=args.dim,
                                   force=args.force, lr_head=lr, lr_quantum=lr)
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
