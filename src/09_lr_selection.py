"""
EXPERIMENT 9 - PER-ARM LEARNING-RATE SELECTION.

THE PROBLEM WITH A SHARED LEARNING RATE
----------------------------------------
Using LR = 1e-3 for every arm looks like fairness - identical hyperparameters -
but it confounds ARCHITECTURE with HYPERPARAMETER SUITABILITY. Measured mean
gradient norms at d=4 differ several-fold across arms:

    quantum_vqc      0.48 - 0.76
    linear           0.97 - 1.39
    fourier_rff      1.27 - 2.55
    deep_funnel      2.88 - 4.79

At a shared LR the quantum arm takes systematically smaller effective steps. So
"the VQC underperforms" and "the VQC was under-stepped" are indistinguishable,
and a reviewer only has to say so to dismiss a negative result. The convergence
audit (best epoch 52.9 quantum vs 56.9/57.9 classical) shows it is not stalling,
but converged is not the same as converged to as good an optimum as a better LR
would have reached.

THE PROTOCOL, AND WHY EACH PIECE IS THERE
------------------------------------------
GRID. {3e-4, 1e-3, 3e-3, 1e-2}, log-spaced, IDENTICAL for every arm. Per-arm
grids would reintroduce the bias this exists to remove.

TUNING SEEDS. Disjoint from config.CONFIRMATORY_SEEDS, so selection never
touches a split the confirmatory sweep will evaluate on. Asserted at import.

CRITERION. Validation AUC, never test. AUC because it is the primary endpoint
and because selecting on F1 interacts with the VQC's calibration failure.

ONE GLOBAL LR PER ARM, not one per cell. At n=5/class the validation set is
10-20 images; selecting per cell on 10 images mostly fits noise and would let
each arm cherry-pick lucky configurations. Aggregating across datasets and
regimes gives a stable estimate. The per-regime breakdown is printed as a
sensitivity check, not used for selection.

SCOPE. Frozen encoder, d=4 - matching the primary comparison. With a frozen
backbone there is no LR_BACKBONE to tune, which removes an axis entirely.

REPORT THE WHOLE SWEEP. The appendix shows every arm's LR-versus-AUC curve, not
just the winners. That is what demonstrates the tuning was not selective.

HOW THE LR REACHES THE TRAINER
------------------------------
Through the lr_head / lr_quantum ARGUMENTS of train_model.

This previously assigned to config.LR_HEAD - a module global - inside the run
loop, restoring it in a finally block. That works until it does not: any escape
from that block leaks a hyperparameter into every subsequent run in the process,
and the shards would carry no trace of it.

OUTPUT
------
artifacts/lr_selection.json, which 01 and 03 read via --use-tuned-lr. Without
this file that flag is inert, so the confirmatory sweep would silently run on
untuned defaults while the log claimed otherwise.

BRACE FOR THIS
--------------
Tuning may change the scarcity crossover: it could strengthen, weaken or vanish.
That is precisely why it runs BEFORE the confirmatory sweep. Doing it afterwards
would mean choosing hyperparameters with knowledge of the outcome.

USAGE
-----
    python src/09_lr_selection.py --quick      # 2 datasets, sanity
    python src/09_lr_selection.py
    python src/09_lr_selection.py --summary-only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm, QUANTUM_ARMS             # noqa: E402
from train.loop import train_model                              # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features
FROZEN = _exp1.FROZEN

EXPERIMENT = "09_lr_selection"
SELECTION_FILE = os.path.join(config.ARTIFACT_ROOT, "lr_selection.json")

# Extended 27 Aug 2026 (Amendment 3a): the first four points returned a
# monotone curve whose maximum sat on the boundary, which is the signature of a
# search range that is too narrow. Adding 3e-2 and 1e-1 moved the optimum
# INTERIOR for every arm, so the selection is now a real optimum rather than a
# grid artifact. Shards are keyed on LR, so the original 960 runs were reused.
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
TUNING_REGIMES = [5, 20, 100]          # scarce, middle, abundant
ARMS = ["linear", "matched_param_fullrank", "fourier_rff", "quantum_vqc"]

# Disjoint from CONFIRMATORY_SEEDS so tuning never sees a confirmatory split.
TUNING_SEEDS = [90001, 90002, 90003, 90004, 90005]

_overlap = set(TUNING_SEEDS) & set(config.CONFIRMATORY_SEEDS)
if _overlap:
    raise RuntimeError(
        f"tuning seeds overlap confirmatory seeds {sorted(_overlap)} - selection "
        f"would then be made on splits the confirmatory sweep evaluates on.")


def run_cell(dataset, n_per_class, seed, arm, lr, dim=4, force=False):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed,
                arm=arm, lr=f"{lr:.0e}")
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    C = num_classes_of(dataset)
    t0 = time.time()

    loaders, _ = get_cached_features(dataset, str(n_per_class), seed)
    model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                      seed=seed, build_backbone=False)

    # Passed as arguments, never by mutating config. The quantum group is empty
    # for classical arms, so setting both is harmless and keeps the grid
    # identical across arms.
    metrics, history, _ = train_model(
        model, loaders["train"], loaders["val"], loaders["test"],
        num_classes=C, use_features=True,
        is_quantum=(arm in QUANTUM_ARMS), verbose=False,
        lr_head=lr, lr_quantum=lr)

    # SELECTION USES VALIDATION ONLY. Test metrics are stored for completeness
    # and are deliberately not consulted by choose_lr().
    grads = [g for g in history["pre_clip_grad_norm"] if g is not None]
    payload = {
        "val_auc": metrics.get("best_val_auc"),
        "val_f1": metrics.get("best_val_f1"),
        "test_metrics": metrics,
        "best_epoch": metrics.get("best_epoch"),
        "epochs_run": metrics.get("epochs_run"),
        "mean_grad_norm": float(np.mean(grads)) if grads else None,
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del model
    torch.cuda.empty_cache()
    return payload


def choose_lr():
    """One LR per arm: the grid point with the highest mean VALIDATION AUC."""
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        return {}, {}

    by_arm_lr = {}
    for r in rows:
        k = r["keys"]
        v = r.get("val_auc")
        if v is not None:
            by_arm_lr.setdefault(k["arm"], {}).setdefault(k["lr"], []).append(v)

    chosen, table = {}, {}
    for arm, lrs in by_arm_lr.items():
        means = {lr: float(np.mean(v)) for lr, v in lrs.items()}
        table[arm] = means
        chosen[arm] = float(max(means, key=means.get))   # numeric, for the JSON
    return chosen, table


def write_selection(chosen, table):
    """
    Persist the choice so 01 and 03 can consume it with --use-tuned-lr.
    Without this file that flag is inert and the sweep silently runs untuned.
    """
    payload = {
        "selected": chosen,
        "full_sweep": table,
        "grid": LR_GRID,
        "tuning_seeds": TUNING_SEEDS,
        "tuning_regimes": TUNING_REGIMES,
        "criterion": "mean validation AUC, aggregated over all tuning cells",
        "git_sha": config.git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(SELECTION_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    return SELECTION_FILE


def summarise(write=True):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    chosen, table = choose_lr()

    # Columns come from the shards actually on disk, NOT from LR_GRID. Deriving
    # them from the constant meant that extending the grid at the command line
    # silently dropped the new points from the table - so the printed sweep
    # omitted the very learning rates that decided the selection, while the
    # "chosen" column named one of them. The appendix table is built from this
    # output, so a partial table is a misreported experiment.
    present = sorted({float(r["keys"]["lr"]) for r in rows})
    grid = [f"{lr:.0e}" for lr in present]
    missing = [f"{lr:.0e}" for lr in LR_GRID if f"{lr:.0e}" not in grid]

    print("\n=== Validation AUC by learning rate (mean over all tuning cells) ===")
    print("Selection is on VALIDATION only, with seeds disjoint from the")
    print("confirmatory set. The full sweep is reported, not just the winners.")
    if missing:
        print(f"NOTE: {missing} are in LR_GRID but have no runs on disk.")
    print(f"\n{'arm':26s} " + " ".join(f"{g:>9s}" for g in grid) + "   chosen")
    print("-" * (26 + 10 * len(grid) + 10))
    for arm in ARMS:
        if arm not in table:
            continue
        cells = [f"{table[arm][g]:9.4f}" if g in table[arm] else "        -"
                 for g in grid]
        star = f"{chosen[arm]:.0e}" if arm in chosen else "-"
        print(f"{arm:26s} " + " ".join(cells) + f"   {star}")

    # Per-regime breakdown: a sensitivity check, NOT used for selection.
    print("\n=== Per-regime breakdown (sensitivity check only) ===")
    print("If the best LR were wildly regime-dependent, one global choice per")
    print("arm would be the wrong model - so this is worth looking at.")
    per = {}
    for r in rows:
        k = r["keys"]
        if r.get("val_auc") is not None:
            per.setdefault((k["arm"], str(k["regime"])), {}) \
               .setdefault(k["lr"], []).append(r["val_auc"])
    for arm in ARMS:
        line, bests = [], []
        for reg in TUNING_REGIMES:
            m = per.get((arm, str(reg)))
            if not m:
                continue
            best = max(m, key=lambda g: float(np.mean(m[g])))
            bests.append(best)
            line.append(f"n={reg}:{best}")
        if line:
            flag = "" if len(set(bests)) == 1 else "   <-- varies by regime"
            print(f"  {arm:26s} " + "  ".join(line) + flag)

    # Gradient norms: the mechanism that motivated tuning in the first place.
    print("\n=== Mean pre-clip gradient norm at the chosen LR ===")
    for arm in ARMS:
        if arm not in chosen:
            continue
        tag = f"{chosen[arm]:.0e}"
        g = [r["mean_grad_norm"] for r in rows
             if r["keys"]["arm"] == arm and r["keys"]["lr"] == tag
             and r.get("mean_grad_norm") is not None]
        if g:
            print(f"  {arm:26s} {np.mean(g):8.3f}")

    # A maximum at either end of the range means the true optimum is probably
    # outside it, and the tuning has not done its job.
    edge = [a for a in chosen
            if f"{chosen[a]:.0e}" in (grid[0], grid[-1])]
    if edge:
        print(f"\nWARNING: {edge} selected a learning rate at the EDGE of the")
        print("searched range. Extend the grid and re-run - a boundary optimum")
        print("means the search was too narrow, not that the boundary is best.")
    else:
        print("\nAll selections are interior to the searched range.")

    print("\n=== Selected learning rates ===")
    for arm, lr in sorted(chosen.items()):
        print(f"  {arm:26s} -> {lr:.0e}")

    if write and chosen:
        path = write_selection(chosen, table)
        print(f"\nWritten to {path}")
        print("01 and 03 consume it with --use-tuned-lr. config.LR_HEAD and")
        print("config.LR_QUANTUM are deliberately NOT edited: shard keys record")
        print("the explicit LR, so changing the default would make old shards")
        print("collide with new ones.")

    print("\nAmend docs/analysis_plan.md with the grid, the selection rule, the")
    print("tuning seeds and these results BEFORE the confirmatory sweep runs.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", type=int, default=TUNING_REGIMES)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=TUNING_SEEDS)
    p.add_argument("--lrs", nargs="+", type=float, default=LR_GRID)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-write", action="store_true",
                   help="print the table without writing lr_selection.json")
    args = p.parse_args()

    if args.summary_only:
        summarise(write=not args.no_write)
        return

    if args.quick:
        args.datasets = ["breastmnist", "pneumoniamnist"]
        args.regimes = [5, 100]
        args.seeds = TUNING_SEEDS[:2]

    total = (len(args.datasets) * len(args.regimes) * len(args.arms)
             * len(args.seeds) * len(args.lrs))
    print(f"LR selection | {total} cells | grid={[f'{l:.0e}' for l in args.lrs]}")
    print(f"  tuning seeds {args.seeds} (disjoint from CONFIRMATORY_SEEDS)")
    print(f"  device={config.DEVICE} sha={config.git_sha()[:8]}")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for n in args.regimes:
            for seed in args.seeds:
                for arm in args.arms:
                    for lr in args.lrs:
                        done += 1
                        out = run_cell(ds, n, seed, arm, lr, dim=args.dim,
                                       force=args.force)
                        if out is None:
                            continue
                        eta = (time.time() - t0) / done * (total - done) / 3600
                        print(f"[{done}/{total}] {ds} n={n} s={seed} {arm:24s} "
                              f"lr={lr:.0e} val_auc={out['val_auc']:.4f} "
                              f"ETA {eta:.1f}h", flush=True)

    summarise(write=not args.no_write)


if __name__ == "__main__":
    main()
