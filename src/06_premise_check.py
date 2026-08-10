"""
EXPERIMENT 6 - PREMISE CHECK.  RUN THIS BEFORE ANYTHING ELSE.

THE QUESTION
------------
The entire paper is built on the premise that compressing a 256-d feature
vector to d=4 destroys separability, and that a variational quantum head can
recover some of what is lost.

If d=4 performs as well as d=256, that premise is empty. There is no
compression penalty to bypass, and the paper is about data scarcity only - a
different, narrower claim.

This is not a hypothetical worry. The conference manuscript reported 0.9046 AUC
on BreastMNIST at d=4 with 100% data, which is close to published unbottlenecked
ResNet-18 MedMNIST benchmark territory. And in that same table the classical
LINEAR head won at 100% data, which is exactly what you would see if the
bottleneck were not binding.

WHAT IT MEASURES
----------------
A pure compression curve: identical architecture, identical training, only d
varies over {4, 8, 16, 32, 64, 256}. Using the linear arm isolates dimensionality
from head expressivity - if AUC is flat in d, the constraint does nothing.

HOW TO READ THE RESULT
----------------------
  Large gap d=4 vs d=256   -> premise holds. Proceed as planned.
  Small gap                -> premise fails at that regime. The paper narrows to
                              data scarcity, and the framing changes BEFORE
                              4,800 runs are spent, not after.

The gap will likely differ by regime: compression may bind under scarcity and
not under abundance, or vice versa. Both regimes are therefore measured.

USAGE
-----
    python src/06_premise_check.py                    # full check
    python src/06_premise_check.py --datasets breastmnist --seeds 42
    python src/06_premise_check.py --summary-only     # re-print from shards
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                        # noqa: E402
import shards                                        # noqa: E402
from data.medmnist_loader import get_loaders, num_classes_of   # noqa: E402
from models.registry import build_arm                # noqa: E402
from train.loop import train_model                   # noqa: E402

EXPERIMENT = "06_premise"
DIMS = [4, 8, 16, 32, 64, 256]
ARMS = ["linear"]          # deliberately the simplest head: isolates d


def run_cell(dataset, regime, dim, seed, arm, force=False):
    keys = dict(dataset=dataset, regime=regime, dim=dim, seed=seed, arm=arm)
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    C = num_classes_of(dataset)
    full = (regime == "full")

    train, val, test, meta = get_loaders(
        dataset,
        n_per_class=None if full else int(regime),
        seed=seed,
        augment=config.AUGMENT_E2E,
        full_data=full,
    )

    model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS, seed=seed)
    metrics, history, _ = train_model(
        model, train, val, test, num_classes=C,
        is_quantum=False, verbose=False)

    shards.write(EXPERIMENT,
                 {"metrics": metrics, "meta": meta,
                  "history": {k: history[k] for k in
                              ("val_f1", "val_auc", "train_f1", "pre_clip_grad_norm")}},
                 **keys)
    return metrics


def summarise():
    """Print the compression curve and the d=4 vs d=256 gap that gates the project."""
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    table = {}
    for r in rows:
        k = r["keys"]
        table.setdefault((k["dataset"], k["regime"]), {}).setdefault(k["dim"], []).append(
            r["metrics"]["auc"])

    print(f"\n{'dataset':16s} {'regime':>7s} " +
          " ".join(f"{'d=' + str(d):>13s}" for d in DIMS) + "   GAP(256-4)")
    print("-" * 118)

    verdicts = []
    for (ds, regime), by_dim in sorted(table.items()):
        cells = []
        for d in DIMS:
            v = [a for a in by_dim.get(d, []) if a is not None]
            cells.append(f"{np.mean(v):.4f}+-{np.std(v):.3f}" if v else "        -    ")
        a4 = [a for a in by_dim.get(4, []) if a is not None]
        a256 = [a for a in by_dim.get(256, []) if a is not None]
        gap = (np.mean(a256) - np.mean(a4)) if (a4 and a256) else float("nan")
        verdicts.append(gap)
        print(f"{ds:16s} {regime:>7s} " + " ".join(f"{c:>13s}" for c in cells) +
              f"   {gap:+.4f}")

    valid = [g for g in verdicts if not np.isnan(g)]
    if valid:
        mg = float(np.mean(valid))
        print("\n" + "=" * 60)
        print(f"MEAN AUC GAP (d=256 - d=4): {mg:+.4f}")
        if mg >= 0.05:
            print("PREMISE HOLDS. Compression costs real separability.")
            print("The bottleneck framing is supported. Proceed to the pilot.")
        elif mg >= 0.02:
            print("PREMISE IS WEAK. The gap is small but non-zero.")
            print("Report it honestly and soften the 'collapse' language;")
            print("check whether the gap concentrates in the scarce regimes.")
        else:
            print("PREMISE FAILS. d=4 performs about as well as d=256.")
            print("There is no compression penalty to bypass. Reframe the paper")
            print("around DATA SCARCITY before running the full sweep.")
        print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", default=["100", "full"],
                   help="n_per_class values, and/or the literal 'full'")
    p.add_argument("--dims", nargs="+", type=int, default=DIMS)
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS[:3])
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if args.summary_only:
        summarise()
        return

    total = len(args.datasets) * len(args.regimes) * len(args.dims) * len(args.seeds)
    print(f"Premise check: {total} cells | device={config.DEVICE} | sha={config.git_sha()[:8]}")

    done = 0
    for ds in args.datasets:
        for regime in args.regimes:
            for dim in args.dims:
                for seed in args.seeds:
                    done += 1
                    tag = f"[{done}/{total}] {ds} r={regime} d={dim} s={seed}"
                    m = run_cell(ds, regime, dim, seed, ARMS[0], force=args.force)
                    if m is None:
                        print(f"{tag} - cached")
                    else:
                        print(f"{tag} - auc={m['auc']:.4f} f1={m['macro_f1']:.4f} "
                              f"({m['epochs_run']} ep, {m['mean_epoch_time']:.2f}s/ep)")

    summarise()


if __name__ == "__main__":
    main()
