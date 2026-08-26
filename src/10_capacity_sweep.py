"""
EXPERIMENT 10 - CAPACITY SWEEP.  THE MECHANISM TEST.

WHY THIS IS THE MOST IMPORTANT EXPERIMENT IN THE PROJECT
---------------------------------------------------------
The paper's thesis is:

    The quantum head's advantage, where it exists, is a REGULARIZATION effect
    of its restricted function class - 24 parameters reaching a 24-dimensional
    manifold inside an 81-dimensional trigonometric span - not a quantum
    computational advantage.

Every other arm tests something adjacent to that claim. None tests the claim.

    matched_param_fullrank   controls CAPACITY at one fixed value
    fourier_rff              controls FUNCTION CLASS
    quantum_reupload         controls SPECTRAL RICHNESS

The mechanism itself - "restriction helps under scarcity and hurts under
abundance" - was inferred FROM the crossover and then used to EXPLAIN the
crossover. That is circular. This experiment breaks the circle by varying
restriction directly, in a purely classical head, and asking whether the same
crossover appears.

THE PREDICTION, RECORDED BEFORE THE RUN
----------------------------------------
Delta(rank) = AUC(rank) - AUC(rank=REFERENCE_RANK), evaluated at each
shots-per-class.

If restriction is the mechanism, LOW-rank heads should behave like the VQC:

    n=5, 10      Delta > 0    restriction prevents overfitting -> helps
    n=50, 100    Delta < 0    restriction prevents fitting     -> hurts

and the slope of Delta on log2(n) should be NEGATIVE for small ranks and flat
for the reference. A 16-parameter CLASSICAL head reproducing the quantum
crossover would show the effect is classically reproducible by restriction
alone - a stronger and more useful claim than "the quantum head sometimes wins".

WHAT WOULD REFUTE IT
--------------------
Flat Delta across n at every rank, or a crossover running the wrong way. Then
restriction is NOT what produces the quantum crossover, and the paper needs a
different explanation. That outcome is equally publishable and must be reported
with the same prominence - which is why this prediction is written down here,
before the data exists.

WHY LowRankHead AND NOT A WIDTH SWEEP
--------------------------------------
Capacity must vary WITHOUT rank varying, or the two are confounded - exactly the
flaw in MatchedParamHead, whose width-3 hidden layer caps rank at 3 regardless
of d.

    r = GELU( ((I + U V^T) z) * scale + bias ),   params = 2*d*rank + 2*d

I + U V^T is generically invertible at every rank including 0, so the map stays
full-rank while capacity ranges over (at d=4):

    rank=0 ->  8 params (diagonal affine)     most restricted
    rank=1 -> 16
    rank=2 -> 24 params                       == VQC exactly
    rank=4 -> 40
    rank=8 -> 72                              least restricted

A width-w MLP cannot go below 2*d*d = 32 parameters while staying full rank, so
it cannot even reach the VQC's 24 - let alone the restricted end where the
interesting behaviour is.

SECOND AXIS: EXPLICIT REGULARIZATION
------------------------------------
--weight-decay sweeps the OTHER kind of restriction. config.WEIGHT_DECAY_VARIANTS
has sat unused since the beginning. If heavy weight decay on a rank-8 head
reproduces what rank-0 does, that is independent confirmation through a
completely different mechanism.

SCOPE
-----
Frozen encoder, d=4, classical arms only. Runs on cached 256-d features, so a
cell costs seconds rather than minutes.

USAGE
-----
    python src/10_capacity_sweep.py --quick
    python src/10_capacity_sweep.py
    python src/10_capacity_sweep.py --ranks 8 --weight-decay 0.0 1e-4 1e-2
    python src/10_capacity_sweep.py --summary-only
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm                           # noqa: E402
from train.loop import train_model                              # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features
FROZEN = _exp1.FROZEN

EXPERIMENT = "10_capacity"

RANKS = list(getattr(config, "LOW_RANK_SWEEP", [0, 1, 2, 4, 8]))
REFERENCE_RANK = max(RANKS)     # least restricted; Delta is measured against it


def run_cell(dataset, n_per_class, seed, rank, dim=4, weight_decay=None,
             force=False):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed,
                arm="low_rank", rank=rank)
    if weight_decay is not None:
        keys["wd"] = f"{float(weight_decay):.0e}"

    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    C = num_classes_of(dataset)
    t0 = time.time()

    loaders, meta = get_cached_features(dataset, str(n_per_class), seed)
    model = build_arm("low_rank", d=dim, num_classes=C,
                      n_layers=config.VQC_LAYERS, seed=seed,
                      build_backbone=False, head_rank=rank)

    metrics, history, _, probs, labels = train_model(
        model, loaders["train"], loaders["val"], loaders["test"],
        num_classes=C, use_features=True, is_quantum=False,
        verbose=False, return_probs=True, weight_decay=weight_decay)

    payload = {
        "metrics": metrics,
        "head": model.head.describe(),
        "capacity": model.capacity_report(),
        "weight_decay": (config.WEIGHT_DECAY if weight_decay is None
                         else float(weight_decay)),
        "predictions_file": shards.save_predictions(
            EXPERIMENT, labels, probs, **keys),
        "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test", "regime")},
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del model
    torch.cuda.empty_cache()
    return metrics


# ------------------------------------------------------------------ summary
def _paired(a, b):
    """Mean paired difference over shared seeds, normal-approximation interval."""
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common
                  if a[s] is not None and b[s] is not None])
    if len(d) < 2:
        return None
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, m - half, m + half, len(d)


def _mean_delta(tbl, rank, regimes):
    """Mean of the per-dataset paired deltas against REFERENCE_RANK."""
    vals = []
    for reg in regimes:
        for (ds, r2) in tbl:
            if r2 != reg:
                continue
            out = _paired(tbl[(ds, reg)].get(rank, {}),
                          tbl[(ds, reg)].get(REFERENCE_RANK, {}))
            if out:
                vals.append(out[0])
    return float(np.mean(vals)) if vals else float("nan")


def summarise(metric="auc"):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    # Default weight decay only, so the two axes never mix in one table.
    default_wd = [r for r in rows if "wd" not in r["keys"]]
    if not default_wd:
        default_wd = rows

    tbl = defaultdict(lambda: defaultdict(dict))
    params = {}
    for r in default_wd:
        k = r["keys"]
        tbl[(k["dataset"], int(k["regime"]))][int(k["rank"])][k["seed"]] = \
            r["metrics"].get(metric)
        params[int(k["rank"])] = (r.get("head") or {}).get("params")

    ranks = sorted({rk for cell in tbl.values() for rk in cell})
    regimes = sorted({reg for (_, reg) in tbl})
    if not ranks or not regimes:
        print("Nothing to summarise.")
        return

    print(f"\n=== {metric.upper()} by head capacity (mean over seeds) ===")
    print("LowRankHead: 2*d*rank + 2*d parameters, full rank at every setting.")
    hdr = "  ".join(f"r={rk}({params.get(rk, '?')}p)".rjust(13) for rk in ranks)
    print(f"\n{'dataset':16s} {'n/cls':>6s}  {hdr}")
    print("-" * (24 + 15 * len(ranks)))
    for (ds, reg) in sorted(tbl):
        cells = []
        for rk in ranks:
            v = [x for x in tbl[(ds, reg)].get(rk, {}).values() if x is not None]
            cells.append(f"{np.mean(v):13.4f}" if v else "            -")
        print(f"{ds:16s} {reg:>6d}  " + "  ".join(cells))

    # --- THE TEST -------------------------------------------------------
    print(f"\n=== Delta vs rank={REFERENCE_RANK} (the least restricted head) ===")
    print("Positive at small n and negative at large n = restriction behaves as")
    print("the mechanism predicts, in a purely classical head.")
    print(f"\n{'rank':>5s} {'params':>7s}  " +
          "  ".join(f"n={r}".rjust(9) for r in regimes) + "     slope")
    print("-" * (16 + 11 * len(regimes) + 12))

    slopes = {}
    for rk in ranks:
        if rk == REFERENCE_RANK:
            continue
        cells, per_n = [], {}
        for reg in regimes:
            d = _mean_delta(tbl, rk, [reg])
            if np.isnan(d):
                cells.append("        -")
            else:
                per_n[reg] = d
                cells.append(f"{d:+9.4f}")

        if len(per_n) >= 3:
            xs = np.log2(sorted(per_n))
            ys = np.array([per_n[n] for n in sorted(per_n)])
            slopes[rk] = float(np.polyfit(xs, ys, 1)[0])
            slope_s = f"{slopes[rk]:+.5f}"
        else:
            slope_s = "     -"
        print(f"{rk:>5d} {str(params.get(rk, '?')):>7s}  " +
              "  ".join(cells) + f"  {slope_s}")

    # --- verdict --------------------------------------------------------
    print(f"\n=== Verdict ===")
    print("Prediction recorded BEFORE this run: if restriction is the mechanism,")
    print("small ranks help at n=5-10, hurt at n=50-100, and slope < 0.")

    scarce, abundant = regimes[:2], regimes[-2:]
    reproduced = []
    for rk in sorted(slopes):
        ms, ml, sl = _mean_delta(tbl, rk, scarce), _mean_delta(tbl, rk, abundant), slopes[rk]
        ok = (not np.isnan(ms) and not np.isnan(ml) and ms > 0 > ml and sl < 0)
        reproduced.append(ok)
        print(f"  rank={rk:<2d} ({params.get(rk, '?')} params): "
              f"scarce {ms:+.4f}  abundant {ml:+.4f}  slope {sl:+.5f}"
              f"{'   <-- crossover reproduced' if ok else ''}")

    if any(reproduced):
        print("\n  At least one purely CLASSICAL restricted head reproduces the")
        print("  crossover. Restriction is sufficient; nothing quantum is needed")
        print("  to produce the effect. This is the paper's mechanism TESTED,")
        print("  not asserted.")
    else:
        print("\n  No classical rank reproduces the crossover. The regularization")
        print("  explanation is NOT supported by this test - report that plainly")
        print("  and revise the mechanism claim rather than keeping it.")

    # --- weight decay axis, if swept ------------------------------------
    wd_rows = [r for r in rows if "wd" in r["keys"]]
    if wd_rows:
        print(f"\n=== Weight-decay axis (independent restriction mechanism) ===")
        print("If heavy decay on a large head reproduces what a small rank does,")
        print("that is the same conclusion reached through different machinery.")
        wd_tbl = defaultdict(lambda: defaultdict(dict))
        for r in wd_rows:
            k = r["keys"]
            wd_tbl[(int(k["regime"]), int(k["rank"]))][k["wd"]][k["seed"]] = \
                r["metrics"].get(metric)
        wds = sorted({w for c in wd_tbl.values() for w in c})
        print(f"\n{'n/cls':>6s} {'rank':>5s}  " +
              "  ".join(w.rjust(9) for w in wds))
        print("-" * (14 + 11 * len(wds)))
        for (reg, rk) in sorted(wd_tbl):
            cells = []
            for w in wds:
                v = [x for x in wd_tbl[(reg, rk)].get(w, {}).values() if x is not None]
                cells.append(f"{np.mean(v):9.4f}" if v else "        -")
            print(f"{reg:>6d} {rk:>5d}  " + "  ".join(cells))

    n_pred = sum(1 for r in rows if r.get("predictions_file"))
    print(f"\n{n_pred}/{len(rows)} runs recorded per-sample predictions.")
    print("EXPLORATORY until pre-registered. Amend docs/analysis_plan.md with")
    print("this hypothesis and its prediction before quoting it as confirmatory.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", type=int, default=config.N_PER_CLASS)
    p.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--weight-decay", nargs="+", type=float, default=[None],
                   help="sweep explicit regularization as a second axis")
    p.add_argument("--quick", action="store_true")
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
        args.seeds = config.ALL_SEEDS[:3]

    # Preflight: show the parity table, so rank=2 == VQC is visible before any
    # compute is spent.
    print(f"Head parameter counts at d={args.dim}:")
    for rk in args.ranks:
        m = build_arm("low_rank", d=args.dim, num_classes=2, seed=42,
                      build_backbone=False, head_rank=rk)
        h = m.head.describe()
        star = "  <-- matches VQC exactly" if h["exact_match"] else ""
        print(f"  rank={rk:<2d} {h['params']:>4d} params "
              f"(VQC: {h['target_params']}){star}")

    total = (len(args.datasets) * len(args.regimes) * len(args.ranks)
             * len(args.seeds) * len(args.weight_decay))
    print(f"\nCapacity sweep | {total} cells | classical only, cached features")
    print(f"  device={config.DEVICE} sha={config.git_sha()[:8]}")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for n in args.regimes:
            for seed in args.seeds:
                for rk in args.ranks:
                    for wd in args.weight_decay:
                        done += 1
                        m = run_cell(ds, n, seed, rk, dim=args.dim,
                                     weight_decay=wd, force=args.force)
                        if m is None:
                            continue
                        eta = (time.time() - t0) / done * (total - done) / 3600
                        print(f"[{done}/{total}] {ds} n={n} s={seed} rank={rk} "
                              f"auc={m['auc']:.4f} f1={m['macro_f1']:.4f} "
                              f"ETA {eta:.1f}h", flush=True)

    summarise(args.metric)


if __name__ == "__main__":
    main()
