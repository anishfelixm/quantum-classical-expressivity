"""
Paired analysis of the Experiment 1 diagnostic shards.

WHY THIS IS SEPARATE FROM 01_frozen_backbone_ablation.py
---------------------------------------------------------
The summary printed by the sweep script hard-codes quantum_vqc vs fourier_rff. That is the
DEQUANTIZATION contrast (matched on basis dimension, 24 vs 324 parameters). It is a real
question, but it is not the parameter-efficiency question, and reporting it as if it were
would misstate the result.

This script makes the primary contrast quantum_vqc vs matched_param — 24 parameters each,
4 features to the classifier each — and keeps the Fourier comparison as a labelled
secondary.

STATISTICAL SCOPE - READ BEFORE QUOTING ANY NUMBER FROM THIS
-------------------------------------------------------------
Result shards store aggregate metrics per (cell, seed), NOT per-sample test predictions.
The nested paired bootstrap specified for the confirmatory analysis resamples BOTH seeds
and test indices, so it cannot be computed from these shards. What follows is a
seed-paired t-interval: it captures training variance only, and is blind to test-set
sampling variance.

On BreastMNIST (n_test = 156) the Hanley-McNeil standard error on AUC is roughly
0.03-0.04, which is larger than several of the effects reported here. Treat every interval
below as EXPLORATORY.

ACTION REQUIRED BEFORE THE CONFIRMATORY SWEEP: persist per-sample test probabilities and
labels in each shard, so the nested bootstrap becomes computable.

USAGE
-----
    python src/analyze_diagnostic.py
    python src/analyze_diagnostic.py --metric macro_f1
    python src/analyze_diagnostic.py --treatment quantum_vqc --control fourier_rff
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import shards          # noqa: E402

FROZEN, ADAPTIVE = "all", "layer3_only"


def load_table(experiment, metric):
    """(dataset, regime, dim, freeze_policy) -> arm -> {seed: value}"""
    tbl = defaultdict(lambda: defaultdict(dict))
    for r in shards.load_all(experiment):
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]), k["dim"], k.get("fp", FROZEN))
        tbl[cell][k["arm"]][k["seed"]] = r["metrics"].get(metric)
    return tbl


def paired(a_by_seed, b_by_seed):
    """
    Paired difference over seeds present in both arms.

    Pairing matters: both arms saw identical splits and identical initialisation seeds,
    so seed-level variance largely cancels and a small consistent difference stays
    detectable. Returns (mean, lo, hi, n, cohens_d, p).
    """
    common = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in common
                  if a_by_seed[s] is not None and b_by_seed[s] is not None])
    if len(d) < 3:
        return (np.nan,) * 6

    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        return mean, mean, mean, n, np.inf, 0.0
    tcrit = stats.t.ppf(0.975, df=n - 1)          # t, not 1.96 - n is 10, not 100
    t_stat, p = stats.ttest_rel(
        [a_by_seed[s] for s in common if a_by_seed[s] is not None],
        [b_by_seed[s] for s in common if b_by_seed[s] is not None])
    return mean, mean - tcrit * se, mean + tcrit * se, n, mean / sd, float(p)


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns a boolean rejection mask under BH-FDR control."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.zeros_like(p, dtype=bool)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    reject = np.zeros_like(p, dtype=bool)
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= rank / m * alpha:
            kmax = rank
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            reject[i] = True
    return reject


def compare(tbl, treatment, control, metric, alpha=0.05):
    rows, pvals = [], []
    for cell in sorted(tbl):
        t = tbl[cell].get(treatment, {})
        c = tbl[cell].get(control, {})
        if not t or not c:
            continue
        mean, lo, hi, n, dz, p = paired(t, c)
        if np.isnan(mean):
            continue
        rows.append({"cell": cell, "mean": mean, "lo": lo, "hi": hi,
                     "n": n, "d": dz, "p": p,
                     "t_mean": np.mean([v for v in t.values() if v is not None]),
                     "c_mean": np.mean([v for v in c.values() if v is not None])})
        pvals.append(p)

    reject = benjamini_hochberg(pvals, alpha)
    for row, rej in zip(rows, reject):
        row["sig"] = bool(rej)

    print(f"\n{'='*104}")
    print(f"  {treatment}  -  {control}      metric: {metric}")
    print(f"  paired over seeds | 95% t-interval | BH-FDR across {len(rows)} cells")
    print(f"{'='*104}")
    print(f"{'dataset':15s} {'n/cls':>6s} {'encoder':>9s} "
          f"{treatment[:9]:>9s} {control[:9]:>9s} {'delta':>8s} "
          f"{'95% CI':>18s} {'d':>6s} {'p':>8s}  verdict")
    print("-" * 104)

    wins = losses = ties = 0
    for row in rows:
        ds, reg, dim, fp = row["cell"]
        enc = "frozen" if fp == FROZEN else "adaptive"
        if row["sig"] and row["mean"] > 0:
            verdict, wins = f"{treatment.split('_')[0].upper()} better", wins + 1
        elif row["sig"] and row["mean"] < 0:
            verdict, losses = f"{control.split('_')[0]} better", losses + 1
        else:
            verdict, ties = "no difference", ties + 1
        print(f"{ds:15s} {reg:>6s} {enc:>9s} {row['t_mean']:9.4f} {row['c_mean']:9.4f} "
              f"{row['mean']:+8.4f} [{row['lo']:+.4f},{row['hi']:+.4f}] "
              f"{row['d']:+6.2f} {row['p']:8.1e}  {verdict}")

    print("-" * 104)
    print(f"  {treatment} better: {wins}   |   {control} better: {losses}   "
          f"|   no difference: {ties}   (after FDR correction)")

    by_ds = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        i = 0 if (row["sig"] and row["mean"] > 0) else (1 if row["sig"] else 2)
        by_ds[row["cell"][0]][i] += 1
    print("\n  per dataset (treatment-better / control-better / tie):")
    for ds, (w, l, t) in sorted(by_ds.items()):
        print(f"    {ds:16s} {w} / {l} / {t}")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="01_frozen")
    p.add_argument("--metric", default="auc")
    p.add_argument("--treatment", default="quantum_vqc")
    p.add_argument("--control", default=None,
                   help="omit to run both the primary and secondary contrasts")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    tbl = load_table(args.experiment, args.metric)
    arms = sorted({a for cell in tbl.values() for a in cell})
    print(f"Loaded {len(tbl)} cells | arms present: {arms}")
    print("\nEXPLORATORY - seed-paired only. Test-set sampling variance is NOT captured.")
    print("See the module docstring before quoting any interval.")

    if args.control:
        compare(tbl, args.treatment, args.control, args.metric, args.alpha)
        return

    if "matched_param" in arms:
        print("\n\n########  PRIMARY (H1): parameter efficiency, 24 params vs 24 params")
        compare(tbl, args.treatment, "matched_param", args.metric, args.alpha)
    else:
        print("\n[!] matched_param shards not found - H1 cannot be evaluated.")

    if "fourier_rff" in arms:
        print("\n\n########  SECONDARY (H3): dequantization, basis-matched (24 vs 324 params)")
        compare(tbl, args.treatment, "fourier_rff", args.metric, args.alpha)

    if "linear" in arms:
        print("\n\n########  REFERENCE: vs the minimum-capacity classical floor")
        compare(tbl, args.treatment, "linear", args.metric, args.alpha)


if __name__ == "__main__":
    main()
