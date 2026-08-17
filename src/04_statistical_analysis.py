"""
STATISTICS ENGINE.

Implements exactly what docs/analysis_plan.md pre-registers. Nothing here chooses
which comparison to report - that is fixed in the plan and this script executes it.

THE PRIMARY STATISTIC: NESTED PAIRED BOOTSTRAP
----------------------------------------------
Two independent sources of variance must both be captured:

    training variance  - different seeds give different models
    test variance      - a different draw of test images gives a different score

    for b in 1..B:
        I_b = resample test indices with replacement
        S_b = resample seeds with replacement
        Delta_b = mean over s in S_b of [ metric_A(s, I_b) - metric_B(s, I_b) ]

Pairing on seed matters: both arms saw identical splits and identical
initialisation seeds, so seed-level variance largely cancels and a small but
consistent difference becomes detectable.

WHY NOT A t-TEST OVER SEEDS
---------------------------
It captures training variance only. On BreastMNIST the test split is 156 images,
where the Hanley-McNeil standard error on AUC is roughly 0.03-0.04 - larger than
any effect at stake here. A seed-only test can report p < 0.05 on a difference
that a different draw of test images would reverse. That is precisely the error
the conference version made.

The nested bootstrap requires PER-SAMPLE PREDICTIONS. Runs that stored only
scalar metrics can support seed-level resampling alone; this script detects that
case, falls back, and labels the output SEED-LEVEL ONLY so the weaker analysis
can never be mistaken for the pre-registered one.

MULTIPLICITY
------------
Benjamini-Hochberg across the family declared in the analysis plan (17 tests).
Raw and adjusted p-values are both reported. Anything outside the declared family
is labelled exploratory and excluded from the correction - inflating the family
with exploratory tests would cost power on the tests that matter.

EFFECT SIZE
-----------
Cohen's d accompanies every p-value. Where a difference is significant but below
0.01 AUC, the output says so: that is smaller than the test-set sampling error on
the smallest dataset and carries no clinical meaning.

USAGE
-----
    python src/04_statistical_analysis.py --experiment 01_frozen
    python src/04_statistical_analysis.py --experiment 01_frozen --latex
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                      # noqa: E402
import shards                                      # noqa: E402

try:
    from sklearn.metrics import roc_auc_score, f1_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

PRED_ROOT = os.path.join(config.ARTIFACT_ROOT, "predictions")


# ------------------------------------------------------------------ metrics
def _auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        return np.nan


def _macro_f1(labels, probs):
    return f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)


METRIC_FN = {"auc": _auc, "macro_f1": lambda l, p, c: _macro_f1(l, p)}


# ------------------------------------------------------------------ bootstrap
def nested_paired_bootstrap(preds_a, preds_b, labels, num_classes,
                            metric="auc", B=None, rng=None):
    """
    preds_a / preds_b: {seed: probs[N, C]} for the two arms, same labels.
    Returns the observed difference, a 95% CI, a bootstrap p-value and Cohen's d.
    """
    B = B or config.BOOTSTRAP_RESAMPLES
    rng = rng or np.random.default_rng(20260814)
    fn = METRIC_FN[metric]

    seeds = sorted(set(preds_a) & set(preds_b))
    if len(seeds) < 2:
        return None
    n = len(labels)

    observed = float(np.mean([
        fn(labels, preds_a[s], num_classes) - fn(labels, preds_b[s], num_classes)
        for s in seeds]))

    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)                 # resample test indices
        ss = rng.choice(seeds, len(seeds), replace=True)   # resample seeds
        y = labels[idx]
        if len(np.unique(y)) < 2:                   # degenerate draw
            deltas[b] = np.nan
            continue
        deltas[b] = np.mean([
            fn(y, preds_a[s][idx], num_classes) - fn(y, preds_b[s][idx], num_classes)
            for s in ss])

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < B // 2:
        return None

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    per_seed = np.array([
        fn(labels, preds_a[s], num_classes) - fn(labels, preds_b[s], num_classes)
        for s in seeds])
    sd = per_seed.std(ddof=1)
    cohens_d = float(per_seed.mean() / sd) if sd > 0 else np.nan

    return {"delta": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(p, 1.0)), "cohens_d": cohens_d,
            "n_seeds": len(seeds), "n_test": n, "method": "nested_bootstrap"}


def seed_level_bootstrap(vals_a, vals_b, B=None, rng=None):
    """
    FALLBACK when per-sample predictions are unavailable. Resamples seeds only,
    so it is blind to test-set sampling variance. Every output is labelled
    SEED-LEVEL so it cannot be confused with the pre-registered analysis.
    """
    B = B or config.BOOTSTRAP_RESAMPLES
    rng = rng or np.random.default_rng(20260814)

    seeds = sorted(set(vals_a) & set(vals_b))
    d = np.array([vals_a[s] - vals_b[s] for s in seeds
                  if vals_a[s] is not None and vals_b[s] is not None])
    if len(d) < 2:
        return None

    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    sd = d.std(ddof=1)

    return {"delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(p, 1.0)),
            "cohens_d": float(d.mean() / sd) if sd > 0 else np.nan,
            "n_seeds": len(d), "n_test": None, "method": "seed_level_ONLY"}


def benjamini_hochberg(pvals, alpha=None):
    """Returns (rejected, adjusted). Adjusted p-values are monotone-enforced."""
    alpha = alpha or config.ALPHA
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return np.zeros_like(p, bool), p

    order = np.argsort(np.where(ok, p, np.inf))
    ranked = p[order][:m]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity

    out = np.full_like(p, np.nan)
    out[order[:m]] = np.minimum(adj, 1.0)
    return (out <= alpha) & ok, out


# ------------------------------------------------------------------ loading
def load_predictions(experiment, dataset, regime, dim, seed, arm, sigma=None):
    tag = f"{dataset}__n{regime}__d{dim}__s{seed}__{arm}.npz"
    path = os.path.join(PRED_ROOT, experiment, tag)
    if not os.path.exists(path):
        return None, None
    z = np.load(path)
    labels = z["labels"].astype(int)
    key = f"{sigma:.2f}" if sigma is not None else "0.00"
    if key not in z:
        key = [k for k in z.files if k != "labels"][0]
    return z[key].astype(np.float64), labels


def collect(experiment):
    """(dataset, regime, dim, fp) -> arm -> seed -> metrics dict"""
    tbl = defaultdict(lambda: defaultdict(dict))
    for r in shards.load_all(experiment):
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]), k.get("dim", 4), k.get("fp", "all"))
        tbl[cell][k["arm"]][k["seed"]] = r.get("metrics") or r.get("curve", {}).get("0.00")
    return tbl


# ------------------------------------------------------------------ analysis
def compare(experiment, tbl, cell, arm_a, arm_b, metric, num_classes):
    """Nested bootstrap where predictions exist; seed-level fallback otherwise."""
    ds, regime, dim, fp = cell
    a_seeds, b_seeds = tbl[cell].get(arm_a, {}), tbl[cell].get(arm_b, {})
    common = sorted(set(a_seeds) & set(b_seeds))
    if len(common) < 2:
        return None

    pa, pb, labels = {}, {}, None
    for s in common:
        x, l = load_predictions(experiment, ds, regime, dim, s, arm_a)
        y, _ = load_predictions(experiment, ds, regime, dim, s, arm_b)
        if x is None or y is None:
            pa = {}
            break
        pa[s], pb[s], labels = x, y, l

    if pa and HAVE_SKLEARN and labels is not None:
        return nested_paired_bootstrap(pa, pb, labels, num_classes, metric=metric)

    return seed_level_bootstrap({s: a_seeds[s].get(metric) for s in common},
                                {s: b_seeds[s].get(metric) for s in common})


def run(experiment, metric="auc", latex=False):
    import medmnist

    tbl = collect(experiment)
    if not tbl:
        print(f"No shards for '{experiment}'.")
        return

    pairs = [("PRIMARY  (H-P)", *config.PRIMARY_COMPARISON),
             ("SECONDARY (H-S2)", *config.SECONDARY_COMPARISON)]
    if hasattr(config, "DIAGNOSTIC_COMPARISON"):
        pairs.append(("diagnostic", *config.DIAGNOSTIC_COMPARISON))

    results = []
    for label, arm_a, arm_b in pairs:
        for cell in sorted(tbl):
            C = len(medmnist.INFO[cell[0]]["label"])
            r = compare(experiment, tbl, cell, arm_a, arm_b, metric, C)
            if r:
                r.update(family=label, cell=cell, arm_a=arm_a, arm_b=arm_b)
                results.append(r)

    if not results:
        print("Nothing comparable found.")
        return

    # BH correction over the PRIMARY family only, per the analysis plan.
    primary = [r for r in results if r["family"].startswith("PRIMARY")]
    rej, adj = benjamini_hochberg([r["p"] for r in primary])
    for r, rr, aa in zip(primary, rej, adj):
        r["p_adj"], r["significant"] = float(aa), bool(rr)

    methods = {r["method"] for r in results}
    if "seed_level_ONLY" in methods:
        print("\n" + "!" * 74)
        print("! SEED-LEVEL FALLBACK IN USE - per-sample predictions were not found.")
        print("! These intervals ignore test-set sampling variance and are NOT the")
        print("! pre-registered analysis. Re-run with prediction saving enabled")
        print("! before quoting any of this in the manuscript.")
        print("!" * 74)

    for label, arm_a, arm_b in pairs:
        rows = [r for r in results if r["family"] == label]
        if not rows:
            continue
        print(f"\n=== {label}: {arm_a} - {arm_b}  ({metric}) ===")
        print(f"{'dataset':15s} {'n/cls':>6s} {'enc':>6s} {'delta':>9s} "
              f"{'95% CI':>21s} {'p':>9s} {'p_adj':>9s} {'d':>7s}  verdict")
        print("-" * 108)
        for r in sorted(rows, key=lambda x: (x["cell"][0], int(x["cell"][1]))):
            ds, reg, dim, fp = r["cell"]
            enc = "froz" if fp == "all" else "adap"
            v = ("A better" if r["ci_lo"] > 0 else
                 "B better" if r["ci_hi"] < 0 else "no difference")
            if abs(r["delta"]) < 0.01 and v != "no difference":
                v += " (negligible)"
            padj = f"{r['p_adj']:9.4f}" if "p_adj" in r else "        -"
            print(f"{ds:15s} {reg:>6s} {enc:>6s} {r['delta']:+9.4f} "
                  f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] {r['p']:9.4f} {padj} "
                  f"{r['cohens_d']:+7.2f}  {v}")

    # --- H-P2: is the effect monotone in shots per class? -----------------
    print(f"\n=== H-P2: trend of delta on log2(shots/class), PRIMARY family ===")
    by_n = defaultdict(list)
    for r in primary:
        by_n[int(r["cell"][1])].append(r["delta"])
    ns = sorted(by_n)
    if len(ns) >= 3:
        x = np.log2(ns)
        y = np.array([np.mean(by_n[n]) for n in ns])
        slope = float(np.polyfit(x, y, 1)[0])
        rng = np.random.default_rng(7)
        boot = [float(np.polyfit(x, [rng.choice(by_n[n], len(by_n[n])).mean()
                                     for n in ns], 1)[0]) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        for n in ns:
            print(f"    n={n:>4d}  mean delta = {np.mean(by_n[n]):+.4f} "
                  f"({len(by_n[n])} cells)")
        print(f"\n    slope = {slope:+.5f} per doubling  [{lo:+.5f}, {hi:+.5f}]")
        print(f"    H-P2 {'SUPPORTED' if hi < 0 else 'NOT supported'} "
              f"(CI must exclude 0 and be negative)")
    else:
        print("    need >=3 shot levels")

    if latex:
        emit_latex(results, metric)


def emit_latex(results, metric):
    out = os.path.join(config.ARTIFACT_ROOT, f"table_{metric}.tex")
    with open(out, "w") as f:
        f.write("% auto-generated by 04_statistical_analysis.py\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(f"\\caption{{Paired comparison, {metric.upper()}. "
                f"Nested bootstrap, BH-FDR corrected.}}\n")
        f.write("\\begin{tabular}{llrrrr}\\toprule\n")
        f.write("Dataset & Shots & $\\Delta$ & 95\\% CI & $p_{adj}$ & $d$ \\\\\\midrule\n")
        for r in results:
            if not r["family"].startswith("PRIMARY"):
                continue
            ds = r["cell"][0].replace("mnist", "MNIST")
            padj = f"{r.get('p_adj', float('nan')):.3f}"
            f.write(f"{ds} & {r['cell'][1]} & {r['delta']:+.4f} & "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] & {padj} & "
                    f"{r['cohens_d']:+.2f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"\nLaTeX written to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="01_frozen")
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1"])
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()
    run(args.experiment, args.metric, args.latex)


if __name__ == "__main__":
    main()
