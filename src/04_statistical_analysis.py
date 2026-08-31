"""
STATISTICS ENGINE.

Implements what docs/analysis_plan.md pre-registers. Nothing here chooses which
comparison to report - that is fixed in the plan and this script executes it.

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
that a different draw of test images would reverse.

HOW PREDICTIONS ARE FOUND
-------------------------
Through shards.load_predictions(), using the keys recorded in each shard.

This previously reconstructed filenames from a hardcoded pattern that matched
what 03 wrote but NOT what 01 wrote, so the primary comparison silently fell
back to seed-level resampling while printing a table that looked correct. The
reader now uses the same naming function as the writer, and the fallback is
loudly labelled wherever it is still used.

load_predictions returns a TUPLE (probs, labels), and (None, None) when the file
is absent - so the missing-file guard must test `xa[0] is None`, not `xa is None`.
A two-tuple is never None, so the old guard never fired: a run whose predictions
were missing had `None` written into the probability dict, and whether that
crashed inside roc_auc_score or silently fell through to the seed-level path
depended on which seed happened to be missing. Partial prediction directories -
the normal state of an interrupted sweep - hit exactly that case.

MULTIPLICITY
------------
Benjamini-Hochberg across the family declared in the analysis plan. The plan
declares 17 tests spanning several experiments, so a single invocation of this
script usually cannot compute all of them. --family-size lets you correct
against the full declared family even when running a subset; without it the
script corrects over what it computed AND prints a warning, because correcting
over a smaller family than declared is anti-conservative.

EFFECT SIZE
-----------
Cohen's d accompanies every p-value. Where a difference is significant but below
0.01 AUC the output says so: that is smaller than the test-set sampling error on
the smallest dataset and carries no clinical meaning.

USAGE
-----
    python src/04_statistical_analysis.py --experiment 01_frozen
    python src/04_statistical_analysis.py --experiment 01_frozen --family-size 17
    python src/04_statistical_analysis.py --experiment 03_robustness --condition 0.20
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                      # noqa: E402
import shards                                      # noqa: E402

from sklearn.metrics import roc_auc_score, f1_score   # noqa: E402


# ------------------------------------------------------------------ metrics
# Why a metric could not be computed, accumulated so a run can explain ITSELF
# instead of emitting a bare NaN that turns into a silent fallback.
_AUC_FAILURES = {}


def _auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError as e:
        # "a class is absent from this draw" is expected during bootstrapping.
        # "scores do not sum to one" is a STORAGE fault and must be surfaced:
        # it silently disabled the pre-registered statistic for every
        # multi-class cell until it was traced by hand.
        msg = str(e)
        kind = ("scores_not_probabilities" if "sum up to" in msg
                else "class_absent" if "present in y_true" in msg
                else msg[:60])
        _AUC_FAILURES[kind] = _AUC_FAILURES.get(kind, 0) + 1
        return np.nan


def _macro_f1(labels, probs, num_classes=None):
    return f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)


METRIC_FN = {"auc": _auc, "macro_f1": _macro_f1}


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

    per_seed = np.array([
        fn(labels, preds_a[s], num_classes) - fn(labels, preds_b[s], num_classes)
        for s in seeds])
    observed = float(np.nanmean(per_seed))

    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)                        # resample test indices
        ss = rng.choice(seeds, len(seeds), replace=True)   # resample seeds
        y = labels[idx]
        if len(np.unique(y)) < 2:                          # degenerate draw
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
    sd = np.nanstd(per_seed, ddof=1)

    return {"delta": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(p, 1.0)),
            "cohens_d": float(observed / sd) if sd > 0 else np.nan,
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


def benjamini_hochberg(pvals, alpha=None, family_size=None):
    """
    Returns (rejected, adjusted). Adjusted p-values are monotone-enforced.

    family_size overrides the number of tests, so a subset of the declared
    family can still be corrected against the full declared m. Correcting over a
    smaller family than was declared is anti-conservative.
    """
    alpha = alpha or config.ALPHA
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return np.zeros_like(p, bool), p

    m = int(family_size) if family_size else n_ok

    order = np.argsort(np.where(ok, p, np.inf))
    ranked = p[order][:n_ok]
    adj = ranked * m / np.arange(1, n_ok + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity

    out = np.full_like(p, np.nan)
    out[order[:n_ok]] = np.minimum(adj, 1.0)
    return (out <= alpha) & ok, out


# ------------------------------------------------------------------ loading
# Keys that identify WHICH MODEL is being compared, not which experimental
# condition it sits in. Everything else defines the cell.
_COMPARISON_KEYS = ("arm", "seed")


def cell_of(keys):
    """
    The experimental condition a shard belongs to: every key except the two
    that a comparison varies over.

    THIS USED TO BE HARDCODED to (dataset, regime, dim, fp), which silently
    discarded every optional axis. The consequence was severe and invisible:

        12_bottleneck_ablation writes three shards per (dataset, regime, dim,
        seed, arm) - bn=learned, bn=pca, bn=random. All three mapped to the
        same cell, so two were OVERWRITTEN and whichever file sorted last
        became "the" result. The experiment that tests whether the head or the
        bottleneck does the work would have been analysed as though the
        bottleneck policy did not exist.

        10_capacity_sweep is the same: five ranks collapsed into one.

    Deriving the cell from the keys means a new axis is handled correctly the
    moment it is added, rather than requiring this function to be remembered.
    """
    return tuple(sorted((k, str(v)) for k, v in keys.items()
                        if k not in _COMPARISON_KEYS))


def cell_label(cell):
    """Readable one-line description of a cell, for tables."""
    d = dict(cell)
    head = f"{d.get('dataset','?')} n={d.get('regime','?')} d={d.get('dim','4')}"
    extra = [f"{k}={v}" for k, v in cell
             if k not in ("dataset", "regime", "dim", "fp", "aug")]
    return head + ("  " + " ".join(extra) if extra else "")


def collect(experiment):
    """cell -> arm -> seed -> full shard record."""
    tbl = defaultdict(lambda: defaultdict(dict))
    seen = {}
    for r in shards.load_all(experiment):
        k = r["keys"]
        cell = cell_of(k)
        slot = (cell, k["arm"], k["seed"])
        if slot in seen:
            # Two shards claiming the same condition, arm and seed. Impossible
            # if the keys are complete, so it means an axis is missing from the
            # shard key itself - a silent overwrite, and worth stopping for.
            raise RuntimeError(
                f"duplicate shard for cell={cell} arm={k['arm']} seed={k['seed']}\n"
                f"  {seen[slot]}\n  {k}\n"
                f"An experimental axis is missing from the shard key.")
        seen[slot] = k
        tbl[cell][k["arm"]][k["seed"]] = r
    return tbl


def _metric_of(record, metric, condition=None):
    """Scalar metric from a shard, for the seed-level fallback path."""
    if "metrics" in record:
        return record["metrics"].get(metric)
    curve = record.get("curve", {})
    return curve.get(condition or "0.00", {}).get(metric)


def _integrity_check(experiment, tbl, metric, condition, tol=1e-3):
    """
    Recompute the metric from stored predictions and compare with the value the
    shard recorded at training time. They should agree to floating-point noise.

    Cheap insurance against a whole class of silent corruption: wrong dtype,
    truncated file, mismatched labels, or predictions written by a different
    model than the metrics. float16 storage produced a 4.7e-03 gap here before
    it was changed to float32, and nothing else in the pipeline noticed.
    """
    import medmnist
    worst, n_checked, n_bad = 0.0, 0, 0
    for cell in sorted(tbl):
        C = len(medmnist.INFO[dict(cell)["dataset"]]["label"])
        for arm, by_seed in tbl[cell].items():
            for seed, rec in by_seed.items():
                stored = _metric_of(rec, metric, condition)
                if stored is None:
                    continue
                probs, labels = shards.load_predictions(
                    experiment, condition=condition, **rec["keys"])
                if probs is None:
                    continue
                got = METRIC_FN[metric](labels, probs, C)
                if np.isnan(got):
                    continue
                n_checked += 1
                gap = abs(got - stored)
                worst = max(worst, gap)
                n_bad += gap > tol

    if not n_checked:
        print("\nIntegrity: no run had both a stored metric and a readable "
              "prediction file - nothing could be cross-checked.")
        return
    print(f"\nIntegrity: {n_checked} runs re-scored from stored predictions; "
          f"max |recomputed - recorded| = {worst:.2e}")
    if n_bad:
        print(f"WARNING: {n_bad} runs differ by more than {tol:g}. The stored")
        print("predictions are not a faithful record of the run - do not quote")
        print("bootstrap intervals until this is explained.")


def _fallback_banner(results):
    """One place explaining a fallback, so both runners report it identically."""
    if "seed_level_ONLY" not in {r["method"] for r in results}:
        return
    n_fb = sum(1 for r in results if r["method"] == "seed_level_ONLY")
    print("\n" + "!" * 74)
    print(f"! SEED-LEVEL FALLBACK IN USE for {n_fb} of {len(results)} comparisons.")
    print("! These intervals ignore test-set sampling variance and are NOT the")
    print("! pre-registered analysis.")
    if _AUC_FAILURES:
        print("!")
        print("! The metric could not be computed. Reasons:")
        for k, v in sorted(_AUC_FAILURES.items(), key=lambda x: -x[1]):
            print(f"!   {v:>8d}x  {k}")
        if "scores_not_probabilities" in _AUC_FAILURES:
            print("!")
            print("! 'scores_not_probabilities' is a STORAGE fault: saved")
            print("! probabilities do not sum to 1 within sklearn's tolerance.")
            print("! shards.load_predictions renormalises rows - if this still")
            print("! fires, that fix is not in place.")
    else:
        print("! Prediction files were absent for those cells.")
    print("!" * 74)


def compare_records(experiment, rec_a, rec_b, metric, num_classes,
                    condition=None):
    """
    Nested bootstrap over two {seed: shard} maps; seed-level fallback otherwise.

    Deliberately agnostic about WHAT differs between the two sides. Two arms in
    one condition, or one arm in two conditions, are the same statistical
    operation - and separating that choice from the machinery is what lets
    H-S5 (rank 0 vs rank 8, same arm) be tested at all.
    """
    common = sorted(set(rec_a) & set(rec_b))
    if len(common) < 2:
        return None

    pa, pb, labels = {}, {}, None
    for s in common:
        xa = shards.load_predictions(experiment, condition=condition,
                                     **rec_a[s]["keys"])
        xb = shards.load_predictions(experiment, condition=condition,
                                     **rec_b[s]["keys"])
        # load_predictions returns (probs, labels), or (None, None) when the file
        # is missing. A 2-tuple is never None, so the array itself must be tested
        # - otherwise a missing file puts None into the dict and the failure
        # surfaces later, inside the bootstrap, as an opaque crash.
        if xa[0] is None or xb[0] is None:
            pa, pb, labels = {}, {}, None
            break
        pa[s], la = xa
        pb[s], _ = xb
        labels = la

    if pa and labels is not None:
        out = nested_paired_bootstrap(pa, pb, labels, num_classes, metric=metric)
        if out:
            return out

    return seed_level_bootstrap(
        {s: _metric_of(rec_a[s], metric, condition) for s in common},
        {s: _metric_of(rec_b[s], metric, condition) for s in common})


def compare(experiment, tbl, cell, arm_a, arm_b, metric, num_classes,
            condition=None):
    """Two ARMS within one experimental condition."""
    return compare_records(experiment, tbl[cell].get(arm_a, {}),
                           tbl[cell].get(arm_b, {}), metric, num_classes,
                           condition)


def group_by(tbl, key):
    """
    base condition -> {value of `key`: cell}

    Splits the cell table along one axis so the same arm can be compared across
    two settings of it. Cells lacking the key are skipped.
    """
    groups = defaultdict(dict)
    for cell in tbl:
        d = dict(cell)
        if key not in d:
            continue
        val = d.pop(key)
        groups[tuple(sorted(d.items()))][val] = cell
    return groups


def run_cross(experiment, key, val_a, val_b, metric="auc",
              family_size=None, condition=None):
    """
    Compare one arm against ITSELF across two values of an experimental axis.

    WHY THIS EXISTS. run() compares arm_a with arm_b inside a cell, which cannot
    express H-S5: `low_rank` at rank 0 versus `low_rank` at rank 8 is the same
    arm in two different cells. Without this, the capacity sweep - 1,000 runs,
    the experiment that tests the paper's mechanism claim - had no confirmatory
    path at all, and 04 simply printed "Nothing comparable found".

    Also covers bottleneck policy (bn=learned vs bn=pca), weight decay, circuit
    depth, and angle scale, none of which are arm-valued either.
    """
    import medmnist

    tbl = collect(experiment)
    if not tbl:
        print(f"No shards for '{experiment}'.")
        return

    groups = group_by(tbl, key)
    if not groups:
        print(f"No shards carry the key '{key}'. Available keys: "
              f"{sorted({k for c in tbl for k, _ in c})}")
        return

    va, vb = str(val_a), str(val_b)
    results = []
    for base, by_val in sorted(groups.items()):
        if va not in by_val or vb not in by_val:
            continue
        cell_a, cell_b = by_val[va], by_val[vb]
        C = len(medmnist.INFO[dict(base)["dataset"]]["label"])
        for arm in sorted(set(tbl[cell_a]) & set(tbl[cell_b])):
            r = compare_records(experiment, tbl[cell_a][arm], tbl[cell_b][arm],
                                metric, C, condition)
            if r:
                r.update(base=base, arm=arm)
                results.append(r)

    if not results:
        print(f"No cell has both {key}={va} and {key}={vb}.")
        return

    rej, adj = benjamini_hochberg([r["p"] for r in results],
                                  family_size=family_size)
    for r, rr, aa in zip(results, rej, adj):
        r["p_adj"], r["significant"] = float(aa), bool(rr)

    _fallback_banner(results)
    _integrity_check(experiment, tbl, metric, condition)

    print(f"\nBH-FDR family size m = {family_size or len(results)}"
          + (f"  (declared; {len(results)} computed here)" if family_size else ""))
    print(f"\n=== CROSS-CONDITION: {key}={va} minus {key}={vb}  ({metric}) ===")
    print(f"Same arm, same seeds, same data. Only {key} differs.")
    print(f"\n{'condition':34s} {'arm':22s} {'delta':>9s} "
          f"{'95% CI':>21s} {'p':>9s} {'p_adj':>9s} {'d':>7s}  verdict")
    print("-" * 126)
    for r in sorted(results, key=lambda x: (dict(x["base"]).get("dataset", ""),
                                            int(dict(x["base"]).get("regime", 0)),
                                            x["arm"])):
        v = (f"{key}={va} better" if r["ci_lo"] > 0 else
             f"{key}={vb} better" if r["ci_hi"] < 0 else "no difference")
        if abs(r["delta"]) < 0.01 and v != "no difference":
            v += " (negligible)"
        print(f"{cell_label(r['base']):34s} {r['arm']:22s} {r['delta']:+9.4f} "
              f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] {r['p']:9.4f} "
              f"{r['p_adj']:9.4f} {r['cohens_d']:+7.2f}  {v}")

    # trend of the contrast across the scarcity axis
    by_n = defaultdict(list)
    for r in results:
        reg = dict(r["base"]).get("regime")
        if reg is not None:
            by_n[int(reg)].append(r["delta"])
    ns = sorted(by_n)
    if len(ns) >= 3:
        print(f"\n=== Trend of the contrast on log2(shots/class) ===")
        x = np.log2(ns)
        slope = float(np.polyfit(x, [np.mean(by_n[n]) for n in ns], 1)[0])
        rng = np.random.default_rng(7)
        boot = [float(np.polyfit(x, [rng.choice(by_n[n], len(by_n[n])).mean()
                                     for n in ns], 1)[0]) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        for n in ns:
            print(f"    n={n:>4d}  mean delta = {np.mean(by_n[n]):+.4f} "
                  f"({len(by_n[n])} cells)")
        print(f"\n    slope = {slope:+.5f} per doubling  [{lo:+.5f}, {hi:+.5f}]")
        print(f"    Negative slope with a CI excluding 0 = the contrast shrinks")
        print(f"    as data grows. Result: "
              f"{'SUPPORTED' if hi < 0 else 'NOT supported'}")


# ------------------------------------------------------------------ analysis
def run(experiment, metric="auc", latex=False, family_size=None, condition=None):
    import medmnist

    tbl = collect(experiment)
    if not tbl:
        print(f"No shards for '{experiment}'.")
        return

    # Declared family members, in the order the analysis plan lists them.
    pairs = [("PRIMARY", *config.PRIMARY_COMPARISON),
             ("SECONDARY", *config.SECONDARY_COMPARISON)]
    exploratory = [("diagnostic", *config.DIAGNOSTIC_COMPARISON)] \
        if hasattr(config, "DIAGNOSTIC_COMPARISON") else []

    results, expl_results = [], []
    for label, arm_a, arm_b in pairs + exploratory:
        bucket = results if label in ("PRIMARY", "SECONDARY") else expl_results
        for cell in sorted(tbl):
            C = len(medmnist.INFO[dict(cell)["dataset"]]["label"])
            r = compare(experiment, tbl, cell, arm_a, arm_b, metric, C, condition)
            if r:
                r.update(family=label, cell=cell, arm_a=arm_a, arm_b=arm_b)
                bucket.append(r)

    if not results and not expl_results:
        print("Nothing comparable found.")
        return

    # BH across the DECLARED family (primary + secondary), not primary alone.
    if results:
        rej, adj = benjamini_hochberg([r["p"] for r in results],
                                      family_size=family_size)
        for r, rr, aa in zip(results, rej, adj):
            r["p_adj"], r["significant"] = float(aa), bool(rr)

    _fallback_banner(results + expl_results)
    _integrity_check(experiment, tbl, metric, condition)

    m_used = family_size or len(results)
    print(f"\nBH-FDR family size m = {m_used}"
          + (f"  (declared; {len(results)} computed here)" if family_size
             else "  (computed here)"))
    if not family_size:
        print(f"WARNING: docs/analysis_plan.md declares "
              f"{getattr(config, 'DECLARED_FAMILY_SIZE', 17)} tests across several")
        print("experiments. Correcting over fewer than the declared family is")
        print("anti-conservative - pass --family-size for the reported table.")

    for label, arm_a, arm_b in pairs + exploratory:
        rows = [r for r in (results + expl_results) if r["family"] == label]
        if not rows:
            continue
        tag = "" if label in ("PRIMARY", "SECONDARY") else "  [EXPLORATORY, uncorrected]"
        print(f"\n=== {label}: {arm_a} - {arm_b}  ({metric}){tag} ===")
        print(f"{'condition':38s} {'enc':>6s} {'delta':>9s} "
              f"{'95% CI':>21s} {'p':>9s} {'p_adj':>9s} {'d':>7s}  verdict")
        print("-" * 128)
        for r in sorted(rows, key=lambda x: (dict(x["cell"]).get("dataset", ""),
                                             int(dict(x["cell"]).get("regime", 0)))):
            cd = dict(r["cell"])
            enc = "froz" if cd.get("fp", "all") == "all" else "adap"
            v = (f"{r['arm_a']} better" if r["ci_lo"] > 0 else
                 f"{r['arm_b']} better" if r["ci_hi"] < 0 else "no difference")
            if abs(r["delta"]) < 0.01 and v != "no difference":
                v += " (negligible)"
            padj = f"{r['p_adj']:9.4f}" if "p_adj" in r else "        -"
            print(f"{cell_label(r['cell']):38s} {enc:>6s} {r['delta']:+9.4f} "
                  f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] {r['p']:9.4f} {padj} "
                  f"{r['cohens_d']:+7.2f}  {v}")

    # --- H-P2: is the effect monotone in shots per class? -----------------
    primary = [r for r in results if r["family"] == "PRIMARY"]
    print(f"\n=== H-P2: trend of delta on log2(shots/class), PRIMARY family ===")
    by_n = defaultdict(list)
    for r in primary:
        by_n[int(dict(r["cell"])["regime"])].append(r["delta"])
    ns = sorted(by_n)
    if len(ns) >= 3:
        x = np.log2(ns)
        slope = float(np.polyfit(x, [np.mean(by_n[n]) for n in ns], 1)[0])
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
            if r["family"] != "PRIMARY":
                continue
            cd = dict(r["cell"])
            ds = cd["dataset"].replace("mnist", "MNIST")
            padj = f"{r.get('p_adj', float('nan')):.3f}"
            f.write(f"{ds} & {cd['regime']} & {r['delta']:+.4f} & "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] & {padj} & "
                    f"{r['cohens_d']:+.2f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"\nLaTeX written to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="01_frozen")
    p.add_argument("--metric", default="auc", choices=["auc", "macro_f1"])
    p.add_argument("--family-size", type=int, default=None,
                   help="declared BH family size; see docs/analysis_plan.md")
    p.add_argument("--condition", default=None,
                   help="sub-condition for multi-condition runs, e.g. 0.20 for noise")
    p.add_argument("--compare-key", default=None,
                   help="compare one arm across two values of this shard key "
                        "(e.g. rank, bn, wd, L). Needed for H-S5, where the two "
                        "sides are the same arm in different conditions.")
    p.add_argument("--compare-values", nargs=2, default=None,
                   metavar=("A", "B"),
                   help="the two values of --compare-key, as A minus B")
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()

    if args.compare_key:
        if not args.compare_values:
            p.error("--compare-key requires --compare-values A B")
        run_cross(args.experiment, args.compare_key,
                  args.compare_values[0], args.compare_values[1],
                  metric=args.metric, family_size=args.family_size,
                  condition=args.condition)
        return

    run(args.experiment, args.metric, args.latex,
        family_size=args.family_size, condition=args.condition)


if __name__ == "__main__":
    main()
