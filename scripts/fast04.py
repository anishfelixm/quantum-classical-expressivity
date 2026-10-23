"""
04_statistical_analysis.py, unchanged, with its two bootstrap loops made fast.

    python fast04.py --experiment X_15_dim8 --family-size 23      (same arguments as 04)

WHY
---
04 computes every bootstrap AUC with scikit-learn, one seed and one arm at a
time. On PathMNIST (7,180 test images, 9 classes) one cell costs 40,000 such
calls, about 13 minutes, and the clipping comparison of Amendment 21 needs
every reported table twice. Projected from PathMNIST, that is hundreds of
CPU-hours. This file changes how the AUCs inside the two bootstrap loops are
computed, and nothing else: cells, draws, pooling, leave-one-out, BH, verdicts
and printing are 04's own code.

WHAT IS REPLACED, AND WHY IT GIVES THE SAME NUMBERS
---------------------------------------------------
Only nested_paired_bootstrap and pooled_nested_bootstrap. Each fast version:
  - draws from the random generator in exactly the same order as 04, so every
    replicate uses the same test indices and the same seeds;
  - computes the observed deltas with 04's own scikit-learn AUC;
  - for each replicate, computes the AUC of every seed and arm on the resampled
    test set in one vectorised pass, then averages over the drawn seeds exactly
    as 04 does.
A bootstrap resample is the original test set with integer weights (how many
times each image was drawn). AUC with ties counted one half (scikit-learn's
trapezoidal area) equals the weighted Mann-Whitney count, which needs no
re-sorting: each score column is sorted once. All sums are of integers, so the
only difference from scikit-learn is the last rounding of a division, ~1e-16.
Anything unusual (a class absent from the draw, rows not summing to 1, a metric
other than AUC, labels outside 0..C-1) is computed by 04's original code.

CHECKS, ALWAYS ON (any failure aborts the run)
----------------------------------------------
  1. In every bootstrap call, the first replicate's fast AUCs are compared with
     scikit-learn for every seed and arm: they must agree within 1e-12.
  2. The first three calls of each function are also run through 04's original
     function with B = 30, and any difference in delta, interval or p-value is
     printed as a "[fast04] NOTE" line (not an abort: a p-value can move by 1/B
     when two arms tie exactly and scikit-learn's rounding breaks the tie).
  3. Downstream, part2.py compares the clipped-side tables with the logs the
     manuscript reports (computed by the original 04): row by row.
"""
import importlib.util
import os
import sys

import numpy as np

REPO = os.environ.get("PART2_REPO", "/content/repo")
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

spec = importlib.util.spec_from_file_location("s04", os.path.join(SRC, "04_statistical_analysis.py"))
s04 = importlib.util.module_from_spec(spec)
sys.modules["s04"] = s04
spec.loader.exec_module(s04)

ORIG_NESTED = s04.nested_paired_bootstrap
ORIG_POOLED = s04.pooled_nested_bootstrap
AUC = s04._auc
TOL = 1e-12
CHECKS = {"nested": 0, "pooled": 0}


class Engine:
    """AUC of K probability matrices (seeds x arms) on weighted resamples of one test set."""

    def __init__(self, mats, labels, num_classes):
        y = np.asarray(labels)
        self.n = len(y)
        self.binary = num_classes == 2
        self.K = len(mats)
        C = mats[0].shape[1]
        self.ok = (y.min() >= 0 and y.max() < C and all(m.shape == mats[0].shape for m in mats)
                   and np.issubdtype(y.dtype, np.integer)
                   and all(np.isfinite(m).all() for m in mats))
        if not self.binary:
            self.ok = self.ok and all(np.allclose(1, m.sum(axis=1)) for m in mats)
        if not self.ok:
            return
        cls = [1] if self.binary else list(range(C))
        self.ncls = len(cls)
        n = self.n
        order, pos, gs, ge = [], [], [], []
        off = 0
        for m in mats:
            for c in cls:
                sc = m[:, c]
                o = np.argsort(sc, kind="mergesort")
                s = sc[o]
                start = np.r_[True, s[1:] != s[:-1]]              # first element of a tie group
                end = np.r_[s[1:] != s[:-1], True]                # last element of a tie group
                first = np.flatnonzero(start)
                last = np.flatnonzero(end)
                grp = np.cumsum(start) - 1
                order.append(o)
                pos.append(y[o] == c)
                gs.append(first[grp] + off)
                ge.append(last[grp] + off)
                off += n
        self.ncol = off // n
        self.order = np.concatenate(order)
        self.pos = np.concatenate(pos).astype(np.float64)
        self.neg = 1.0 - self.pos
        self.gs = np.concatenate(gs)
        self.ge = np.concatenate(ge)
        self.ties = bool((self.gs != self.ge).any())

    def eval(self, idx):
        """
        AUC per matrix for the resample idx; None if a class is absent (04's own
        code then handles that replicate). Columns are blocks of n sorted scores.
        For each positive: weight x (negative weight strictly below its tie group
        + half the negative weight inside it). Every quantity is a sum of
        integers, so exact; only the final division rounds.
        """
        n, k = self.n, self.ncol
        w = np.bincount(idx, minlength=n).astype(np.float64)[self.order]
        wp = w * self.pos
        wn = w * self.neg
        cn = np.cumsum(wn)                       # inclusive, global
        ex = cn - wn                             # exclusive, global
        col0 = ex.reshape(k, n)[:, :1]           # negatives before each column
        if self.ties:
            below = ex[self.gs].reshape(k, n) - col0
            tied = (cn[self.ge] - ex[self.gs]).reshape(k, n)
            num = (wp.reshape(k, n) * (below + 0.5 * tied)).sum(1)
        else:
            num = (wp.reshape(k, n) * (ex.reshape(k, n) - col0)).sum(1)
        P = wp.reshape(k, n).sum(1)
        N = wn.reshape(k, n).sum(1)
        if (P == 0).any() or (N == 0).any():
            return None
        auc = (num / (P * N)).reshape(self.K, self.ncls)
        return auc[:, 0] if self.binary else np.array([np.average(r) for r in auc])


def _check_first(eng, y_full, mats, idx, auc, num_classes, where):
    ref = np.array([AUC(y_full[idx], m[idx], num_classes) for m in mats])
    bad = ~(np.abs(ref - auc) <= TOL)
    if bad.any():
        raise RuntimeError(f"fast04 AUC CHECK FAILED in {where}: fast {auc[bad][:3]} vs sklearn {ref[bad][:3]}")


def fast_nested(preds_a, preds_b, labels, num_classes, metric="auc", B=None, rng=None):
    if metric != "auc":
        return ORIG_NESTED(preds_a, preds_b, labels, num_classes, metric=metric, B=B, rng=rng)
    seeds = sorted(set(preds_a) & set(preds_b))
    if len(seeds) < 2:
        return None
    mats = [preds_a[s] for s in seeds] + [preds_b[s] for s in seeds]
    eng = Engine(mats, labels, num_classes)
    if not eng.ok:
        return ORIG_NESTED(preds_a, preds_b, labels, num_classes, metric=metric, B=B, rng=rng)
    if CHECKS["nested"] < 3 and B is None and rng is None:
        CHECKS["nested"] += 1
        a = ORIG_NESTED(preds_a, preds_b, labels, num_classes, metric=metric, B=30)
        b = _nested(preds_a, preds_b, labels, num_classes, eng, seeds, mats, 30, None)
        _same(a, b, "nested_paired_bootstrap")
    return _nested(preds_a, preds_b, labels, num_classes, eng, seeds, mats, B, rng)


def _nested(preds_a, preds_b, labels, num_classes, eng, seeds, mats, B, rng):
    # --- 04's code, with the replicate AUCs taken from the engine ---
    B = B or s04.config.BOOTSTRAP_RESAMPLES
    rng = rng or np.random.default_rng(20260814)
    fn = AUC
    n = len(labels)
    S = len(seeds)
    at = {s: i for i, s in enumerate(seeds)}

    per_seed = np.array([
        fn(labels, preds_a[s], num_classes) - fn(labels, preds_b[s], num_classes)
        for s in seeds])
    observed = float(np.nanmean(per_seed))

    deltas = np.empty(B)
    checked = False
    for b in range(B):
        idx = rng.integers(0, n, n)
        ss = rng.choice(seeds, len(seeds), replace=True)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            deltas[b] = np.nan
            continue
        auc = eng.eval(idx)
        if auc is None:
            deltas[b] = np.mean([
                fn(y, preds_a[s][idx], num_classes) - fn(y, preds_b[s][idx], num_classes)
                for s in ss])
            continue
        if not checked:
            _check_first(eng, labels, mats, idx, auc, num_classes, "nested_paired_bootstrap")
            checked = True
        deltas[b] = np.mean([auc[at[s]] - auc[S + at[s]] for s in ss])

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


def fast_pooled(cells, metric="auc", B=None, rng=None):
    if metric != "auc" or len(cells) < 2:
        return ORIG_POOLED(cells, metric=metric, B=B, rng=rng)
    prep = []
    for pa, pb, y, C in cells:
        sds = sorted(set(pa) & set(pb))
        mats = [pa[s] for s in sds] + [pb[s] for s in sds]
        eng = Engine(mats, y, C)
        if not eng.ok:
            return ORIG_POOLED(cells, metric=metric, B=B, rng=rng)
        prep.append((eng, sds, mats))
    if CHECKS["pooled"] < 3 and rng is None:
        CHECKS["pooled"] += 1
        a = ORIG_POOLED(cells, metric=metric, B=30)
        b = _pooled(cells, prep, 30, None)
        _same(a, b, "pooled_nested_bootstrap")
    return _pooled(cells, prep, B, rng)


def _pooled(cells, prep, B, rng):
    # --- 04's code, with the replicate AUCs taken from the engines ---
    B = B or s04.config.BOOTSTRAP_RESAMPLES
    rng = rng or np.random.default_rng(20260814)
    fn = AUC

    per_cell_obs = []
    for pa, pb, y, C in cells:
        sds = sorted(set(pa) & set(pb))
        per_cell_obs.append(np.nanmean([fn(y, pa[s], C) - fn(y, pb[s], C)
                                        for s in sds]))
    observed = float(np.nanmean(per_cell_obs))

    checked = [False] * len(cells)
    deltas = np.full(B, np.nan)
    for b in range(B):
        per_cell = []
        for ci, (pa, pb, y, C) in enumerate(cells):
            eng, sds, mats = prep[ci]
            idx = rng.integers(0, len(y), len(y))
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            ss = rng.choice(sds, len(sds), replace=True)
            auc = eng.eval(idx)
            if auc is None:
                d = np.mean([fn(yb, pa[s][idx], C) - fn(yb, pb[s][idx], C)
                             for s in ss])
            else:
                if not checked[ci]:
                    _check_first(eng, y, mats, idx, auc, C, "pooled_nested_bootstrap")
                    checked[ci] = True
                at = {s: i for i, s in enumerate(sds)}
                S = len(sds)
                d = np.mean([auc[at[s]] - auc[S + at[s]] for s in ss])
            if not np.isnan(d):
                per_cell.append(d)
        if per_cell:
            deltas[b] = np.mean(per_cell)

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < B // 2:
        return None
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    sd = np.nanstd(per_cell_obs, ddof=1)
    d_ok = len(cells) >= 3 and sd > 0
    return {"delta": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(p, 1.0)),
            "cohens_d": float(observed / sd) if d_ok else np.nan,
            "n_datasets": len(cells), "method": "pooled_nested_bootstrap"}


NOTES = []


def _same(a, b, where):
    def note(msg):
        NOTES.append(msg)
        print(f"[fast04] NOTE {where}: {msg}", file=sys.stderr, flush=True)
    if (a is None) != (b is None):
        return note(f"original {a} vs fast {b}")
    if a is None:
        return
    for k in a:
        x, y = a[k], b[k]
        both_nan = isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y)
        if isinstance(x, float) and not both_nan:
            if not (x == y or abs(x - y) <= TOL):
                note(f"{k}: original {x!r} vs fast {y!r}")
        elif x != y and not both_nan:
            note(f"{k}: original {x!r} vs fast {y!r}")


s04.nested_paired_bootstrap = fast_nested
s04.pooled_nested_bootstrap = fast_pooled

if __name__ == "__main__":
    sys.argv[0] = os.path.join(SRC, "04_statistical_analysis.py")
    s04.main()
    print(f"\n[fast04] AUC check passed in every bootstrap call; function checks run: "
          f"nested {CHECKS['nested']}, pooled {CHECKS['pooled']}; differences from 04: "
          f"{len(NOTES)}" + "".join(f"\n[fast04]   {n}" for n in NOTES))
