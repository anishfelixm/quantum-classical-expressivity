"""
THE CONFIRMATORY TABLE. 23 declared tests, one statistic each, one correction.

    python src/13_family_table.py                       # every group, from cache
    python src/13_family_table.py --groups primary hs2  # compute a subset
    python src/13_family_table.py --recompute           # ignore the cache
    python src/13_family_table.py --selftest            # statistics core only

WHY THIS FILE EXISTS
--------------------
docs/analysis_plan.md section 4 declares 23 tests, pooled across datasets, with
Benjamini-Hochberg applied ONCE across all of them. Until now no script produced
that table. 04_statistical_analysis.py prints per-cell tables - per-dataset
breakdowns, which section 4 itself classes as exploratory - and corrected them
with whatever subset of the family each invocation happened to compute. Every
confirmatory p-value in the manuscript comes from here and nowhere else.

Every computational choice is fixed in Amendment 15, written 1 October 2026,
before this script was first run.

THE STATISTIC: NESTED BOOTSTRAP, POOLED WITH EQUAL WEIGHT PER CELL
-------------------------------------------------------------------
For each cell (dataset x shots-per-class), each replicate resamples the test
indices and the seeds, and recomputes the cell's statistic. A pooled statistic
is the mean over cells within the same replicate, so a dataset with 7,180 test
images carries the same weight as one with 156. This is section 2's statistic,
applied uniformly to every member of the family.

Two decisions that change numbers relative to 04, both stated in Amendment 15:

  H-P2 is the slope of pooled delta(n) on log2(n) computed INSIDE the nested
  bootstrap. 04 resampled the four per-dataset deltas at each n, which ignores
  test-set and seed variance entirely. Both are printed; this one is the
  confirmatory value.

  p-values carry the +1 correction, 2*min((k<=0)+1, (k>=0)+1)/(B+1), so none
  is exactly zero (Davison & Hinkley 1997). Slightly conservative.

THE VERDICT
-----------
Supported  <=>  BH-adjusted p <= 0.05  AND  the estimate has the predicted sign.

m = 23 always. A test whose data are missing or incomplete is "not computed"
and is NOT removed from the family - shrinking m would be anti-conservative.

INTEGRITY CHECKS, RUN EVERY TIME
--------------------------------
  - The fast vectorised AUC and macro-F1 are checked against scikit-learn on
    real predictions before any bootstrap runs. Any mismatch aborts.
  - Test-set labels must be identical across every seed and arm in a cell.
  - The observed H-P1 delta(5) must reproduce 04's value; 04's H-P2 interval is
    recomputed with 04's exact procedure and printed beside the nested one.
  - A group is analysed only when every cell is present with its planned seed
    count; partial data from a sweep still running is reported, never used.
"""
import argparse
import importlib
import json
import os
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

FAMILY_SIZE = 23
ALPHA = 0.05
B_DEFAULT = 2000
REGIMES = (5, 10, 20, 50, 100)
SIGMAS = ("0.05", "0.10", "0.15", "0.20")


# ============================================================ metrics
def auc_batch(y, P):
    """
    ROC AUC for S seeds at once. y: [N] ints; P: [S, N, C] probabilities.

    Binary: AUC of the class-1 score. Multi-class: macro one-vs-rest, the same
    definition 04 uses. Computed by the Mann-Whitney rank identity with average
    ranks for ties, which is exactly scikit-learn's treatment of ties. Returns
    NaN for every seed when a class is absent from y - 04 does the same, because
    a one-vs-rest AUC is undefined for an absent class.

    Why not scikit-learn: the family table needs ~10^7 AUCs, and ranking all
    seeds and classes in one call is ~20x faster. It is checked against
    scikit-learn on real data before every run (selfcheck_metrics).
    """
    S, N, C = P.shape
    if C == 2:
        pos = y == 1
        npos = int(pos.sum())
        nneg = N - npos
        if npos == 0 or nneg == 0:
            return np.full(S, np.nan)
        R = rankdata(P[:, :, 1], axis=1)
        return (R[:, pos].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)

    counts = np.bincount(y, minlength=C)
    if (counts == 0).any():
        return np.full(S, np.nan)
    R = rankdata(P, axis=1)                                  # [S, N, C]
    out = np.empty((S, C))
    for c in range(C):
        pos = y == c
        npos = counts[c]
        out[:, c] = (R[:, pos, c].sum(1) - npos * (npos + 1) / 2) / (npos * (N - npos))
    return out.mean(1)


def f1_batch(y, P):
    """
    Macro-F1 of the argmax prediction for S seeds at once.

    Matches scikit-learn's f1_score(average="macro", zero_division=0): the macro
    average runs over labels present in y_true OR y_pred, per seed.
    """
    S, N, C = P.shape
    yt = np.eye(C, dtype=bool)[y]                            # [N, C]
    yp = P.argmax(2)[..., None] == np.arange(C)              # [S, N, C]
    tp = (yp & yt).sum(1)
    fp = (yp & ~yt).sum(1)
    fn = (~yp & yt).sum(1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    present = yt.any(0)[None, :] | yp.any(1)
    return (f1 * present).sum(1) / present.sum(1)


def selfcheck_metrics(y, P, label, n_check=5, tol=1e-9):
    """Abort if the fast metrics disagree with scikit-learn on real predictions."""
    from sklearn.metrics import f1_score, roc_auc_score
    C = P.shape[2]
    a, f = auc_batch(y, P[:n_check]), f1_batch(y, P[:n_check])
    for s in range(min(n_check, P.shape[0])):
        ref_a = (roc_auc_score(y, P[s, :, 1]) if C == 2 else
                 roc_auc_score(y, P[s], multi_class="ovr", average="macro",
                               labels=np.arange(C)))
        ref_f = f1_score(y, P[s].argmax(1), average="macro", zero_division=0)
        # Written as "not (<= tol)" so a NaN on either side FAILS: every
        # comparison involving NaN is False, and "> tol" would let it through.
        if not (abs(a[s] - ref_a) <= tol and abs(f[s] - ref_f) <= tol):
            raise RuntimeError(
                f"metric self-check FAILED for {label}, seed index {s}: "
                f"auc {a[s]!r} vs sklearn {ref_a!r}; f1 {f[s]!r} vs {ref_f!r}")


def _nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


# ============================================================ cell statistics
# Each takes (cell, idx, ss): idx resamples test images, ss resamples seed
# positions. Every array in a cell is aligned on the same seed list, so one ss
# pairs every arm and every condition on the same seeds.
def stat_diff(cell, idx, ss):
    """Cell delta: mean over seeds of AUC(a) - AUC(b)."""
    y = cell["y"][idx]
    A = cell["arms"]["a"][ss][:, idx]
    B = cell["arms"]["b"][ss][:, idx]
    return np.array([_nanmean(auc_batch(y, A) - auc_batch(y, B))])


def stat_calibration_gap(cell, idx, ss):
    """
    H-S3, one value per sigma: G = relative F1 loss - relative AUC loss, averaged
    over seeds. sigma=0 and sigma share the same resampled images and seeds, so
    every G is paired. A seed whose clean F1 is 0 has no relative loss and is
    skipped.
    """
    y = cell["y"][idx]
    P0 = cell["arms"]["0.00"][ss][:, idx]
    a0, f0 = auc_batch(y, P0), f1_batch(y, P0)
    out = []
    for s in SIGMAS:
        Ps = cell["arms"][s][ss][:, idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (f0 - f1_batch(y, Ps)) / f0 - (a0 - auc_batch(y, Ps)) / a0
        g[(f0 == 0) | ~np.isfinite(g)] = np.nan
        out.append(_nanmean(g))
    return np.array(out)


def stat_encoder_interaction(cell, idx, ss):
    """H-S4: |delta_frozen| - |delta_adaptive|, deltas averaged over seeds first."""
    y = cell["y"][idx]
    a = {k: auc_batch(y, cell["arms"][k][ss][:, idx]) for k in ("vf", "mf", "va", "ma")}
    return np.array([abs(_nanmean(a["vf"] - a["mf"])) - abs(_nanmean(a["va"] - a["ma"]))])


STATS = {"diff": stat_diff, "gap": stat_calibration_gap,
         "encoder": stat_encoder_interaction}


def _cell_job(job):
    """Observed statistic plus B nested-bootstrap replicates for one cell."""
    cell, stat_name, B, seed = job
    stat = STATS[stat_name]
    N, S = len(cell["y"]), cell["n_seeds"]
    obs = stat(cell, np.arange(N), np.arange(S))
    reps = np.full((B, obs.size), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(B):
        idx = rng.integers(0, N, N)
        if np.unique(cell["y"][idx]).size < 2:       # degenerate draw
            continue
        reps[b] = stat(cell, idx, rng.integers(0, S, S))
    return obs, reps


def cell_seed(group, cell):
    """Stable per-cell RNG seed. Python's hash() is salted per process; crc32 is not."""
    return 20261001 + zlib.crc32(f"{group}|{cell['dataset']}|{cell['regime']}".encode())


# ============================================================ summaries
def ci_p(estimate, reps, B=B_DEFAULT):
    """Percentile CI and +1-corrected two-sided bootstrap p-value."""
    r = reps[np.isfinite(reps)]
    if r.size < B // 2:          # more than half the draws degenerate: no inference
        return None
    lo, hi = np.percentile(r, [2.5, 97.5])
    k_le, k_ge = int((r <= 0).sum()), int((r >= 0).sum())
    p = min(1.0, 2 * min(k_le + 1, k_ge + 1) / (r.size + 1))
    return {"estimate": float(estimate), "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(p), "b_valid": int(r.size)}


def slope(x, Y):
    """OLS slope of Y on x along the last axis; vectorised over replicates."""
    xc = x - x.mean()
    return (Y - Y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def pool(cells, obs, reps, by_regime):
    """Equal-weight mean over cells, within each replicate."""
    if not by_regime:
        return {"all": (_nanmean(np.stack(obs), 0), _nanmean(np.stack(reps), 0))}
    out = {}
    for n in REGIMES:
        ix = [i for i, c in enumerate(cells) if int(c["regime"]) == n]
        out[n] = (_nanmean(np.stack([obs[i] for i in ix]), 0),
                  _nanmean(np.stack([reps[i] for i in ix]), 0))
    return out


def regime_slope(pooled):
    x = np.log2(np.array(REGIMES, float))
    o = np.array([pooled[n][0][0] for n in REGIMES])
    R = np.stack([pooled[n][1][:, 0] for n in REGIMES], axis=1)
    R = R[np.isfinite(R).all(1)]
    return float(slope(x, o)), slope(x, R)


def benjamini_hochberg(p, m):
    """BH-adjusted p-values with m fixed at the declared family size."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    n = int(ok.sum())
    out = np.full_like(p, np.nan)
    if n == 0:
        return out
    m = max(m, n)
    order = np.argsort(np.where(ok, p, np.inf))[:n]
    adj = p[order] * m / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def hp2_as_in_04(cells, obs):
    """04's original H-P2 interval, reproduced exactly (rng seed 7, B=2000)."""
    by_n = {n: [float(obs[i][0]) for i, c in enumerate(cells) if int(c["regime"]) == n]
            for n in REGIMES}
    x = np.log2(np.array(REGIMES, float))
    est = float(np.polyfit(x, [np.mean(by_n[n]) for n in REGIMES], 1)[0])
    rng = np.random.default_rng(7)
    boot = [float(np.polyfit(x, [rng.choice(by_n[n], len(by_n[n])).mean()
                                 for n in REGIMES], 1)[0]) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, float(lo), float(hi)


# ============================================================ data (pod only)
class Ctx:
    """Project modules, imported lazily so the statistics core runs without torch."""

    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        src = here if os.path.exists(os.path.join(here, "config.py")) else \
            os.path.join(os.getcwd(), "src")
        sys.path.insert(0, src)
        self.config = importlib.import_module("config")
        self.shards = importlib.import_module("shards")
        self.s04 = importlib.import_module("04_statistical_analysis")


def _stack(ctx, experiment, recs, seeds, condition=None):
    P, y = [], None
    for s in seeds:
        probs, labels = ctx.shards.load_predictions(experiment, condition=condition,
                                                    **recs[s]["keys"])
        if probs is None:
            return None, None
        lab = np.asarray(labels).ravel().astype(int)
        if y is None:
            y = lab
        elif not np.array_equal(y, lab):
            raise RuntimeError(f"{experiment}: test labels differ across seeds")
        P.append(np.asarray(probs, np.float32))
    return np.stack(P), y


def _cell(ds, reg, seeds, y, arms):
    return {"dataset": ds, "regime": str(reg), "seeds": seeds,
            "n_seeds": len(seeds), "y": y, "arms": arms}


def load_pairs(ctx, experiment, arm_a, arm_b, keep=None):
    """One cell per (dataset, n): arm_a and arm_b on their common seeds."""
    tbl = ctx.s04.collect(experiment)
    cells = []
    for cell in sorted(tbl):
        cd = dict(cell)
        if keep and not keep(cd):
            continue
        A, B = tbl[cell].get(arm_a, {}), tbl[cell].get(arm_b, {})
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            continue
        PA, ya = _stack(ctx, experiment, A, seeds)
        PB, yb = _stack(ctx, experiment, B, seeds)
        if PA is None or PB is None:
            continue
        if not np.array_equal(ya, yb):
            raise RuntimeError(f"{experiment}: labels differ between arms")
        cells.append(_cell(cd["dataset"], cd["regime"], seeds, ya, {"a": PA, "b": PB}))
    return cells


def load_cross(ctx, experiment, key, va, vb, arm):
    """Same arm at two values of `key` (H-S5: low_rank at rank 0 vs rank 8)."""
    tbl = ctx.s04.collect(experiment, force_keys=(key,))
    cells = []
    for base, by_val in sorted(ctx.s04.group_by(tbl, key).items()):
        if va not in by_val or vb not in by_val:
            continue
        A, B = tbl[by_val[va]].get(arm, {}), tbl[by_val[vb]].get(arm, {})
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            continue
        PA, ya = _stack(ctx, experiment, A, seeds)
        PB, yb = _stack(ctx, experiment, B, seeds)
        if PA is None or PB is None or not np.array_equal(ya, yb):
            continue
        bd = dict(base)
        cells.append(_cell(bd["dataset"], bd["regime"], seeds, ya, {"a": PA, "b": PB}))
    return cells


def load_noise(ctx, experiment, arm):
    """H-S3: one arm, clean and four noise levels, aligned on seed."""
    tbl = ctx.s04.collect(experiment)
    cells = []
    for cell in sorted(tbl):
        recs = tbl[cell].get(arm, {})
        seeds = sorted(recs)
        if len(seeds) < 2:
            continue
        arms, y0 = {}, None
        for cond in ("0.00",) + SIGMAS:
            P, y = _stack(ctx, experiment, recs, seeds, condition=cond)
            if P is None:
                arms = None
                break
            if y0 is not None and not np.array_equal(y, y0):
                raise RuntimeError(f"{experiment}: labels differ across sigma")
            arms[cond], y0 = P, y
        if arms:
            cd = dict(cell)
            cells.append(_cell(cd["dataset"], cd["regime"], seeds, y0, arms))
    return cells


def load_encoder(ctx, experiment):
    """H-S4: frozen and adaptive, both arms, on seeds common to all four."""
    tbl = ctx.s04.collect(experiment)
    by = {}
    for cell in tbl:
        cd = dict(cell)
        by.setdefault((cd["dataset"], cd["regime"]), {})[cd.get("fp")] = cell
    cells = []
    for (ds, reg), fps in sorted(by.items()):
        if "all" not in fps or "layer3_only" not in fps:
            continue
        f, a = tbl[fps["all"]], tbl[fps["layer3_only"]]
        src = {"vf": f.get("quantum_vqc", {}), "mf": f.get("matched_param_fullrank", {}),
               "va": a.get("quantum_vqc", {}), "ma": a.get("matched_param_fullrank", {})}
        seeds = sorted(set.intersection(*(set(v) for v in src.values())))
        if len(seeds) < 2:
            continue
        arms, y0 = {}, None
        for k, recs in src.items():
            P, y = _stack(ctx, experiment, recs, seeds)
            if P is None or (y0 is not None and not np.array_equal(y, y0)):
                arms = None
                break
            arms[k], y0 = P, y
        if arms:
            cells.append(_cell(ds, reg, seeds, y0, arms))
    return cells


# ============================================================ the family
# name -> (namespace, loader, statistic, regime-grouped?, planned seeds)
# Planned seeds of None means "whatever was run, but identical across cells".
def _bn(policy):
    return lambda cd: cd.get("bn", "learned") == policy


GROUPS = {
    "primary": ("01_frozen_tuned", lambda c, e: load_pairs(c, e, "quantum_vqc", "matched_param_fullrank"), "diff", True, 40),
    "hs1":     ("01_frozen_tuned", lambda c, e: load_pairs(c, e, "quantum_reupload", "quantum_vqc"), "diff", True, 40),
    "hs2":     ("01_frozen_tuned", lambda c, e: load_pairs(c, e, "quantum_vqc", "fourier_rff"), "diff", True, 40),
    "hs3":     ("03_robustness_v2", lambda c, e: load_noise(c, e, "quantum_vqc"), "gap", False, 10),
    "hs4":     ("25_encoder", load_encoder, "encoder", False, 10),
    "hs5":     ("10_capacity", lambda c, e: load_cross(c, e, "rank", "0", "8", "low_rank"), "diff", True, None),
    "hs6_pca": ("12_bottleneck", lambda c, e: load_pairs(c, e, "quantum_vqc", "matched_param_fullrank", _bn("pca")), "diff", True, None),
    "hs6_random": ("12_bottleneck", lambda c, e: load_pairs(c, e, "quantum_vqc", "matched_param_fullrank", _bn("random")), "diff", True, None),
    "hs7a":    ("14_readout", lambda c, e: load_pairs(c, e, "quantum_rich", "quantum_rich_padded"), "diff", False, 10),
    "hs7b":    ("14_readout", lambda c, e: load_pairs(c, e, "quantum_rich", "quantum_vqc"), "diff", False, 10),
}

# test id, group, how the statistic is read, predicted sign (+1, -1, 0 = none),
# and what it measures.
TESTS = (
    [("H-P1", "primary", ("regime", 5), +1, "VQC - MPFR, delta(5) pooled"),
     ("H-P2", "primary", ("slope",), -1, "slope of delta on log2 n")]
    + [(f"H-S1 n={n}", "hs1", ("regime", n), {5: -1, 10: -1, 20: 0, 50: +1, 100: +1}[n],
        "reupload - VQC") for n in REGIMES]
    + [(f"H-S2 n={n}", "hs2", ("regime", n), -1, "VQC - Fourier RFF") for n in REGIMES]
    + [(f"H-S3 s={s}", "hs3", ("index", i), +1, "VQC rel F1 loss - rel AUC loss")
       for i, s in enumerate(SIGMAS)]
    + [("H-S4", "hs4", ("index", 0), +1, "|delta frozen| - |delta adaptive|"),
       ("H-S5a", "hs5", ("regime", 5), +1, "low_rank r0 - r8, delta(5)"),
       ("H-S5b", "hs5", ("slope",), -1, "slope of r0 - r8 delta"),
       ("H-S6 pca", "hs6_pca", ("regime", 5), +1, "VQC - MPFR delta(5), PCA bottleneck"),
       ("H-S6 random", "hs6_random", ("regime", 5), +1, "VQC - MPFR delta(5), random bottleneck"),
       ("H-S7a", "hs7a", ("index", 0), +1, "rich - padded, all cells"),
       ("H-S7b", "hs7b", ("index", 0), +1, "rich - VQC, all cells")]
)
assert len(TESTS) == FAMILY_SIZE, len(TESTS)


def completeness(cells, by_regime, planned):
    """Reason the group cannot be analysed, or None."""
    if not cells:
        return "no data"
    need = 20                                  # 4 datasets x 5 regimes, always
    if len(cells) < need:
        return f"{len(cells)}/{need} cells"
    from collections import Counter
    per_reg = Counter(c["regime"] for c in cells)
    if sorted(per_reg) != sorted(str(n) for n in REGIMES) or set(per_reg.values()) != {4}:
        return f"cells per regime {dict(per_reg)}, need 4 at each of {REGIMES}"
    if len({c["dataset"] for c in cells}) != 4:
        return "need 4 datasets"
    seeds = {c["n_seeds"] for c in cells}
    if planned is not None and seeds != {planned}:
        return f"seeds per cell {sorted(seeds)}, planned {planned}"
    if len(seeds) != 1:
        return f"unequal seeds across cells {sorted(seeds)}"
    return None


def run_group(ctx, name, B, workers, cache_dir, recompute):
    experiment, loader, stat, by_regime, planned = GROUPS[name]
    t0 = time.time()
    cells = loader(ctx, experiment)
    why = completeness(cells, by_regime, planned)
    if why:
        print(f"  {name:11s} {experiment:18s} NOT COMPUTED - {why}")
        return {"status": why}

    for c in cells:                                        # integrity, every run
        for k, P in c["arms"].items():
            selfcheck_metrics(c["y"], P, f"{name}/{c['dataset']}/n={c['regime']}/{k}")

    meta = {"experiment": experiment, "stat": stat, "B": B,
            "cells": [[c["dataset"], c["regime"], c["seeds"]] for c in cells]}
    path = os.path.join(cache_dir, f"{name}.npz")
    if not recompute and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["meta"]) == json.dumps(meta):
            obs, reps = list(z["obs"]), list(z["reps"])
            print(f"  {name:11s} {experiment:18s} cached ({len(cells)} cells, "
                  f"{cells[0]['n_seeds']} seeds)")
            return {"status": "ok", "cells": cells, "obs": obs, "reps": reps,
                    "by_regime": by_regime}

    jobs = [(c, stat, B, cell_seed(name, c)) for c in cells]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_cell_job, jobs))
    else:
        res = [_cell_job(j) for j in jobs]
    obs, reps = [r[0] for r in res], [r[1] for r in res]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, obs=np.stack(obs), reps=np.stack(reps), meta=json.dumps(meta))
    print(f"  {name:11s} {experiment:18s} computed ({len(cells)} cells, "
          f"{cells[0]['n_seeds']} seeds) in {time.time() - t0:.0f}s")
    return {"status": "ok", "cells": cells, "obs": obs, "reps": reps,
            "by_regime": by_regime}


def assemble(groups, B=B_DEFAULT):
    rows = []
    for tid, g, how, sign, what in TESTS:
        G = groups.get(g, {"status": "not requested and not cached"})
        if G["status"] != "ok":
            rows.append({"test": tid, "what": what, "sign": sign, "status": G["status"]})
            continue
        pooled = pool(G["cells"], G["obs"], G["reps"], G["by_regime"])
        if how[0] == "regime":
            o, r = pooled[how[1]]
            s = ci_p(o[0], r[:, 0], B)
        elif how[0] == "slope":
            est, R = regime_slope(pooled)
            s = ci_p(est, R, B)
        else:
            o, r = pooled["all"]
            s = ci_p(o[how[1]], r[:, how[1]], B)
        if s is None:
            rows.append({"test": tid, "what": what, "sign": sign,
                         "status": "too few valid replicates"})
            continue
        rows.append({"test": tid, "what": what, "sign": sign, "status": "ok", **s})

    adj = benjamini_hochberg([r.get("p", np.nan) for r in rows], FAMILY_SIZE)
    for r, a in zip(rows, adj):
        if r["status"] != "ok":
            r["verdict"] = "not computed"
            continue
        r["p_adj"] = float(a)
        sig = a <= ALPHA
        if r["sign"] == 0:
            r["verdict"] = "difference (no prediction)" if sig else "no difference"
        elif sig and np.sign(r["estimate"]) == r["sign"]:
            r["verdict"] = "SUPPORTED"
        elif sig:
            r["verdict"] = "OPPOSITE to prediction"
        else:
            r["verdict"] = "not supported"
    return rows


def report(rows, groups, out_dir, B=B_DEFAULT):
    if B != B_DEFAULT:
        print("\n" + "!" * 74)
        print(f"! B = {B}, not {B_DEFAULT}. THIS IS NOT THE CONFIRMATORY TABLE.")
        print("! Testing only - do not quote any number below.")
        print("!" * 74)
    print(f"\n=== CONFIRMATORY FAMILY: {FAMILY_SIZE} tests, BH-FDR m = {FAMILY_SIZE} ===")
    print(f"Nested bootstrap, equal weight per cell, B={B}, +1-corrected p.")
    print("Supported = p_adj <= 0.05 AND estimate has the predicted sign.\n")
    print(f"{'test':12s} {'estimate':>9s} {'95% CI':>21s} {'p':>7s} {'p_adj':>7s} "
          f"{'pred':>5s}  verdict")
    print("-" * 92)
    for r in rows:
        pred = {1: "+", -1: "-", 0: "none"}[r["sign"]]
        if r["status"] != "ok":
            print(f"{r['test']:12s} {'':>9s} {'':>21s} {'':>7s} {'':>7s} {pred:>5s}  "
                  f"NOT COMPUTED ({r['status']})")
            continue
        print(f"{r['test']:12s} {r['estimate']:+9.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"{r['p']:7.4f} {r['p_adj']:7.4f} {pred:>5s}  {r['verdict']}")
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{n_ok}/{FAMILY_SIZE} computed. Missing tests stay in the family: m is never "
          f"reduced below {FAMILY_SIZE}.")

    P = groups.get("primary", {})
    if P.get("status") == "ok":
        pooled = pool(P["cells"], P["obs"], P["reps"], True)
        print(f"\nAnchor: observed H-P1 delta(5) = {pooled[5][0][0]:+.4f} "
              f"(04 reported +0.0142; must agree)")
        est, lo, hi = hp2_as_in_04(P["cells"], P["obs"])
        hp2 = next(r for r in rows if r["test"] == "H-P2")
        print(f"H-P2 as computed by 04 (dataset-level resampling): {est:+.5f} "
              f"[{lo:+.5f}, {hi:+.5f}]   (04 reported [-0.00844, -0.00127])")
        if hp2["status"] == "ok":
            print(f"H-P2 nested (confirmatory, Amendment 15):          "
                  f"{hp2['estimate']:+.5f} [{hp2['ci_lo']:+.5f}, {hp2['ci_hi']:+.5f}]")

    for g in ("hs5", "hs6_pca", "hs6_random"):
        G = groups.get(g, {})
        if G.get("status") == "ok":
            est, R = regime_slope(pool(G["cells"], G["obs"], G["reps"], True))
            s = ci_p(est, R, B)
            if s:
                print(f"{g:11s} slope (descriptive): {s['estimate']:+.5f} "
                      f"[{s['ci_lo']:+.5f}, {s['ci_hi']:+.5f}]")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "family_table.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(out_dir, "family_table.tex"), "w") as f:
        f.write("% generated by 13_family_table.py\n\\begin{tabular}{lrrrrl}\\toprule\n")
        f.write("Test & Estimate & 95\\% CI & $p$ & $p_{adj}$ & Verdict \\\\\\midrule\n")
        for r in rows:
            if r["status"] == "ok":
                f.write(f"{r['test']} & {r['estimate']:+.4f} & [{r['ci_lo']:+.4f}, "
                        f"{r['ci_hi']:+.4f}] & {r['p']:.4f} & {r['p_adj']:.4f} & "
                        f"{r['verdict']} \\\\\n")
            else:
                f.write(f"{r['test']} & \\multicolumn{{5}}{{l}}{{not computed}} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
    print(f"\nWritten: {out_dir}/family_table.json, family_table.tex")


# ============================================================ self-test
def selftest():
    """Statistics core only - runs anywhere numpy, scipy and scikit-learn exist."""
    rng = np.random.default_rng(0)
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    for C in (2, 8):
        y = rng.integers(0, C, 300)
        P = rng.dirichlet(np.ones(C), size=(6, 300)).astype(np.float32)
        P[:, :50] = P[:, :1]                                     # force ties
        try:
            selfcheck_metrics(y, P, f"C={C}", n_check=6)
            chk(f"C={C}: AUC and F1 match scikit-learn, with ties", True)
        except RuntimeError as e:
            chk(f"C={C}: {e}", False)
        idx = rng.integers(0, 300, 300)                          # bootstrap duplicates
        try:
            selfcheck_metrics(y[idx], P[:, idx], f"C={C} resampled", n_check=6)
            chk(f"C={C}: match on a resample with duplicated indices", True)
        except RuntimeError as e:
            chk(f"C={C}: {e}", False)
    y = np.zeros(100, int); y[:3] = 1; y[3:5] = 2
    Pz = np.tile(np.eye(3)[0], (2, 100, 1)).astype(np.float32)   # predicts one class
    from sklearn.metrics import f1_score
    chk("macro-F1 with classes absent from predictions",
        abs(f1_batch(y, Pz)[0] - f1_score(y, Pz[0].argmax(1), average="macro",
                                          zero_division=0)) < 1e-12)
    y4 = np.r_[np.zeros(50, int), np.ones(50, int) * 2]          # class 1 absent of 3
    chk("AUC is NaN when a class is absent (as in 04)",
        np.isnan(auc_batch(y4, rng.dirichlet(np.ones(3), (2, 100)))).all())

    x = np.log2(np.array(REGIMES, float))
    Y = rng.normal(size=(7, 5))
    chk("vectorised slope equals np.polyfit",
        np.allclose(slope(x, Y), [np.polyfit(x, Y[i], 1)[0] for i in range(7)]))

    p = np.array([0.001, 0.01, 0.02, 0.04, np.nan])
    chk("BH, m=23: 0.01 at rank 2 -> 0.115", abs(benjamini_hochberg(p, 23)[1] - 0.115) < 1e-12)
    chk("BH keeps NaN as not computed", np.isnan(benjamini_hochberg(p, 23)[4]))
    chk("+1 correction: never exactly 0", ci_p(1.0, np.ones(2000))["p"] > 0)

    # Coverage: true effect +0.05 on AUC is recovered with a CI that covers it.
    N, S = 400, 20
    y = rng.integers(0, 2, N)
    def scores(shift):
        return np.stack([np.clip(np.where(y == 1, 0.5 + shift, 0.5) +
                                 rng.normal(0, 0.3, N), 0, 1) for _ in range(S)])
    pa, pb = scores(0.30), scores(0.25)
    cell = {"dataset": "t", "regime": "5", "y": y, "n_seeds": S,
            "arms": {"a": np.stack([1 - pa, pa], -1), "b": np.stack([1 - pb, pb], -1)}}
    obs, reps = _cell_job((cell, "diff", 2000, 1))
    s = ci_p(obs[0], reps[:, 0])
    truth = _nanmean(auc_batch(y, cell["arms"]["a"]) - auc_batch(y, cell["arms"]["b"]))
    chk(f"nested CI covers the observed effect ({s['estimate']:+.4f} in "
        f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}])",
        s["ci_lo"] < truth < s["ci_hi"] and s["ci_lo"] > 0)
    cell0 = dict(cell, arms={"a": cell["arms"]["b"], "b": cell["arms"]["b"]})
    obs0, reps0 = _cell_job((cell0, "diff", 2000, 2))
    s0 = ci_p(obs0[0], reps0[:, 0])
    chk("identical arms: estimate 0, p = 1", obs0[0] == 0 and s0["p"] == 1.0)
    print("\n" + ("SELF-TEST PASSED" if ok else "SELF-TEST FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--workers", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--namespace", nargs="+", default=[], metavar="GROUP=NS",
                    help="override a group's shard namespace, e.g. hs5=10_capacity_sweep")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--B", type=int, default=B_DEFAULT,
                    help=f"testing only; the confirmatory table uses {B_DEFAULT}")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    for item in args.namespace:
        g, ns = item.split("=", 1)
        GROUPS[g] = (ns,) + GROUPS[g][1:]

    ctx = Ctx()
    out_dir = ctx.config.ARTIFACT_ROOT
    cache_dir = os.path.join(out_dir, "family_table_cache")
    print(f"Family table | workers={args.workers} | B={args.B} | "
          f"sha={ctx.config.git_sha()[:8]}")
    groups = {}
    for g in GROUPS:
        if g in args.groups:
            groups[g] = run_group(ctx, g, args.B, args.workers, cache_dir,
                                  args.recompute)
        elif os.path.exists(os.path.join(cache_dir, f"{g}.npz")):
            groups[g] = run_group(ctx, g, args.B, args.workers, cache_dir, False)
    report(assemble(groups, args.B), groups, out_dir, args.B)


if __name__ == "__main__":
    main()
