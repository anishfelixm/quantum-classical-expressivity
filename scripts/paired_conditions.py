"""
Shared engine for Amendments 20 and 21: the same contrast under two training
conditions, compared on identical bootstrap draws.

Imported by scripts/analyse_budget_extension.py and scripts/analyse_clip_check.py.
Not run directly.

WHAT IT REUSES, AND WHY
-----------------------
The statistic is the confirmatory one, taken from src/13_family_table.py
rather than reimplemented: per (dataset, n) cell, the mean over seeds of
AUC(a) - AUC(b); a nested bootstrap that resamples test images and seeds
within the cell (B = 2000); cells pooled with equal weight inside each
replicate; percentile intervals; +1-corrected p-values; the slope on log2 n
fitted inside each replicate. AUC is the vectorised implementation that 13
checks against scikit-learn before every use, and the same check runs here.

PAIRING TWO CONDITIONS
----------------------
A contrast is computed under condition R (the reference namespace) and
condition X (the new one) on the SAME seeds in every cell. Both computations
draw their bootstrap indices from the same generator seed, ft.cell_seed(name,
cell), and a cell has the same number of test images and seeds in both, so
replicate b resamples the same images and the same seed positions under R and
under X. The change X - R is then computed replicate by replicate, which gives
a paired interval for the change itself rather than two intervals to eyeball.

When `name` is the group name 13 or 14 uses for the same contrast ("primary",
"hs1", "hs2"; "x_<name>" for 14's groups), condition R reproduces the
published interval exactly, which is checked and printed as an anchor.
"""
import importlib
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

ft = importlib.import_module("13_family_table")
REGIMES = ft.REGIMES
B_DEFAULT = ft.B_DEFAULT


def context():
    return ft.Ctx()


def _cells_by_dn(ctx, ns):
    """(dataset, regime) -> arm -> seed -> record, for the d = 4 frozen cells of ns."""
    tbl = ctx.s04.collect(ns)
    out = {}
    for cell, arms in tbl.items():
        cd = dict(cell)
        if str(cd.get("dim")) != "4":
            continue
        key = (cd["dataset"], str(cd["regime"]))
        if key in out:
            raise RuntimeError(f"{ns}: two d=4 cells for {key}; an axis is unaccounted for")
        out[key] = arms
    return out


def load_two(ctx, side_a, side_b, seeds=None):
    """
    Cells holding arm A from namespace A and arm B from namespace B, aligned on
    the seeds both have (optionally restricted to `seeds`). side = (ns, arm).

    Returns a list of 13-style cells: {"dataset", "regime", "seeds", "n_seeds",
    "y", "arms": {"a": [S,N,C], "b": [S,N,C]}}. Test labels must agree exactly
    across seeds, arms and namespaces, or this raises.
    """
    (ns_a, arm_a), (ns_b, arm_b) = side_a, side_b
    ta = _cells_by_dn(ctx, ns_a)
    tb = ta if ns_b == ns_a else _cells_by_dn(ctx, ns_b)
    cells = []
    for key in sorted(set(ta) & set(tb), key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else 10**9)):
        A, Bm = ta[key].get(arm_a, {}), tb[key].get(arm_b, {})
        common = sorted(set(A) & set(Bm))
        if seeds is not None:
            common = [s for s in common if s in set(seeds)]
        if len(common) < 2:
            continue
        PA, ya = ft._stack(ctx, ns_a, A, common)
        PB, yb = ft._stack(ctx, ns_b, Bm, common)
        if PA is None or PB is None:
            continue
        if not np.array_equal(ya, yb):
            raise RuntimeError(f"test labels differ: {side_a} vs {side_b} at {key}")
        cells.append(ft._cell(key[0], key[1], common, ya, {"a": PA, "b": PB}))
    return cells


def align(cells_r, cells_x):
    """Restrict both lists to the cells and seeds they share, cell by cell."""
    xr = {(c["dataset"], c["regime"]): c for c in cells_r}
    xx = {(c["dataset"], c["regime"]): c for c in cells_x}
    out_r, out_x = [], []
    for key in sorted(set(xr) & set(xx)):
        r, x = xr[key], xx[key]
        common = sorted(set(r["seeds"]) & set(x["seeds"]))
        if len(common) < 2:
            continue
        if not np.array_equal(r["y"], x["y"]):
            raise RuntimeError(f"test labels differ between conditions at {key}")
        for c, lst in ((r, out_r), (x, out_x)):
            ix = [c["seeds"].index(s) for s in common]
            lst.append(ft._cell(c["dataset"], c["regime"], common, c["y"],
                                {k: v[ix] for k, v in c["arms"].items()}))
    return out_r, out_x


def complete(cells, planned_seeds):
    """None if every (dataset, n) of the primary grid is present with the planned seeds."""
    return ft.completeness(cells, True, planned_seeds)


def _run_jobs(jobs, workers):
    if workers > 1 and len(jobs) > 1:
        mp = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp) as ex:
            return list(ex.map(ft._cell_job, jobs))
    return [ft._cell_job(j) for j in jobs]


def bootstrap(cells, name, B=B_DEFAULT, workers=1):
    """Observed statistic and B replicates per cell, with 13's per-cell seeds."""
    for c in cells:
        for k, P in c["arms"].items():
            ft.selfcheck_metrics(c["y"], P, f"{name}/{c['dataset']}/n={c['regime']}/{k}")
    res = _run_jobs([(c, "diff", B, ft.cell_seed(name, c)) for c in cells], workers)
    return [r[0] for r in res], [r[1] for r in res]


def pooled_diff(cells, name, B=B_DEFAULT, workers=1):
    """Pooled a - b per n, one condition: [{"n", "stat"}]."""
    obs, reps = bootstrap(cells, name, B, workers)
    pooled = ft.pool(cells, obs, reps, True)
    return [{"n": n, "stat": ft.ci_p(pooled[n][0][0], pooled[n][1][:, 0], B)} for n in REGIMES]


def summarise_one(cells, name, B=B_DEFAULT, workers=1):
    """One condition: pooled Delta(n) per n and the slope, as {"R": ...} rows."""
    obs, reps = bootstrap(cells, name, B, workers)
    pooled = ft.pool(cells, obs, reps, True)
    rows = [{"n": n, "R": ft.ci_p(pooled[n][0][0], pooled[n][1][:, 0], B)} for n in REGIMES]
    est, R = ft.regime_slope(pooled)
    return rows, {"R": ft.ci_p(est, R, B)}


def summarise_pair(cells_r, cells_x, name, B=B_DEFAULT, workers=1):
    """
    Pooled Delta(n) under R and under X, the paired change X - R per n, and the
    slope on log2 n under each and its change. Cells must already be aligned.
    """
    obs_r, reps_r = bootstrap(cells_r, name, B, workers)
    obs_x, reps_x = bootstrap(cells_x, name, B, workers)
    pr = ft.pool(cells_r, obs_r, reps_r, True)
    px = ft.pool(cells_x, obs_x, reps_x, True)
    rows = []
    for n in REGIMES:
        (o_r, r_r), (o_x, r_x) = pr[n], px[n]
        rows.append({"n": n,
                     "R": ft.ci_p(o_r[0], r_r[:, 0], B),
                     "X": ft.ci_p(o_x[0], r_x[:, 0], B),
                     "change": ft.ci_p(o_x[0] - o_r[0], r_x[:, 0] - r_r[:, 0], B)})
    x = np.log2(np.array(REGIMES, float))
    o_r = np.array([pr[n][0][0] for n in REGIMES])
    o_x = np.array([px[n][0][0] for n in REGIMES])
    R_r = np.stack([pr[n][1][:, 0] for n in REGIMES], axis=1)
    R_x = np.stack([px[n][1][:, 0] for n in REGIMES], axis=1)
    ok = np.isfinite(R_r).all(1) & np.isfinite(R_x).all(1)
    s_r, s_x = float(ft.slope(x, o_r)), float(ft.slope(x, o_x))
    S_r, S_x = ft.slope(x, R_r[ok]), ft.slope(x, R_x[ok])
    slope = {"R": ft.ci_p(s_r, S_r, B), "X": ft.ci_p(s_x, S_x, B),
             "change": ft.ci_p(s_x - s_r, S_x - S_r, B)}
    return rows, slope


def fmt(s, w=4):
    if not s:
        return "n/a".rjust(27)
    flag = "*" if (s["ci_lo"] > 0 or s["ci_hi"] < 0) else " "
    return f"{s['estimate']:+.{w}f} [{s['ci_lo']:+.{w}f},{s['ci_hi']:+.{w}f}]{flag}"


def excludes_zero(s):
    return bool(s) and (s["ci_lo"] > 0 or s["ci_hi"] < 0)
