"""
EXPLORATORY COMPANIONS TO THE CONFIRMATORY TABLE.

    python src/14_exploratory_table.py --workers 7
    python src/14_exploratory_table.py --groups reupload_vs_control --workers 7

None of these is a member of the 23-test confirmatory family, and none can
change a verdict in it. Each analysis is corrected by Benjamini-Hochberg WITHIN
itself, and the manuscript labels every number here as exploratory.

WHY EACH ONE EXISTS
-------------------
reupload_vs_control  NOT pre-specified. H-S1 showed re-uploading beats the
    single-encoding VQC at n >= 10, which makes "does it beat the classical
    control?" unavoidable: because all arms share the same 40 seeds and 20
    cells, the observed answer is exactly H-S1 + H-P at each n, so any reader
    can derive it from the confirmatory table. Computing it properly, with
    intervals, is the alternative to leaving a reader's arithmetic unqualified.

ansatz               Amendment 14 (E1), specified before its run.
    quantum_basic - quantum_vqc, 24 parameters each.

gap_control          Amendment 13, specified before the v2 noise run.
gap_vqc_minus_control
    H-S3 established that the VQC's macro-F1 falls further than its AUC under
    noise. It did not establish that this is specific to the VQC. G for the
    parameter-matched control, and the paired difference G_vqc - G_control on
    the same seeds and resampled images, answer that.

bottleneck_learned_reference
    H-S6 found no n=5 advantage under frozen PCA or random projections, in an
    experiment with 5 seeds at the untuned default learning rate. A null there
    means something only if the SAME experiment shows the advantage under its
    own learned bottleneck. This is that within-experiment reference.

METHOD
------
Identical to 13_family_table.py, whose engine is imported rather than copied:
nested bootstrap resampling test indices and seeds within each cell, equal
weight per cell, B = 2000, +1-corrected p-values, and the same metric
self-check against scikit-learn before any bootstrap runs.
"""
import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ft = importlib.import_module("13_family_table")


# ------------------------------------------------------------ extra statistic
def stat_gap_difference(cell, idx, ss):
    """
    G_a - G_b per sigma, paired: both arms are scored on the same resampled
    images and the same resampled seeds, and the difference is taken per seed
    before averaging.
    """
    y = cell["y"][idx]

    def per_seed_gaps(tag):
        P0 = cell["arms"][tag + "0.00"][ss][:, idx]
        a0, f0 = ft.auc_batch(y, P0), ft.f1_batch(y, P0)
        out = []
        for s in ft.SIGMAS:
            Ps = cell["arms"][tag + s][ss][:, idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                g = (f0 - ft.f1_batch(y, Ps)) / f0 - (a0 - ft.auc_batch(y, Ps)) / a0
            g[(f0 == 0) | ~np.isfinite(g)] = np.nan
            out.append(g)
        return out

    ga, gb = per_seed_gaps("a"), per_seed_gaps("b")
    return np.array([ft._nanmean(ga[i] - gb[i]) for i in range(len(ft.SIGMAS))])


STATS = {"diff": ft.stat_diff, "gap": ft.stat_calibration_gap,
         "gapdiff": stat_gap_difference}


def _job(job):
    cell, stat_name, B, seed = job
    stat = STATS[stat_name]
    N, S = len(cell["y"]), cell["n_seeds"]
    obs = stat(cell, np.arange(N), np.arange(S))
    reps = np.full((B, obs.size), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(B):
        idx = rng.integers(0, N, N)
        if np.unique(cell["y"][idx]).size < 2:
            continue
        reps[b] = stat(cell, idx, rng.integers(0, S, S))
    return obs, reps


# ------------------------------------------------------------ extra loader
def load_noise_pair(ctx, experiment, arm_a, arm_b):
    """Both arms, clean and four noise levels, on their common seeds."""
    tbl = ctx.s04.collect(experiment)
    cells = []
    for cell in sorted(tbl):
        A, B = tbl[cell].get(arm_a, {}), tbl[cell].get(arm_b, {})
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            continue
        arms, y0 = {}, None
        for tag, recs in (("a", A), ("b", B)):
            for cond in ("0.00",) + ft.SIGMAS:
                P, y = ft._stack(ctx, experiment, recs, seeds, condition=cond)
                if P is None:
                    arms = None
                    break
                if y0 is not None and not np.array_equal(y, y0):
                    raise RuntimeError(f"{experiment}: labels differ between arms")
                arms[tag + cond], y0 = P, y
            if arms is None:
                break
        if arms:
            cd = dict(cell)
            cells.append(ft._cell(cd["dataset"], cd["regime"], seeds, y0, arms))
    return cells


# name -> (namespace, loader, statistic, regime-grouped?, planned seeds, label)
GROUPS = {
    "reupload_vs_control": (
        "01_frozen_tuned",
        lambda c, e: ft.load_pairs(c, e, "quantum_reupload", "matched_param_fullrank"),
        "diff", True, 40, "re-upload VQC - matched control (NOT pre-specified)"),
    "ansatz": (
        "21_ansatz",
        lambda c, e: ft.load_pairs(c, e, "quantum_basic", "quantum_vqc"),
        "diff", True, 10, "BasicEntangler - StronglyEntangling (Amendment 14)"),
    "gap_control": (
        "03_robustness_v2",
        lambda c, e: ft.load_noise(c, e, "matched_param_fullrank"),
        "gap", False, 10, "control's rel F1 loss - rel AUC loss (Amendment 13)"),
    "gap_vqc_minus_control": (
        "03_robustness_v2",
        lambda c, e: load_noise_pair(c, e, "quantum_vqc", "matched_param_fullrank"),
        "gapdiff", False, 10, "G_vqc - G_control, paired (Amendment 13)"),
    "bottleneck_learned_reference": (
        "12_bottleneck",
        lambda c, e: ft.load_pairs(c, e, "quantum_vqc", "matched_param_fullrank",
                                   ft._bn("learned")),
        "diff", True, None, "VQC - control, learned bottleneck, H-S6's own experiment"),
}


def run_group(ctx, name, B, workers, cache_dir, recompute):
    experiment, loader, stat, by_regime, planned, label = GROUPS[name]
    t0 = time.time()
    cells = loader(ctx, experiment)
    why = ft.completeness(cells, by_regime, planned)
    if why:
        print(f"  {name:28s} NOT COMPUTED - {why}")
        return None
    for c in cells:
        for k, P in c["arms"].items():
            ft.selfcheck_metrics(c["y"], P, f"{name}/{c['dataset']}/n={c['regime']}/{k}")

    meta = {"experiment": experiment, "stat": stat, "B": B,
            "cells": [[c["dataset"], c["regime"], c["seeds"]] for c in cells]}
    path = os.path.join(cache_dir, f"{name}.npz")
    if not recompute and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["meta"]) == json.dumps(meta):
            print(f"  {name:28s} cached")
            return cells, list(z["obs"]), list(z["reps"]), by_regime, label

    jobs = [(c, stat, B, ft.cell_seed("x_" + name, c)) for c in cells]
    if workers > 1:
        ctx_mp = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=ctx_mp) as ex:
            res = list(ex.map(_job, jobs))
    else:
        res = [_job(j) for j in jobs]
    obs, reps = [r[0] for r in res], [r[1] for r in res]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, obs=np.stack(obs), reps=np.stack(reps), meta=json.dumps(meta))
    print(f"  {name:28s} computed ({len(cells)} cells, {cells[0]['n_seeds']} seeds) "
          f"in {time.time() - t0:.0f}s", flush=True)
    return cells, obs, reps, by_regime, label


def summarise(name, result, B):
    cells, obs, reps, by_regime, label = result
    pooled = ft.pool(cells, obs, reps, by_regime)
    rows = []
    if by_regime:
        for n in ft.REGIMES:
            o, r = pooled[n]
            s = ft.ci_p(o[0], r[:, 0], B)
            if s:
                rows.append((f"n={n}", s))
    else:
        o, r = pooled["all"]
        for i, sig in enumerate(ft.SIGMAS):
            s = ft.ci_p(o[i], r[:, i], B)
            if s:
                rows.append((f"sigma={sig}", s))

    adj = ft.benjamini_hochberg([s["p"] for _, s in rows], len(rows))
    print(f"\n--- {name}: {label}")
    print(f"    {'':10s} {'estimate':>9s} {'95% CI':>21s} {'p':>7s} {'p_adj':>7s}  "
          f"(BH within this analysis, m={len(rows)})")
    out = []
    for (tag, s), a in zip(rows, adj):
        flag = ("CI excludes 0" if s["ci_lo"] > 0 or s["ci_hi"] < 0 else "")
        print(f"    {tag:10s} {s['estimate']:+9.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] "
              f"{s['p']:7.4f} {a:7.4f}  {flag}")
        out.append({"analysis": name, "row": tag, **s, "p_adj_within": float(a)})
    if by_regime:
        est, R = ft.regime_slope(pooled)
        s = ft.ci_p(est, R, B)
        if s:
            print(f"    {'slope':10s} {s['estimate']:+9.5f} [{s['ci_lo']:+.5f},"
                  f"{s['ci_hi']:+.5f}]  (descriptive)")
            out.append({"analysis": name, "row": "slope", **s})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--B", type=int, default=ft.B_DEFAULT,
                    help=f"testing only; reported analyses use {ft.B_DEFAULT}")
    args = ap.parse_args()

    ctx = ft.Ctx()
    cache_dir = os.path.join(ctx.config.ARTIFACT_ROOT, "exploratory_cache")
    print(f"EXPLORATORY companions | workers={args.workers} | B={args.B} | "
          f"sha={ctx.config.git_sha()[:8]}", flush=True)
    if args.B != ft.B_DEFAULT:
        print(f"!!! B={args.B}: TESTING ONLY - do not quote these numbers !!!")

    results = {}
    for g in args.groups:
        r = run_group(ctx, g, args.B, args.workers, cache_dir, args.recompute)
        if r:
            results[g] = r

    print("\n" + "=" * 78)
    print("EXPLORATORY. None of these is in the confirmatory family; none changes")
    print("a verdict in it. BH is applied within each analysis separately.")
    print("=" * 78)
    rows = []
    for g in args.groups:
        if g in results:
            rows += summarise(g, results[g], args.B)
    out = os.path.join(ctx.config.ARTIFACT_ROOT, "exploratory_table.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
