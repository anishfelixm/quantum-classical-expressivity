"""
Amendment 20 analysis: the 1,000-epoch re-executions (B_<source>) against the
100-epoch runs of the same seeds.

    python scripts/analyse_budget.py --workers 8          (from the repo root, qml_v2)

Writes logs_stats_budget.txt in the current directory. Reads shards and
predictions only. Every quantity was fixed in Amendment 20 before the runs.

PAIRING. A run in B_01_frozen_tuned is the run of 01_frozen_tuned with the same
keys, re-executed by the code of the commit that recorded it, with MAX_EPOCHS
1,000 and PATIENCE 200 and nothing else changed. Its partner is the recorded
run itself (Amendment 21's re-execution checks that the machine reproduces
it). For full data the partner is R_17_fulldata, the 100-epoch re-execution on
the same GPU, because the recorded full-data runs ran on CPU.

SECTIONS
--------
1. Per arm and n: runs, epochs run and selected under each budget, runs that
   reached 1,000 epochs, and steps above the clip norm in the long runs.
2. Per arm and n: pooled AUC change, 1,000 minus 100 epochs (nested bootstrap,
   equal weight per dataset). P1: |change| < 0.01 everywhere.
3. Per contrast and n: Delta under each budget and the paired change; the
   slope on log2 n under each. P2: wherever the 100-epoch interval (these ten
   seeds) excludes zero, the 1,000-epoch estimate has the same sign. P3: the
   VQC - matched slope is negative under 1,000 epochs.
4. Full data: per arm and dataset, AUC under each budget; pooled VQC - matched
   under each, nested bootstrap per dataset, equal weight (descriptive).
"""
import argparse
import collections
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_conditions as pc          # noqa: E402

ARMS = ["quantum_vqc", "matched_param_fullrank", "fourier_rff", "quantum_reupload", "fourier_rff_r2"]
CONTRASTS = [
    ("primary", "VQC - matched (H-P1/H-P2)", "quantum_vqc", "matched_param_fullrank"),
    ("hs1", "re-upload - VQC (H-S1)", "quantum_reupload", "quantum_vqc"),
    ("hs2", "VQC - Fourier (H-S2)", "quantum_vqc", "fourier_rff"),
    ("e7", "re-upload - Fourier {-2..2}^d (E7)", "quantum_reupload", "fourier_rff_r2"),
    ("rvc", "re-upload - matched", "quantum_reupload", "matched_param_fullrank"),
]
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--B", type=int, default=pc.B_DEFAULT)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    ctx = pc.context()
    ft = pc.ft
    seeds = list(ctx.config.ALL_SEEDS)
    BUD, PRI = "B_01_frozen_tuned", "01_frozen_tuned"
    out(f"Amendment 20 analysis | sha={ctx.config.git_sha()[:8]} | B={args.B} | seeds {seeds}"
        + ("   (NOT B=2000: testing only)" if args.B != pc.B_DEFAULT else ""))

    # ---------------------------------------------------------------- 1
    out("\n=== 1. Runs, epochs, steps above 20 (d = 4) ===")
    out(f"{'arm':24s} {'n':>4s} {'runs':>5s} {'epochs 100/1000':>16s} {'selected 100/1000':>18s} "
        f"{'at 1000':>7s} {'runs w/ step>20':>15s}")
    pri = {tuple(sorted((k, str(v)) for k, v in r["keys"].items())): r for r in ctx.shards.load_all(PRI)}
    by = collections.defaultdict(list)
    for r in ctx.shards.load_all(BUD):
        key = tuple(sorted((k, str(v)) for k, v in r["keys"].items()))
        by[(r["keys"]["arm"], int(r["keys"]["regime"]))].append((r, pri.get(key)))
    for arm in ARMS:
        for n in pc.REGIMES:
            rows = by.get((arm, n), [])
            if not rows:
                out(f"{arm:24s} {n:4d}     0")
                continue
            m1 = [r["metrics"] for r, _ in rows]
            m0 = [q["metrics"] for _, q in rows if q is not None]
            med = statistics.median
            out(f"{arm:24s} {n:4d} {len(rows):5d} "
                f"{med(x['epochs_run'] for x in m0) if m0 else float('nan'):7.1f}/"
                f"{med(x['epochs_run'] for x in m1):<8.1f} "
                f"{med(x['best_epoch'] + 1 for x in m0) if m0 else float('nan'):8.1f}/"
                f"{med(x['best_epoch'] + 1 for x in m1):<9.1f} "
                f"{sum(1 for x in m1 if x['epochs_run'] >= 1000):7d} "
                f"{sum(1 for r, _ in rows if (r.get('recorded', {}).get('above_threshold') or 0) > 0):15d}")

    # ---------------------------------------------------------------- 2
    out("\n=== 2. Pooled AUC change per arm, 1000 minus 100 epochs ===")
    out("P1: |change| < 0.01 for every arm at every n.  * = 95% interval excludes 0")
    p1 = []
    for arm in ARMS:
        cells = pc.load_two(ctx, (BUD, arm), (PRI, arm), seeds=seeds)
        why = pc.complete(cells, len(seeds))
        if why and not args.allow_incomplete:
            out(f"  {arm:24s} incomplete - {why}")
            continue
        rows = pc.pooled_diff(cells, "budget_arm_" + arm, args.B, args.workers)
        p1 += [(arm, r["n"], round(r["stat"]["estimate"], 4)) for r in rows
               if r["stat"] and abs(r["stat"]["estimate"]) >= 0.01]
        out(f"  {arm:24s} " + "  ".join(f"n={r['n']}: {pc.fmt(r['stat'])}" for r in rows)
            + (f"   INCOMPLETE ({why})" if why else ""))
    out(f"P1 {'HOLDS' if not p1 else 'FAILS at ' + str(p1)}")

    # ---------------------------------------------------------------- 3
    out("\n=== 3. Contrasts under each budget (same ten seeds, identical draws) ===")
    p2, p3 = [], None
    for name, label, a, b in CONTRASTS:
        cr = pc.load_two(ctx, (PRI, a), (PRI, b), seeds=seeds)
        cx = pc.load_two(ctx, (BUD, a), (BUD, b), seeds=seeds)
        why = pc.complete(cx, len(seeds))
        if why and not args.allow_incomplete:
            out(f"\n{label}: 1000-epoch data incomplete - {why}")
            continue
        cr, cx = pc.align(cr, cx)
        if pc.complete(cr, None):
            out(f"\n{label}: grid incomplete after alignment - {pc.complete(cr, None)}")
            continue
        rows, slope = pc.summarise_pair(cr, cx, "budget_" + name, args.B, args.workers)
        out(f"\n{label}   ({cr[0]['n_seeds']} seeds per cell){'   INCOMPLETE' if why else ''}")
        out(f"  {'n':>4s}  {'100 epochs':>27s}  {'1000 epochs':>27s}  {'change':>27s}")
        for r in rows:
            out(f"  {r['n']:>4d}  {pc.fmt(r['R'])}  {pc.fmt(r['X'])}  {pc.fmt(r['change'])}")
            if pc.excludes_zero(r["R"]) and (r["R"]["estimate"] > 0) != (r["X"]["estimate"] > 0):
                p2.append((label, r["n"]))
        out(f"  slope {pc.fmt(slope['R'], 5)} {pc.fmt(slope['X'], 5)} {pc.fmt(slope['change'], 5)}")
        if name == "primary" and slope["X"]:
            p3 = slope["X"]["estimate"]
    out("")
    out(f"P2 {'HOLDS' if not p2 else 'FAILS: sign flips at ' + str(p2)}")
    out("P3 not computed" if p3 is None else
        f"P3 {'HOLDS' if p3 < 0 else 'FAILS'}: VQC - matched slope under 1000 epochs {p3:+.5f}")

    # ---------------------------------------------------------------- 4
    out("\n=== 4. Full data (R_17_fulldata, 100 epochs on GPU, vs B_17_fulldata, 1000 epochs) ===")
    r17 = {tuple(sorted((k, str(v)) for k, v in r["keys"].items())): r
           for r in ctx.shards.load_all("R_17_fulldata")}
    b17 = {tuple(sorted((k, str(v)) for k, v in r["keys"].items())): r
           for r in ctx.shards.load_all("B_17_fulldata")}
    if not b17:
        out("  no B_17_fulldata runs")
    else:
        per = collections.defaultdict(list)
        for key, rb in b17.items():
            rr = r17.get(key)
            if rr:
                k = dict(key)
                per[(k["arm"], k["dataset"])].append((rr["metrics"]["auc"], rb["metrics"]["auc"],
                                                      rr["metrics"]["epochs_run"], rb["metrics"]["epochs_run"]))
        out(f"  {'arm':24s} {'dataset':15s} {'seeds':>5s} {'AUC 100':>8s} {'AUC 1000':>9s} {'change':>8s} {'epochs 100/1000':>16s}")
        for (arm, ds), v in sorted(per.items()):
            m = statistics.mean
            out(f"  {arm:24s} {ds:15s} {len(v):5d} {m(x[0] for x in v):8.4f} {m(x[1] for x in v):9.4f} "
                f"{m(x[1] - x[0] for x in v):+8.4f} {m(x[2] for x in v):7.1f}/{m(x[3] for x in v):<8.1f}")
        # pooled VQC - matched under each budget, nested bootstrap per dataset cell
        for ns, label in (("R_17_fulldata", "100 epochs (GPU)"), ("B_17_fulldata", "1000 epochs (GPU)")):
            cells = []
            tbl = ctx.s04.collect(ns)
            for cell, arms in tbl.items():
                A, Bm = arms.get("quantum_vqc", {}), arms.get("matched_param_fullrank", {})
                common = sorted(set(A) & set(Bm))
                if len(common) < 2:
                    continue
                PA, ya = ft._stack(ctx, ns, A, common)
                PB, yb = ft._stack(ctx, ns, Bm, common)
                cells.append(ft._cell(dict(cell)["dataset"], "full", common, ya, {"a": PA, "b": PB}))
            if len(cells) < 4:
                out(f"  {label}: {len(cells)} of 4 datasets available")
                continue
            res = pc._run_jobs([(c, "diff", args.B, ft.cell_seed("fulldata", c)) for c in cells], args.workers)
            obs = np.mean([r[0][0] for r in res])
            reps = np.nanmean(np.stack([r[1][:, 0] for r in res]), axis=0)
            out(f"  pooled VQC - matched, {label}: {pc.fmt(ft.ci_p(obs, reps, args.B))}  "
                f"({cells[0]['n_seeds']} seeds)")

    dst = os.path.join(os.getcwd(), "logs_stats_budget.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst}")


if __name__ == "__main__":
    main()
