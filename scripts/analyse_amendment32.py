"""
AMENDMENT 32 - THE SINE-COSINE MODEL ON RAW Z. Exploratory.

    python scripts/analyse_amendment32.py select                 # after 38_lr_fourier_nosquash
    python scripts/analyse_amendment32.py contrasts --workers 7  # after 39_fourier_cpu (B = 2000)
    python scripts/analyse_amendment32.py select --unclipped     # after the unclipped stage
    python scripts/analyse_amendment32.py contrasts --unclipped --workers 7

WHY
---
Amendment 31 found that the circuit gains 0.013-0.020 AUC on raw z (no squash),
and that, each head on raw z at its own tuned rate, it leads the matched head at
n = 5 and 10. The function-class comparison (H-S2: VQC - fourier_rff) was made on
squashed z only. For ANY input the circuit's output is a combination of the same
sines and cosines of its angles, so fourier_rff on raw z is still the circuit's
function class. The question is empirical: does the circuit's function class gain
from raw z too, and does the circuit then beat it?

GROUPS
------
    fc_nosquash           VQC raw z (3e-2, 36_nosquash) - fourier_rff raw z (own rate)
    fc_squashed_cpu       VQC squashed (1e-2, 31) - fourier_rff squashed (1e-2): H-S2 on CPU
    fourier_squash_cost   fourier_rff raw z (own rate) - fourier_rff squashed (1e-2)   descriptive

PREDICTIONS (Amendment 32, written before any run)
    P1  fc_nosquash: no interval lies above zero, at any n.
    P2  fc_squashed_cpu: estimate <= 0 at n = 10, 20, 50 and 100 (H-S2's direction).
    P3  fourier_squash_cost at n = 5: estimate above zero.

METHOD
------
The engine of 14_exploratory_table.py and the loaders of analyse_amendment31.py,
unchanged: nested bootstrap, B = 2000, equal weight per cell, BH within each
analysis (m = 5); runs paired on (dataset, n, seed), test labels checked equal.
--unclipped runs both stages on the views X_<namespace> (build_unclipped_view.py),
with the rates of the procedure as specified, so each contrast differs from its
recorded twin only in clipping; group names and bootstrap draws are unchanged.
"""
import argparse
import importlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
a31 = importlib.import_module("analyse_amendment31")
ex, ft = a31.ex, a31.ft

NS_TUNE, NS_FC = "38_lr_fourier_nosquash", "39_fourier_cpu"
NS_VQC_RAW, NS_VQC_SQ = a31.NS_NOSQ, a31.NS_VQC31
GRID, INHERITED = a31.GRID, a31.INHERITED
ARM = "fourier_rff"
RUNS = len(GRID) * 3 * 4 * 5                                  # 360
RATES_FILE = "fourier_nosquash_rate.json"
VQC_RATES_FILE = a31.RATES_FILE                               # Amendment 31's selection


def select(ctx, unclipped=False):
    ns = a31.view(NS_TUNE, unclipped)
    rows = [r for r in ctx.shards.load_all(ns)
            if r["keys"]["arm"] == ARM and str(r["keys"].get("notanh")) == "1"]
    shas = sorted({r.get("git_sha", "?")[:8] for r in rows})
    print(f"AMENDMENT 32, STAGE 1 {'WITHOUT CLIPPING ' if unclipped else ''}(exploratory) | {ns} | "
          f"{len(rows)} of {RUNS} runs | commits {shas}")
    if len(rows) != RUNS or len(shas) != 1:
        sys.exit("not complete, or more than one commit: no selection is made")
    curve = {g: float(np.mean([r["val_auc"] for r in rows if r["keys"]["lr"] == g])) for g in GRID}
    best = max(GRID, key=lambda g: curve[g])
    edge = best in (GRID[0], GRID[-1])
    print(f"{'fourier_rff (raw z)':24s} " + " ".join(f"{g:>8s}" for g in GRID) + "  selected  gap vs 1e-02")
    print(f"{'':24s} " + " ".join(f"{curve[g]:8.4f}" for g in GRID)
          + f"  {best:>8s}  {curve[best] - curve[INHERITED]:+.4f}" + ("  EDGE" if edge else ""))
    out = {ARM: {"selected": best, "curve": curve, "edge": edge}}
    name = "fourier_nosquash_rate_unclipped.json" if unclipped else RATES_FILE
    if unclipped:
        p = os.path.join(ctx.config.ARTIFACT_ROOT, RATES_FILE)
        if os.path.exists(p):
            with open(p) as f:
                rec = json.load(f)[ARM]["selected"]
            print(f"\nSelection with clipping (the procedure) {rec}, without it {best}"
                  + ("" if rec == best else "   CHANGED"))
    elif edge:
        print("\nSTOP: the selection lies on the edge of the grid; extend the grid first (Amendment 31's rule).")
    path = os.path.join(ctx.config.ARTIFACT_ROOT, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWritten: {path}")


def make_groups(rf, rv, unclipped=False):
    fc, vraw, vsq = (a31.view(n, unclipped) for n in (NS_FC, NS_VQC_RAW, NS_VQC_SQ))
    four_raw = (fc, a31.arm_is(ARM, notanh=1, lrh=rf))
    four_sq = (fc, lambda k: k["arm"] == ARM and "notanh" not in k and k.get("lrh") == INHERITED)
    vqc_raw = (vraw, a31.arm_is("quantum_vqc", notanh=1, lrh=rv, lrq=rv))
    vqc_sq = (vsq, a31.arm_is("quantum_vqc", lrh=INHERITED, lrq=INHERITED))

    def g(a, b, label):
        return (f"{a[0]}x{b[0]}", lambda c, e, a=a, b=b: a31.pair(c, a, b), "diff", True, 40, label)

    return {
        "fc_nosquash": g(vqc_raw, four_raw, f"VQC raw z ({rv}) - fourier_rff raw z ({rf})"),
        "fc_squashed_cpu": g(vqc_sq, four_sq, "VQC squashed (1e-02) - fourier_rff squashed (1e-02), CPU"),
        "fourier_squash_cost": g(four_raw, four_sq, f"fourier_rff raw z ({rf}) - fourier_rff squashed (1e-02)"),
    }


def report_predictions(rows):
    print("\n" + "=" * 78)
    print("PREDICTIONS (Amendment 32)")
    print("=" * 78)
    est = {(r["analysis"], r["row"]): r for r in rows}
    ns = [f"n={n}" for n in ft.REGIMES]
    p1 = [est.get(("fc_nosquash", n)) for n in ns]
    if all(p1):
        above = [n for n, r in zip(ns, p1) if r["ci_lo"] > 0]
        print(f"P1  fc_nosquash: no interval above zero: " + ", ".join(
            f"{n} {r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]" for n, r in zip(ns, p1))
              + f"  ->  {'HOLDS' if not above else 'FAILS at ' + ', '.join(above)}")
    else:
        print("P1  not evaluable")
    p2 = [est.get(("fc_squashed_cpu", n)) for n in ns[1:]]
    if all(p2):
        bad = [n for n, r in zip(ns[1:], p2) if r["estimate"] > 0]
        print("P2  fc_squashed_cpu <= 0 at n >= 10: " + ", ".join(
            f"{n} {r['estimate']:+.4f}" for n, r in zip(ns[1:], p2))
              + f"  ->  {'HOLDS' if not bad else 'FAILS at ' + ', '.join(bad)}")
    else:
        print("P2  not evaluable")
    r = est.get(("fourier_squash_cost", "n=5"))
    print(f"P3  fourier_squash_cost(5) above zero: {r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
          f"  ->  {'HOLDS' if r['estimate'] > 0 else 'FAILS'}" if r else "P3  not evaluable")


def contrasts(ctx, args):
    root = ctx.config.ARTIFACT_ROOT
    paths = [os.path.join(root, RATES_FILE), os.path.join(root, VQC_RATES_FILE)]
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"{p} missing")
    with open(paths[0]) as f:
        rf = json.load(f)[ARM]
    with open(paths[1]) as f:
        rv = json.load(f)["quantum_vqc"]["selected"]
    if rf["edge"]:
        sys.exit("the stage-1 selection is on the grid edge: extend the grid first")
    groups = make_groups(rf["selected"], rv, args.unclipped)
    ex.GROUPS.update(groups)
    suffix = "_unclipped" if args.unclipped else ""
    print(f"AMENDMENT 32{', WITHOUT CLIPPING' if args.unclipped else ''} (exploratory) | "
          f"workers={args.workers} | B={args.B} | sha={ctx.config.git_sha()[:8]} | "
          f"rates: fourier_rff raw z {rf['selected']}, VQC raw z {rv}", flush=True)
    if args.B != ft.B_DEFAULT:
        print(f"!!! B={args.B}: TESTING ONLY - do not quote these numbers !!!")
    cache_dir = os.path.join(root, "amendment32_cache" + suffix)
    results, rows = {}, []
    for g in groups:
        r = ex.run_group(ctx, g, args.B, args.workers, cache_dir, args.recompute)
        if r:
            results[g] = r
    for g in groups:
        if g in results:
            rows += ex.summarise(g, results[g], args.B)
    report_predictions(rows)
    if args.unclipped:
        a31.side_by_side(rows, os.path.join(root, "amendment32.json"))
    out = os.path.join(root, f"amendment32{suffix}.json")
    with open(out, "w") as f:
        json.dump({"rate": rf, "vqc_rate": rv, "rows": rows}, f, indent=1)
    print(f"\nWritten: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "contrasts"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--B", type=int, default=ft.B_DEFAULT,
                    help=f"testing only; reported analyses use {ft.B_DEFAULT}")
    ap.add_argument("--unclipped", action="store_true")
    args = ap.parse_args()
    ctx = ft.Ctx()
    if args.stage == "select":
        select(ctx, args.unclipped)
    else:
        contrasts(ctx, args)


if __name__ == "__main__":
    main()
