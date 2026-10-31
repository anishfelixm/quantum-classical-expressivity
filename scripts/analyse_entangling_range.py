"""
AMENDMENT 26 - THE ENTANGLING-RANGE CONTROL. Exploratory.

    python scripts/analyse_entangling_range.py --workers 7

WHAT IT ANSWERS
---------------
quantum_reupload and quantum_vqc differ in two things: the input is encoded
twice, and the second entangling layer uses CNOT range 1 instead of 2 (a
PennyLane default, found by reading the code on 31 October 2026). The five
re-uploading tests of the confirmatory family therefore measure both together.
Experiment 31 trains three arms on the same machine, the same 40 seeds, the
same 20 cells and the same rate, so each contrast below changes one thing:

    range_effect        vqc_range1 - vqc        only the second layer's range
    reupload_effect     reupload - vqc_range1   only the second encoding
    hs1_same_device     reupload - vqc          what H-S1 measured, re-run here

The confirmatory H-S1 verdicts are fixed and are not revisited: they compare
the circuits as built. This analysis says how to read them.

PREDICTIONS (Amendment 26, written before any run of experiment 31)
    P1  |pooled range_effect| < 0.005 AUC at every n.
    P2  reupload_effect > 0 with its 95% interval above zero at n = 50 and
        n = 100, the two cells where H-S1 was supported.
    hs1_same_device is descriptive: it shows whether the recorded H-S1 pattern
    reappears on this machine.

DEVICE CHECK (descriptive). The new quantum_vqc runs are compared with the
recorded quantum_vqc runs of 01_frozen_tuned, seed by seed, to show how far the
change of machine moves the same arm. It is reported, not tested.

METHOD
------
The engine of 14_exploratory_table.py, which imports 13_family_table.py's, is
used unchanged: nested bootstrap over test images and seeds within each cell,
equal weight per cell, B = 2000, +1-corrected p-values, BH within each
analysis. Results go to entangling_range.json and their own cache folder, so
no existing output is touched.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
ex = importlib.import_module("14_exploratory_table")
ft = ex.ft

NS = "31_entangling_range"
GROUPS = {
    "range_effect": (
        NS, lambda c, e: ft.load_pairs(c, e, "quantum_vqc_range1", "quantum_vqc"),
        "diff", True, 40, "VQC ranges (1,1) - VQC ranges (1,2): the entangling range alone"),
    "reupload_effect": (
        NS, lambda c, e: ft.load_pairs(c, e, "quantum_reupload", "quantum_vqc_range1"),
        "diff", True, 40, "re-upload VQC - VQC ranges (1,1): re-uploading alone"),
    "hs1_same_device": (
        NS, lambda c, e: ft.load_pairs(c, e, "quantum_reupload", "quantum_vqc"),
        "diff", True, 40, "re-upload VQC - VQC: the H-S1 contrast, re-run on one machine"),
}


def device_check(ctx):
    """New quantum_vqc minus recorded quantum_vqc, per cell, over common seeds."""
    s04 = ctx.s04
    new, old = s04.collect(NS), s04.collect("01_frozen_tuned")
    rows = []
    for cell in sorted(new):
        a = new[cell].get("quantum_vqc", {})
        b = old.get(cell, {}).get("quantum_vqc", {})
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        m = s04._metric_of
        d = [m(a[s], "auc") - m(b[s], "auc") for s in seeds
             if m(a[s], "auc") is not None and m(b[s], "auc") is not None]
        if d:
            cd = dict(cell)
            rows.append({"dataset": cd["dataset"], "regime": cd["regime"],
                         "seeds": len(d), "mean_diff": float(np.mean(d)),
                         "max_abs_diff": float(np.max(np.abs(d)))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--B", type=int, default=ft.B_DEFAULT,
                    help=f"testing only; reported analyses use {ft.B_DEFAULT}")
    args = ap.parse_args()

    ex.GROUPS.update(GROUPS)           # the same engine, three more contrasts
    ctx = ft.Ctx()
    cache_dir = os.path.join(ctx.config.ARTIFACT_ROOT, "entangling_cache")
    print(f"AMENDMENT 26 (exploratory) | workers={args.workers} | B={args.B} | "
          f"sha={ctx.config.git_sha()[:8]}", flush=True)
    if args.B != ft.B_DEFAULT:
        print(f"!!! B={args.B}: TESTING ONLY - do not quote these numbers !!!")

    results, rows = {}, []
    for g in GROUPS:
        r = ex.run_group(ctx, g, args.B, args.workers, cache_dir, args.recompute)
        if r:
            results[g] = r
    for g in GROUPS:
        if g in results:
            rows += ex.summarise(g, results[g], args.B)

    print("\n" + "=" * 78)
    print("PREDICTIONS (Amendment 26)")
    print("=" * 78)
    est = {(r["analysis"], r["row"]): r for r in rows}
    p1 = [est.get(("range_effect", f"n={n}")) for n in ft.REGIMES]
    if all(p1):
        worst = max(abs(r["estimate"]) for r in p1)
        print(f"P1  |range_effect| < 0.005 at every n: largest {worst:.4f}  ->  "
              f"{'HOLDS' if worst < 0.005 else 'FAILS'}")
    else:
        print("P1  not evaluable: range_effect incomplete")
    p2 = [est.get(("reupload_effect", f"n={n}")) for n in (50, 100)]
    if all(p2):
        ok = all(r["ci_lo"] > 0 for r in p2)
        print("P2  reupload_effect interval above zero at n = 50 and 100: "
              + ", ".join(f"n={n} {r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
                          for n, r in zip((50, 100), p2))
              + f"  ->  {'HOLDS' if ok else 'FAILS'}")
    else:
        print("P2  not evaluable: reupload_effect incomplete")

    dev = device_check(ctx)
    if dev:
        print("\nDEVICE CHECK (descriptive): new quantum_vqc - recorded quantum_vqc")
        for r in dev:
            print(f"    {r['dataset']:15s} n={r['regime']:>4s}  mean {r['mean_diff']:+.4f}  "
                  f"max |diff| {r['max_abs_diff']:.4f}  ({r['seeds']} seeds)")

    out = os.path.join(ctx.config.ARTIFACT_ROOT, "entangling_range.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "device_check": dev}, f, indent=1)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
