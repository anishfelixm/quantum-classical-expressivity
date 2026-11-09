"""
AMENDMENT 31 - EACH ARM AT ITS OWN TUNED RATE, AND THE CIRCUIT WITHOUT THE SQUASH.
Exploratory.

    python scripts/analyse_amendment31.py select                 # after 35_lr_nosquash (seconds)
    python scripts/analyse_amendment31.py contrasts --workers 7  # after 36 and 37 (B = 2000)

WHY
---
Three comparisons left one arm at a disadvantage that the study can remove:

1. Amendment 28 removed the squash from the matched head only, at the rate
   tuned WITH the squash. The circuit was never run without it, and neither
   arm had a rate tuned for raw z. Here both arms are tuned on raw z by the
   unchanged procedure (09_lr_selection.py: same grid, tuning seeds, criterion)
   and the 40-seed comparison is made with each at its own selected rate.
2. The restriction follow-up ran low_rank k = 8 at 1e-2; Amendment 29's tuning
   selected 3e-3 for it (gap 0.0092) and 1e-2 for k = 0.
3. The re-upload function-class control ran fourier_rff_r2 at 1e-2; Amendment
   29 selected 3e-3 for it (gap 0.0053) and 1e-2 for the re-upload circuit.

The original results are fixed and stay reported. These runs are reported
beside them. The rates for 2 and 3 are Amendment 29's selections (made, like
every tuning run of the study, with clipping at 20); they were fixed on
9 November 2026, before any run of this amendment.

DEVICE
------
Everything here ran on Colab CPU, so every contrast pairs CPU runs: the circuit
side of 3, and the squashed VQC of 1's descriptive contrast, are the CPU runs
of 31_entangling_range (Amendment 26), which reproduce exactly in another
session (Amendment 27). 2 and 3 are also run at the INHERITED rate on CPU, so
the effect of the rate is separated from the change of device.

GROUPS
------
    nosquash_fair               VQC raw z (own rate) - matched raw z (own rate)
    vqc_squash_cost             VQC raw z (own rate) - VQC squashed (1e-2)      descriptive
    restriction_cpu_inherited   low_rank k=0 (1e-2) - k=8 (1e-2)                 the follow-up, on CPU
    restriction_tuned           low_rank k=0 (1e-2) - k=8 (3e-3)
    reupload_fc_cpu_inherited   re-upload VQC (1e-2) - fourier_rff_r2 (1e-2)    the control, on CPU
    reupload_fc_tuned           re-upload VQC (1e-2) - fourier_rff_r2 (3e-3)

PREDICTIONS (Amendment 31, written before any run)
    P1  nosquash_fair at n = 5 is NOT positive: its interval includes zero or
        lies below it.
    P2  restriction_tuned at n = 5: its interval includes zero, and its
        estimate is below restriction_cpu_inherited's at n = 5.
    P3  reupload_fc_tuned at n = 5: its estimate is below
        reupload_fc_cpu_inherited's at n = 5.

METHOD
------
The engine of 14_exploratory_table.py, unchanged: nested bootstrap over test
images and seeds within each cell, equal weight per cell, B = 2000,
+1-corrected p-values, BH within each analysis (m = 5). Runs are paired on
(dataset, n, seed); test labels are checked equal.
"""
import argparse
import importlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
ex = importlib.import_module("14_exploratory_table")
ft = ex.ft

NS_TUNE, NS_NOSQ, NS_GAPS = "35_lr_nosquash", "36_nosquash", "37_rate_gaps"
NS_VQC31, NS_SQ32 = "31_entangling_range", "32_squash"
GRID = ["3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01"]
TUNE_ARMS = ("quantum_vqc", "matched_param_fullrank")
RUNS_PER_ARM = len(GRID) * 3 * 4 * 5                       # 360
INHERITED, K8_RATE, FC_RATE = "1e-02", "3e-03", "3e-03"    # Amendment 29's selections
RATES_FILE = "nosquash_rates.json"


# ------------------------------------------------------------------ select
def select(ctx):
    """Amendment 31 stage 1: each arm's rate on raw z, by 09's rule (highest
    mean validation AUC over all tuning runs; test metrics never read)."""
    rows = ctx.shards.load_all(NS_TUNE)
    shas = sorted({r.get("git_sha", "?")[:8] for r in rows})
    print(f"AMENDMENT 31, STAGE 1 (exploratory) | {NS_TUNE} | {len(rows)} runs | commits {shas}")
    by = defaultdict(list)
    for r in rows:
        k = r["keys"]
        if str(k.get("notanh")) == "1" and int(k["dim"]) == 4:
            by[k["arm"]].append(r)
    for arm in TUNE_ARMS:
        print(f"  {arm:24s} {len(by[arm]):4d} of {RUNS_PER_ARM}")
    if any(len(by[a]) != RUNS_PER_ARM for a in TUNE_ARMS) or len(shas) != 1:
        sys.exit("not complete, or more than one commit: no selection is made")

    out = {}
    print(f"\n{'arm (raw z)':24s} " + " ".join(f"{g:>8s}" for g in GRID) + "  selected  gap vs 1e-02")
    for arm in TUNE_ARMS:
        curve = {g: float(np.mean([r["val_auc"] for r in by[arm]
                                   if r["keys"]["lr"] == g and r.get("val_auc") is not None]))
                 for g in GRID}
        best = max(GRID, key=lambda g: curve[g])
        edge = best in (GRID[0], GRID[-1])
        print(f"{arm:24s} " + " ".join(f"{curve[g]:8.4f}" for g in GRID)
              + f"  {best:>8s}  {curve[best] - curve[INHERITED]:+.4f}" + ("  EDGE" if edge else ""))
        out[arm] = {"selected": best, "curve": curve, "edge": edge}
    if any(v["edge"] for v in out.values()):
        print("\nSTOP: a selection lies on the edge of the grid. Amendment 31 extends the grid by one "
              "point in that direction (as Amendment 3a) before any 40-seed run.")
    path = os.path.join(ctx.config.ARTIFACT_ROOT, RATES_FILE)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWritten: {path}")


# ------------------------------------------------------------------ loading
def pick(ctx, ns, want):
    """(dataset, n) -> seed -> record, for the runs of ns whose keys pass `want`."""
    out = defaultdict(dict)
    for r in ctx.shards.load_all(ns):
        k = r["keys"]
        if not want(k):
            continue
        cell = (k["dataset"], str(k["regime"]))
        if k["seed"] in out[cell]:
            raise RuntimeError(f"{ns}: two runs for {cell} seed {k['seed']} under one selection")
        out[cell][k["seed"]] = r
    return out


def pair(ctx, side_a, side_b):
    """side = (namespace, key predicate). One cell per (dataset, n), common seeds."""
    (ns_a, want_a), (ns_b, want_b) = side_a, side_b
    A, B = pick(ctx, ns_a, want_a), pick(ctx, ns_b, want_b)
    cells = []
    for cell in sorted(set(A) & set(B)):
        seeds = sorted(set(A[cell]) & set(B[cell]))
        if len(seeds) < 2:
            continue
        PA, ya = ft._stack(ctx, ns_a, A[cell], seeds)
        PB, yb = ft._stack(ctx, ns_b, B[cell], seeds)
        if PA is None or PB is None:
            continue
        if not np.array_equal(ya, yb):
            raise RuntimeError(f"test labels differ between {ns_a} and {ns_b} at {cell}")
        cells.append(ft._cell(cell[0], cell[1], seeds, ya, {"a": PA, "b": PB}))
    return cells


def arm_is(arm, **kv):
    def want(k):
        return k["arm"] == arm and all(str(k.get(a)) == str(v) for a, v in kv.items())
    return want


def make_groups(rates):
    rv, rm = rates["quantum_vqc"]["selected"], rates["matched_param_fullrank"]["selected"]
    vqc_raw = (NS_NOSQ, arm_is("quantum_vqc", notanh=1, lrh=rv, lrq=rv))
    # The matched head on raw z at 1e-2 already exists (Amendment 28); at any other
    # selected rate it is run in 36_nosquash.
    matched_raw = ((NS_SQ32, arm_is("matched_param_fullrank", notanh=1, lrh=INHERITED))
                   if rm == INHERITED else
                   (NS_NOSQ, arm_is("matched_param_fullrank", notanh=1, lrh=rm)))
    vqc_sq = (NS_VQC31, arm_is("quantum_vqc", lrh=INHERITED, lrq=INHERITED))
    reup = (NS_VQC31, arm_is("quantum_reupload", lrh=INHERITED, lrq=INHERITED))
    k0 = (NS_GAPS, arm_is("low_rank", rank=0, lrh=INHERITED))
    k8_inh = (NS_GAPS, arm_is("low_rank", rank=8, lrh=INHERITED))
    k8_tun = (NS_GAPS, arm_is("low_rank", rank=8, lrh=K8_RATE))
    fc_inh = (NS_GAPS, arm_is("fourier_rff_r2", lrh=INHERITED))
    fc_tun = (NS_GAPS, arm_is("fourier_rff_r2", lrh=FC_RATE))

    def g(a, b, label):
        return (f"{a[0]}x{b[0]}", lambda c, e, a=a, b=b: pair(c, a, b), "diff", True, 40, label)

    return {
        "nosquash_fair": g(vqc_raw, matched_raw,
                           f"VQC raw z ({rv}) - matched raw z ({rm}): each on raw z at its own rate"),
        "vqc_squash_cost": g(vqc_raw, vqc_sq, f"VQC raw z ({rv}) - VQC squashed (1e-02)"),
        "restriction_cpu_inherited": g(k0, k8_inh, "low_rank k=0 (1e-02) - k=8 (1e-02), CPU"),
        "restriction_tuned": g(k0, k8_tun, f"low_rank k=0 (1e-02) - k=8 ({K8_RATE}), each at its tuned rate"),
        "reupload_fc_cpu_inherited": g(reup, fc_inh, "re-upload VQC (1e-02) - fourier_rff_r2 (1e-02), CPU"),
        "reupload_fc_tuned": g(reup, fc_tun, f"re-upload VQC (1e-02) - fourier_rff_r2 ({FC_RATE}), each at its tuned rate"),
    }


def report_predictions(rows):
    print("\n" + "=" * 78)
    print("PREDICTIONS (Amendment 31)")
    print("=" * 78)
    est = {(r["analysis"], r["row"]): r for r in rows}

    def show(r):
        return f"{r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"

    a = est.get(("nosquash_fair", "n=5"))
    print(f"P1  nosquash_fair(5) not positive: {show(a)}  ->  {'HOLDS' if a['ci_lo'] <= 0 else 'FAILS'}"
          if a else "P1  not evaluable")
    t, i = est.get(("restriction_tuned", "n=5")), est.get(("restriction_cpu_inherited", "n=5"))
    if t and i:
        ok = t["ci_lo"] <= 0 <= t["ci_hi"] and t["estimate"] < i["estimate"]
        print(f"P2  restriction_tuned(5) {show(t)} includes zero and is below inherited {show(i)}  ->  "
              f"{'HOLDS' if ok else 'FAILS'}")
    else:
        print("P2  not evaluable")
    t, i = est.get(("reupload_fc_tuned", "n=5")), est.get(("reupload_fc_cpu_inherited", "n=5"))
    if t and i:
        print(f"P3  reupload_fc_tuned(5) {show(t)} below inherited {show(i)}  ->  "
              f"{'HOLDS' if t['estimate'] < i['estimate'] else 'FAILS'}")
    else:
        print("P3  not evaluable")
    print("Recorded on GPU, for reference: restriction follow-up n=5 +0.0097 [+0.0007,+0.0187]; "
          "re-upload function class n=5 +0.0150 [+0.0022,+0.0279].")


def contrasts(ctx, args):
    path = os.path.join(ctx.config.ARTIFACT_ROOT, RATES_FILE)
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run `select` first")
    with open(path) as f:
        rates = json.load(f)
    if any(v["edge"] for v in rates.values()):
        sys.exit("a stage-1 selection is on the grid edge: extend the grid first (Amendment 31)")
    groups = make_groups(rates)
    ex.GROUPS.update(groups)
    print(f"AMENDMENT 31 (exploratory) | workers={args.workers} | B={args.B} | "
          f"sha={ctx.config.git_sha()[:8]} | rates on raw z: VQC {rates['quantum_vqc']['selected']}, "
          f"matched {rates['matched_param_fullrank']['selected']}", flush=True)
    if args.B != ft.B_DEFAULT:
        print(f"!!! B={args.B}: TESTING ONLY - do not quote these numbers !!!")
    cache_dir = os.path.join(ctx.config.ARTIFACT_ROOT, "amendment31_cache")
    results, rows = {}, []
    for g in groups:
        r = ex.run_group(ctx, g, args.B, args.workers, cache_dir, args.recompute)
        if r:
            results[g] = r
    for g in groups:
        if g in results:
            rows += ex.summarise(g, results[g], args.B)
    report_predictions(rows)
    out = os.path.join(ctx.config.ARTIFACT_ROOT, "amendment31.json")
    with open(out, "w") as f:
        json.dump({"rates": rates, "rows": rows}, f, indent=1)
    print(f"\nWritten: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "contrasts"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--B", type=int, default=ft.B_DEFAULT,
                    help=f"testing only; reported analyses use {ft.B_DEFAULT}")
    args = ap.parse_args()
    ctx = ft.Ctx()
    if args.stage == "select":
        select(ctx)
    else:
        contrasts(ctx, args)


if __name__ == "__main__":
    main()
