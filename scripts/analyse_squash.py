"""
AMENDMENT 28 - DOES THE SQUASH HANDICAP THE MATCHED HEAD? Exploratory.

    python scripts/analyse_squash.py --workers 7
    python scripts/analyse_squash.py --smoke          # after the first seeds: does the pipeline work?
    python scripts/analyse_squash.py --unclipped --workers 7   # Amendment 30

WHY THIS EXISTS
---------------
Every head receives z~ = (pi/2) tanh(z). The circuit needs it: the angle
encoding is 2*pi-periodic, so unbounded inputs would wrap around. The classical
heads do not need it; they receive it so that every arm sees the same input.

The exploratory tanh ablation (20_tanh: d = 4, 10 seeds, classical arms only)
found that removing the squash raised the matched head's AUC at n = 5 on all
four datasets (BloodMNIST +0.032, BreastMNIST +0.018, PneumoniaMNIST +0.014,
PathMNIST +0.012; none significant after correction within that table). Their
simple mean, +0.019, is about the size of the primary advantage (+0.0142). If
the squash costs the matched head that much, the n = 5 advantage could be the
squash's cost to the control rather than anything the circuit does.

Amendment 14 decided not to run a primary comparison without the squash,
because it would have been chosen after seeing that pattern. Amendment 28 runs
it as an exploratory check, its predictions written before any run, and reports
it whatever it shows. The confirmatory verdicts are fixed and are not revisited.

DESIGN (Amendment 28)
---------------------
32_squash: matched_param_fullrank with the squash (--with-tanh, key notanh=0)
and without it (--no-tanh, key notanh=1), 40 confirmatory seeds, 4 datasets,
n = 5..100, d = 4, frozen backbone, learned projection, learning rate 1e-2 (the
tuned rate, selected with the squash; not re-tuned without it). Colab CPU, the
same machine type as 31_entangling_range. The squash adds no parameters, so the
two variants of a seed start from identical weights and see identical batches.

The VQC side is quantum_vqc from 31_entangling_range (Amendment 26): the same
40 seeds, cells, rate, feature caches and machine type. Runs are matched on
(dataset, n, seed); the test labels are checked equal.

CONTRASTS
---------
    squash_cost        matched without squash - matched with squash   (32 only)
    primary_cpu        VQC - matched with squash      (the primary contrast, on CPU)
    primary_nosquash   VQC - matched without squash

PREDICTIONS (Amendment 28, written before any run of 32_squash)
    P1  squash_cost at n = 5 is positive, with its 95% interval above zero.
    P2  primary_nosquash at n = 5 is NOT positive: its interval includes zero
        or lies below it.
    primary_cpu is descriptive: whether the recorded n = 5 advantage
    (+0.0142, GPU) reappears on CPU against the same control.

WITHOUT CLIPPING (Amendment 30, 9 November 2026)
------------------------------------------------
--unclipped runs the same three contrasts on the views X_32_squash and
X_31_entangling_range (scripts/build_unclipped_view.py): every run, with its
unclipped CPU re-execution in place where clipping acted (100 matched-head runs
of 32_squash, 2 quantum_vqc runs of 31_entangling_range). Group names, and so
every bootstrap seed, are unchanged: recorded and unclipped estimates use
identical draws. Results go to squash_check_unclipped.json and
squash_cache_unclipped; the recorded rows are printed beside them when
squash_check.json is present.

METHOD
------
The engine of 14_exploratory_table.py (itself 13_family_table.py's), unchanged:
nested bootstrap over test images and seeds within each cell, equal weight per
cell, B = 2000, +1-corrected p-values, BH within each analysis (m = 5). Results
go to squash_check.json and their own cache folder.
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

NS_SQUASH = "32_squash"
NS_VQC = "31_entangling_range"
MATCHED = "matched_param_fullrank"


def load_across(ctx, exp_a, arm_a, exp_b, arm_b, notanh_b):
    """
    arm_a from exp_a against arm_b (at notanh = notanh_b) from exp_b, one cell per
    (dataset, n), on their common seeds.

    13's load_pairs needs both arms in one namespace. Here they are in two: the
    VQC runs of Amendment 26 and the matched runs of Amendment 28. Cells are
    matched on dataset and n only, because the two namespaces carry different
    optional keys (lrq for circuits, notanh for the squash).
    """
    ta = ctx.s04.collect(exp_a)
    tb = ctx.s04.collect(exp_b, force_keys=("notanh",))
    a_by = {}
    for cell in ta:
        cd = dict(cell)
        recs = ta[cell].get(arm_a)
        if recs:
            key = (cd["dataset"], str(cd["regime"]))
            if key in a_by:
                raise RuntimeError(f"{exp_a}: two cells for {key}")
            a_by[key] = recs
    cells = []
    for cell in sorted(tb):
        cd = dict(cell)
        if str(cd.get("notanh")) != str(notanh_b):
            continue
        B = tb[cell].get(arm_b, {})
        A = a_by.get((cd["dataset"], str(cd["regime"])), {})
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            continue
        PA, ya = ft._stack(ctx, exp_a, A, seeds)
        PB, yb = ft._stack(ctx, exp_b, B, seeds)
        if PA is None or PB is None:
            continue
        if not np.array_equal(ya, yb):
            raise RuntimeError(f"test labels differ between {exp_a} and {exp_b} "
                               f"({cd['dataset']}, n={cd['regime']})")
        cells.append(ft._cell(cd["dataset"], cd["regime"], seeds, ya, {"a": PA, "b": PB}))
    return cells


def load_squash_pairs(ctx, experiment, arm):
    """
    The same arm without (notanh = 1) and with (notanh = 0) the squash, one cell
    per (dataset, n), on common seeds. As 13's load_cross, except that key values
    are compared as text: 01 stores notanh as an integer, and load_cross looks
    values up exactly as given.
    """
    tbl = ctx.s04.collect(experiment, force_keys=("notanh",))
    by_base = {}
    for base, by_val in ctx.s04.group_by(tbl, "notanh").items():
        by_base[base] = {str(v): cell for v, cell in by_val.items()}
    cells = []
    for base, by_val in sorted(by_base.items()):
        if "1" not in by_val or "0" not in by_val:
            continue
        A, B = tbl[by_val["1"]].get(arm, {}), tbl[by_val["0"]].get(arm, {})
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            continue
        PA, ya = ft._stack(ctx, experiment, A, seeds)
        PB, yb = ft._stack(ctx, experiment, B, seeds)
        if PA is None or PB is None:
            continue
        if not np.array_equal(ya, yb):
            raise RuntimeError(f"{experiment}: test labels differ with and without the squash")
        bd = dict(base)
        cells.append(ft._cell(bd["dataset"], bd["regime"], seeds, ya, {"a": PA, "b": PB}))
    return cells


def make_groups(ns_vqc, ns_squash):
    """The three contrasts on the given namespaces (recorded, or their unclipped views)."""
    return {
        "squash_cost": (
            ns_squash, lambda c, e: load_squash_pairs(c, e, MATCHED),
            "diff", True, 40, "matched without squash - matched with squash"),
        "primary_cpu": (
            f"{ns_vqc}x{ns_squash}",
            lambda c, e: load_across(c, ns_vqc, "quantum_vqc", ns_squash, MATCHED, "0"),
            "diff", True, 40, "VQC - matched with squash (the primary contrast, on CPU)"),
        "primary_nosquash": (
            f"{ns_vqc}x{ns_squash}",
            lambda c, e: load_across(c, ns_vqc, "quantum_vqc", ns_squash, MATCHED, "1"),
            "diff", True, 40, "VQC - matched without squash"),
    }


GROUPS = make_groups(NS_VQC, NS_SQUASH)


def side_by_side(rows, recorded_path):
    """Recorded and unclipped estimates of the same rows (Amendment 30, descriptive)."""
    if not os.path.exists(recorded_path):
        print(f"\n(no {os.path.basename(recorded_path)} here: recorded side not shown)")
        return
    with open(recorded_path) as f:
        rec = {(r["analysis"], r["row"]): r for r in json.load(f)["rows"]}
    print("\nRECORDED vs UNCLIPPED (same bootstrap draws)")
    for r in rows:
        a = rec.get((r["analysis"], r["row"]))
        if a is None:
            continue
        print(f"    {r['analysis']:18s} {r['row']:6s} recorded {a['estimate']:+.4f} "
              f"[{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}]   unclipped {r['estimate']:+.4f} "
              f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]   change {r['estimate'] - a['estimate']:+.4f}")


def report_predictions(rows):
    print("\n" + "=" * 78)
    print("PREDICTIONS (Amendment 28)")
    print("=" * 78)
    est = {(r["analysis"], r["row"]): r for r in rows}

    def show(r):
        return f"{r['estimate']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"

    c5 = est.get(("squash_cost", "n=5"))
    if c5:
        print(f"P1  squash_cost(5) interval above zero: {show(c5)}  ->  "
              f"{'HOLDS' if c5['ci_lo'] > 0 else 'FAILS'}")
    else:
        print("P1  not evaluable: squash_cost incomplete")
    d5 = est.get(("primary_nosquash", "n=5"))
    if d5:
        print(f"P2  primary_nosquash(5) not positive (interval includes or lies below zero): "
              f"{show(d5)}  ->  {'HOLDS' if d5['ci_lo'] <= 0 else 'FAILS'}")
    else:
        print("P2  not evaluable: primary_nosquash incomplete")
    p5 = est.get(("primary_cpu", "n=5"))
    if p5:
        print(f"    primary_cpu(5), descriptive: {show(p5)}  (recorded on GPU: +0.0142 [+0.0025,+0.0256])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--B", type=int, default=ft.B_DEFAULT,
                    help=f"testing only; reported analyses use {ft.B_DEFAULT}")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline test on the seeds done so far: any equal seed count, "
                         "B = 50, nothing written. Never quoted.")
    ap.add_argument("--unclipped", action="store_true",
                    help="Amendment 30: the same contrasts on the unclipped views X_32_squash, "
                         "X_31_entangling_range")
    args = ap.parse_args()
    if args.unclipped and args.smoke:
        ap.error("--smoke and --unclipped do not combine")

    groups = make_groups("X_" + NS_VQC, "X_" + NS_SQUASH) if args.unclipped else GROUPS
    if args.smoke:
        args.B = 50
        groups = {g: v[:4] + (None,) + v[5:] for g, v in GROUPS.items()}
    ex.GROUPS.update(groups)           # the same engine, three more contrasts
    ctx = ft.Ctx()
    suffix = "_unclipped" if args.unclipped else ""
    cache_dir = os.path.join(ctx.config.ARTIFACT_ROOT,
                             "squash_cache_smoke" if args.smoke else "squash_cache" + suffix)
    tag = "AMENDMENT 30, WITHOUT CLIPPING" if args.unclipped else "AMENDMENT 28"
    print(f"{tag} (exploratory) | workers={args.workers} | B={args.B} | "
          f"sha={ctx.config.git_sha()[:8]}", flush=True)
    if args.B != ft.B_DEFAULT:
        print(f"!!! B={args.B}: TESTING ONLY - do not quote these numbers !!!")

    results, rows = {}, []
    for g in groups:
        r = ex.run_group(ctx, g, args.B, args.workers, cache_dir, args.recompute)
        if r:
            results[g] = r
    if args.smoke:
        # No estimate is printed: looking at a contrast on a few seeds, while the
        # run is still going, is a peek at the outcome. Only whether each contrast
        # can be computed is shown.
        ok = all(g in results for g in groups)
        print(f"\nSMOKE TEST: {len(results)} of {len(groups)} contrasts computed "
              f"({', '.join(sorted(results)) or 'none'}). No estimates shown. "
              f"{'OK' if ok else 'FAILED'}")
        sys.exit(0 if ok else 1)
    for g in groups:
        if g in results:
            rows += ex.summarise(g, results[g], args.B)
    report_predictions(rows)
    if args.unclipped:
        side_by_side(rows, os.path.join(ctx.config.ARTIFACT_ROOT, "squash_check.json"))

    out = os.path.join(ctx.config.ARTIFACT_ROOT, f"squash_check{suffix}.json")
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
