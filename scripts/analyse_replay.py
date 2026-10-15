"""
Amendment 21 analysis: re-execution of every reported run, and what removing
gradient clipping changes.

    python scripts/analyse_replay.py identity            # after wave 1 (or during: partial)
    python scripts/analyse_replay.py results --workers 8 # after wave 2

Writes logs_stats_replay_<section>.txt in the current directory. Every quantity
was fixed in Amendment 21 before the runs. Writes nothing into any recorded
namespace; the "results" section builds read-only views X_<source> out of hard
links (below) and computes in private cache folders.

identity
--------
1. Re-execution at the original protocol (R_<source>, the recorded commit's
   code, clip 20): per source, runs re-executed, bit-identical, different
   (with the first differing field). 17_fulldata is reported separately: it
   ran on CPU and is re-executed on GPU, so identity is not expected there.
   Prediction P1: every run except 17_fulldata is identical.
2. Step records: per source, arm and configuration, dataset and n - runs with
   any step whose pre-clip norm exceeded 20, steps above 20 of all steps, the
   largest step norm. Prediction P2: none for quantum_vqc or quantum_reupload.
3. The list of runs that need an unclipped counterpart (wave 2), and how many
   of them exist.

results
-------
4. The unclipped condition of a source is the view X_<source>: for every
   recorded run, its U_ re-execution if clipping acted in it, otherwise the
   recorded run itself (a run in which no step exceeded 20 is, by construction,
   identical with and without clipping, and section 1 verifies the machine
   reproduces it). The view is a folder of hard links - no copies, no edits.
5. Every analysis of 13_family_table.py (the 23 confirmatory tests) and of
   14_exploratory_table.py, computed on the recorded namespaces and on the
   views, with the same bootstrap draws (same per-cell generator seeds), so
   each change has a paired interval. The recorded side must reproduce the
   published numbers (anchor). Analyses whose namespaces contain no run in
   which clipping acted are reported as unchanged without recomputation.
6. The confirmatory family's verdicts (BH m = 23) under each condition.
   Prediction P3: no verdict changes.
7. Learning-rate selection (09): mean validation AUC per arm and rate, and the
   selected rate, under each condition.
8. Every other source: number of runs that changed. Where any did, the
   commands that recompute the affected reported numbers on its view.
"""
import argparse
import collections
import importlib
import json
import os
import shutil
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_conditions as pc          # noqa: E402

LINES = []
SOURCES = ["01_frozen_tuned", "03_robustness_v2", "09_lr_selection", "10_capacity",
           "12_bottleneck", "14_readout", "15_dim8", "16_dim16", "17_fulldata", "18_depth",
           "19_angle", "20_tanh", "21_ansatz", "25_encoder", "26_bottleneck_tuned",
           "27_capacity_tuned", "29_compression_frozen"]
CROSS_DEVICE = {"17_fulldata"}
CIRCUITS = {"quantum_vqc", "quantum_reupload"}


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def kname(keys):
    return tuple(sorted((k, str(v)) for k, v in keys.items()))


def load(ctx, ns):
    return {kname(r["keys"]): r for r in ctx.shards.load_all(ns)}


def config_label(k):
    return " ".join(f"{a}={b}" for a, b in k if a not in ("dataset", "regime", "seed", "arm"))


# ------------------------------------------------------------------ identity
def identity(ctx):
    out("=" * 100)
    out("1. RE-EXECUTION AT THE ORIGINAL PROTOCOL (R_<source>): IDENTITY WITH THE RECORDED RUN")
    out("=" * 100)
    p1_bad, need_u, n_missing = 0, {}, [0]
    for src in SOURCES:
        rec, rep = load(ctx, src), load(ctx, "R_" + src)
        verdicts = collections.Counter()
        diffs = []
        for key, r in rep.items():
            v = r.get("replay_check", {}).get("verdict", "?")
            verdicts[v.split(" ")[0]] += 1
            if v.startswith("DIFFERENT") and len(diffs) < 5:
                diffs.append((dict(key).get("arm"), dict(key).get("dataset"), dict(key).get("regime"),
                              dict(key).get("seed"), r["replay_check"].get("first_difference"),
                              r["replay_check"].get("predictions")))
        tag = "  (cross-device: identity not expected)" if src in CROSS_DEVICE else ""
        inc = "" if len(rep) == len(rec) else f"  INCOMPLETE ({len(rec) - len(rep)} not yet re-executed)"
        out(f"{src:24s} recorded {len(rec):5d}  re-executed {len(rep):5d}  identical "
            f"{verdicts['IDENTICAL']:5d}  different {verdicts['DIFFERENT']:5d}{tag}{inc}")
        n_missing[0] += len(rec) - len(rep)
        for d in diffs:
            out(f"      e.g. {d}")
        if src not in CROSS_DEVICE:
            p1_bad += verdicts["DIFFERENT"]
    out("")
    out(("P1 HOLDS so far: every re-executed run is bit-identical to its record."
         if p1_bad == 0 else f"P1 FAILS: {p1_bad} re-executed runs differ from their records.")
        + (f" {n_missing[0]} recorded runs are not yet re-executed." if n_missing[0] else ""))

    out("\n" + "=" * 100)
    out("2. STEP RECORDS (pre-clip norm of every optimisation step, original trajectories)")
    out("=" * 100)
    p2_bad = 0
    for src in SOURCES:
        rep = load(ctx, "R_" + src)
        if not rep:
            continue
        tab = collections.defaultdict(list)
        for key, r in rep.items():
            k = dict(key)
            tab[(k["arm"], config_label(key))].append((k["dataset"], k["regime"], r.get("recorded", {}), key))
        out(f"\n{src}")
        out(f"  {'arm':24s} {'config':44s} {'runs':>5s} {'any>20':>7s} {'steps>20':>18s} "
            f"{'max step':>9s} {'nonfinite':>9s}")
        for (arm, cfg), rows in sorted(tab.items()):
            rr = [x[2] for x in rows if x[2]]
            anyx = [x for x in rows if (x[2].get("above_threshold") or 0) > 0 or (x[2].get("nonfinite") or 0) > 0]
            above = sum(x.get("above_threshold") or 0 for x in rr)
            steps = sum(x.get("steps") or 0 for x in rr)
            mx = max((x.get("max_step_norm") or 0.0) for x in rr) if rr else float("nan")
            nf = sum(x.get("nonfinite") or 0 for x in rr)
            out(f"  {arm:24s} {cfg[:44]:44s} {len(rows):5d} {len(anyx):7d} {above:8d}/{steps:<9d} "
                f"{mx:9.3f} {nf:9d}")
            if arm in CIRCUITS and anyx:
                p2_bad += len(anyx)
            for x in anyx:
                need_u.setdefault(src, []).append(x[3])
    out("")
    out("P2 HOLDS: no step of any circuit run exceeded 20." if p2_bad == 0
        else f"P2 FAILS: {p2_bad} circuit runs had a step above 20.")

    out("\n" + "=" * 100)
    out("3. RUNS IN WHICH CLIPPING ACTED, AND THEIR UNCLIPPED RE-EXECUTIONS (U_<source>)")
    out("=" * 100)
    for src in SOURCES:
        n = need_u.get(src, [])
        have = load(ctx, "U_" + src)
        done = sum(1 for k in n if k in have)
        by = collections.Counter(dict(k)["arm"] for k in n)
        out(f"{src:24s} runs with a step above 20: {len(n):5d}  re-executed unclipped: {done:5d}  "
            f"{dict(by) if by else ''}")
    return need_u


# ------------------------------------------------------------------ views
def build_view(ctx, src):
    """X_<src>: hard links to U_ runs where they exist, else to the recorded run."""
    cfg = ctx.config
    view = "X_" + src
    sdir, pdir = os.path.join(cfg.SHARD_DIR, view), os.path.join(cfg.PREDICTION_DIR, view)
    for d in (sdir, pdir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    rec, unc = load(ctx, src), load(ctx, "U_" + src)
    n_u = 0
    for key, r in rec.items():
        keys = r["keys"]
        use = "U_" + src if key in unc else src
        n_u += key in unc
        os.link(ctx.shards.shard_path(use, **keys), os.path.join(sdir, ctx.shards.shard_name(view, **keys)))
        pp = ctx.shards.pred_path(use, **keys)
        if os.path.exists(pp):
            os.link(pp, os.path.join(pdir, os.path.basename(pp)))
    return view, n_u, len(rec)


def results(ctx, B, workers):
    ft = pc.ft
    et = importlib.import_module("14_exploratory_table")
    cfg = ctx.config
    work = os.path.join(cfg.ARTIFACT_ROOT, "replay_analysis")
    os.makedirs(work, exist_ok=True)
    out("=" * 100)
    out("4. VIEWS OF THE UNCLIPPED CONDITION")
    out("=" * 100)
    changed, ready = {}, {}
    for src in SOURCES:
        if src in CROSS_DEVICE:
            continue
        rec, rep, unc = load(ctx, src), load(ctx, "R_" + src), load(ctx, "U_" + src)
        flagged = [k for k, r in rep.items()
                   if (r.get("recorded", {}).get("above_threshold") or 0) > 0
                   or (r.get("recorded", {}).get("nonfinite") or 0) > 0]
        missing_r = len([k for k in rec if k not in rep])
        missing_u = len([k for k in flagged if k not in unc])
        if not rec:
            ready[src] = "NOT ESTABLISHED: the source has no recorded runs here"
        elif missing_r or missing_u:
            ready[src] = (f"NOT ESTABLISHED: {missing_r} of {len(rec)} runs not re-executed, "
                          f"{missing_u} of {len(flagged)} flagged runs without an unclipped run")
        else:
            ready[src] = "ok"
        view, n_u, n = build_view(ctx, src)
        changed[src] = n_u
        out(f"{view:28s} {n:5d} runs, of which {n_u:5d} are unclipped re-executions"
            + ("" if ready[src] == "ok" else f"   {ready[src]}"))

    out("\n" + "=" * 100)
    out("5-6. CONFIRMATORY FAMILY (13_family_table) - RECORDED vs UNCLIPPED, identical draws")
    out("=" * 100)
    # private copies of the published caches, so nothing published is overwritten
    cache_r = os.path.join(work, "family_recorded")
    cache_x = os.path.join(work, "family_unclipped")
    for d in (cache_r, cache_x):
        os.makedirs(d, exist_ok=True)
    pub = os.path.join(cfg.ARTIFACT_ROOT, "family_table_cache")
    if os.path.isdir(pub):
        for fn in os.listdir(pub):
            shutil.copy2(os.path.join(pub, fn), os.path.join(cache_r, fn))
    groups_r, groups_x = {}, {}
    base = {g: v for g, v in ft.GROUPS.items()}
    for g, spec in base.items():
        ns = spec[0]
        ft.GROUPS[g] = spec
        groups_r[g] = ft.run_group(ctx, g, B, workers, cache_r, False)
        if ready.get(ns) != "ok":
            groups_x[g] = {"status": ready.get(ns, "source not replayed")}
            out(f"  {g:11s} {ns:20s} {groups_x[g]['status']}")
            continue
        if changed.get(ns, 0) == 0:
            groups_x[g] = groups_r[g]
            out(f"  {g:11s} {ns:20s} unchanged: no run had a step above 20")
            continue
        ft.GROUPS[g] = ("X_" + ns,) + spec[1:]
        groups_x[g] = ft.run_group(ctx, g, B, workers, cache_x, True)
        ft.GROUPS[g] = spec
        _report_group_change(ft, g, groups_r[g], groups_x[g], B)
    rows_r, rows_x = ft.assemble(groups_r, B), ft.assemble(groups_x, B)
    fam_path = os.path.join(cfg.ARTIFACT_ROOT, "family_table.json")
    if os.path.exists(fam_path):
        with open(fam_path, encoding="utf-8") as f:
            pub_rows = {r["test"]: r for r in json.load(f)}
        bad = [r["test"] for r in rows_r if r.get("status") == "ok" and r["test"] in pub_rows
               and (abs(r["estimate"] - pub_rows[r["test"]].get("estimate", np.nan)) > 1e-12
                    or abs(r["p_adj"] - pub_rows[r["test"]].get("p_adj", np.nan)) > 1e-12)]
        out(f"\nAnchor: recorded-side family {'REPRODUCES' if not bad else 'DOES NOT REPRODUCE'} "
            f"family_table.json{'' if not bad else ': ' + ', '.join(bad)}")
    n_changed = 0
    out(f"\n  {'test':12s} {'estimate rec':>12s} {'p_adj rec':>9s} {'verdict recorded':>27s} | "
        f"{'estimate unc':>12s} {'p_adj unc':>9s} {'verdict unclipped':>27s}")
    for a, b in zip(rows_r, rows_x):
        ch = a.get("verdict") != b.get("verdict")
        n_changed += ch
        out(f"  {a['test']:12s} {a.get('estimate', np.nan):+12.4f} {a.get('p_adj', np.nan):9.4f} "
            f"{a.get('verdict', '?'):>27s} | {b.get('estimate', np.nan):+12.4f} "
            f"{b.get('p_adj', np.nan):9.4f} {b.get('verdict', '?'):>27s}{'   <- CHANGED' if ch else ''}")
    not_ready = sorted({spec[0] for spec in base.values() if ready.get(spec[0]) != "ok"})
    if not_ready:
        out(f"\nP3 NOT YET DECIDABLE: namespaces not established: {', '.join(not_ready)}")
    else:
        out(f"\nP3 {'HOLDS' if n_changed == 0 else 'FAILS'}: {n_changed} of {len(rows_r)} verdicts change.")

    out("\n" + "=" * 100)
    out("5. EXPLORATORY AND FOLLOW-UP ANALYSES (14_exploratory_table) - RECORDED vs UNCLIPPED")
    out("=" * 100)
    cache_er = os.path.join(work, "exploratory_recorded")
    cache_ex = os.path.join(work, "exploratory_unclipped")
    for d in (cache_er, cache_ex):
        os.makedirs(d, exist_ok=True)
    pub = os.path.join(cfg.ARTIFACT_ROOT, "exploratory_cache")
    if os.path.isdir(pub):
        for fn in os.listdir(pub):
            shutil.copy2(os.path.join(pub, fn), os.path.join(cache_er, fn))
    for g, spec in list(et.GROUPS.items()):
        ns = spec[0]
        if ready.get(ns) != "ok":
            out(f"  {g:28s} {ns:20s} {ready.get(ns, 'source not replayed')}")
            continue
        if changed.get(ns, 0) == 0:
            out(f"  {g:28s} {ns:20s} unchanged: no run had a step above 20")
            continue
        r = et.run_group(ctx, g, B, workers, cache_er, False)
        et.GROUPS[g] = ("X_" + ns,) + spec[1:]
        x = et.run_group(ctx, g, B, workers, cache_ex, True)
        et.GROUPS[g] = spec
        if r is None or x is None:
            out(f"  {g}: not computable ({'recorded' if r is None else 'view'} incomplete)")
            continue
        _report_group_change(ft, g, (r[0], r[1], r[2], r[3]), (x[0], x[1], x[2], x[3]), B, tuple_form=True)

    out("\n" + "=" * 100)
    out("7. LEARNING-RATE SELECTION (09) - RECORDED vs UNCLIPPED")
    out("=" * 100)
    for label, ns in (("recorded", "09_lr_selection"), ("unclipped", "X_09_lr_selection")):
        acc = collections.defaultdict(list)
        for r in ctx.shards.load_all(ns):
            if r.get("val_auc") is not None:
                acc[(r["keys"]["arm"], r["keys"]["lr"])].append(r["val_auc"])
        arms = sorted({a for a, _ in acc})
        out(f"  {label}:")
        for a in arms:
            lrs = sorted({lr for b, lr in acc if b == a}, key=float)
            means = {lr: statistics.mean(acc[(a, lr)]) for lr in lrs}
            best = max(means, key=means.get)
            out(f"    {a:24s} " + "  ".join(f"{lr}:{means[lr]:.4f}" for lr in lrs) + f"   selected {best}")

    out("\n" + "=" * 100)
    out("8. OTHER SOURCES")
    out("=" * 100)
    for src in SOURCES:
        if src in CROSS_DEVICE:
            continue
        n = changed.get(src, 0)
        if ready.get(src) != "ok":
            out(f"  {src:24s} {ready.get(src)}")
        elif n == 0:
            out(f"  {src:24s} no run had a step above 20: every number from it is unchanged")
        else:
            out(f"  {src:24s} {n} runs changed. Recompute its reported numbers on X_{src}, e.g.\n"
                f"      python src/04_statistical_analysis.py --experiment X_{src}")


def _report_group_change(ft, g, R, X, B, tuple_form=False):
    if tuple_form:
        cells_r, obs_r, reps_r, by_r = R
        cells_x, obs_x, reps_x, by_x = X
    else:
        if R.get("status") != "ok" or X.get("status") != "ok":
            out(f"  {g}: not computable (recorded {R.get('status')}, view {X.get('status')})")
            return
        cells_r, obs_r, reps_r, by_r = R["cells"], R["obs"], R["reps"], R["by_regime"]
        cells_x, obs_x, reps_x, by_x = X["cells"], X["obs"], X["reps"], X["by_regime"]
    pr, px = ft.pool(cells_r, obs_r, reps_r, by_r), ft.pool(cells_x, obs_x, reps_x, by_x)
    out(f"\n  {g}")
    for key in pr:
        (o_r, r_r), (o_x, r_x) = pr[key], px[key]
        for i in range(len(o_r)):
            a = ft.ci_p(o_r[i], r_r[:, i], B)
            b = ft.ci_p(o_x[i], r_x[:, i], B)
            c = ft.ci_p(o_x[i] - o_r[i], r_x[:, i] - r_r[:, i], B)
            lab = f"n={key}" if by_r else f"index {i}"
            out(f"    {lab:9s} recorded {pc.fmt(a)}  unclipped {pc.fmt(b)}  change {pc.fmt(c)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section", choices=["identity", "results"])
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--B", type=int, default=pc.B_DEFAULT)
    args = p.parse_args()
    ctx = pc.context()
    out(f"Amendment 21 analysis ({args.section}) | sha={ctx.config.git_sha()[:8]} | B={args.B}"
        + ("   (NOT B=2000: testing only)" if args.B != pc.B_DEFAULT else ""))
    if args.section == "identity":
        identity(ctx)
    else:
        results(ctx, args.B, args.workers)
    dst = os.path.join(os.getcwd(), f"logs_stats_replay_{args.section}.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst}")


if __name__ == "__main__":
    main()
