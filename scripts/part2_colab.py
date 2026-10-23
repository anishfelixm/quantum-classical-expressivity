"""
Part 2 of the Colab analyses (22 October 2026). Three checks the first pass left
open, all computed from the backed-up records; nothing is trained.

 A. Amendment 21, item 6 ("Everything else"): every other reported table is
    recomputed by the script that produced it, 04_statistical_analysis.py with
    the same arguments, on the recorded namespace and on its unclipped view
    X_<ns> (the U_ run where clipping acted, the recorded run otherwise; for
    17_fulldata, whose GPU re-executions cannot match the CPU records, R_17 and
    X_17 are compared instead). Plus, per namespace, arm, dataset and n, the
    mean test AUC under each condition and the largest change.
 B. Amendment 20, P0 for the 38 full-data budget runs that were executed before
    their GPU partner existed: each B_17 run must repeat its R_17 run bit for
    bit over the epochs R_17 ran.
 C. Amendment 23, prediction (ii): per binary run, the largest change in AUC
    under end-of-circuit depolarising noise, so the FAILS line can be read.

    python /content/part2.py          (from /content/repo, QML_ARTIFACT_ROOT set)
Writes to $PART2_OUT (a Drive folder). Resumable: re-running skips every 04
output already finished, so a Colab disconnect costs only the jobs in flight.
The 04 runs are independent and run in parallel, one thread each.
"""
import collections
import concurrent.futures as cf
import json
import math
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("PART2_REPO", "/content/repo")
sys.path.insert(0, os.path.join(REPO, "src"))
import shards   # noqa: E402
import config   # noqa: E402

# Written straight to Drive, so a Colab disconnect loses nothing already finished.
OUT = os.environ.get("PART2_OUT", os.path.join(REPO, "logs_part2"))
# 04 itself, with its two bootstrap loops computed exactly but ~20x faster (see fast04.py).
FAST04 = os.environ.get("FAST04", "/content/fast04.py")
# The logs the manuscript reports, made by the original 04 on the recorded runs.
# The clipped side recomputed here must reproduce them row for row.
COMMITTED = {"d8": "logs_stats_d8.txt", "d16": "logs_stats_d16.txt", "noise": "logs_stats_noise.txt",
             "bottleneck": "logs_stats_12b.txt", "hs7a": "logs_stats_hs7a.txt",
             "hs7b": "logs_stats_hs7b.txt", "angle": "logs_stats_angle_v2.txt",
             "tanh": "logs_stats_tanh_v2.txt", "depth_4v2": "logs_stats_depth.txt",
             "depth_1v2": "logs_stats_depth.txt", "rank": "logs_stats_10_rank.txt",
             "compression": "logs_stats_compression_frozen.txt"}
# Rough cost order (PathMNIST rows dominate), largest first, so the longest job never starts last.
SIZE = {"capacity_tuned_rank": 5, "primary": 9, "tanh": 8, "bottleneck": 7, "noise": 6, "capacity_tuned": 6, "encoder": 5,
        "angle": 5, "depth_4v2": 4, "depth_1v2": 4, "ansatz": 3, "compression": 3}
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def kname(k):
    return tuple(sorted((a, str(b)) for a, b in k.items()))


def load(ns):
    return {kname(r["keys"]): r for r in shards.load_all(ns)}


def build_view(ns, base):
    """X_<ns>: U_ run where one exists, else the run in `base`. Hard links only."""
    view = "X_" + ns
    sdir, pdir = os.path.join(config.SHARD_DIR, view), os.path.join(config.PREDICTION_DIR, view)
    for d in (sdir, pdir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    rec, unc = load(ns), load("U_" + ns)
    n_u = 0
    for key, r in rec.items():
        keys = r["keys"]
        use = "U_" + ns if key in unc else base
        n_u += key in unc
        os.link(shards.shard_path(use, **keys), os.path.join(sdir, shards.shard_name(view, **keys)))
        pp = shards.pred_path(use, **keys)
        if os.path.exists(pp):
            os.link(pp, os.path.join(pdir, os.path.basename(pp)))
    return view, n_u, len(rec)


def auc_of(r):
    for path in (("metrics", "auc"), ("test_metrics", "auc"), ("curve", "0.00", "auc")):
        x = r
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            return float(x)
    return None


# ------------------------------------------------------------------ A
F23 = ["--family-size", "23"]
RUNS = [  # label, namespace, arguments of 04 as in the reported logs, cross-condition (key, A, B)
    ("primary", "01_frozen_tuned", F23, None),
    ("d8", "15_dim8", F23, None),
    ("d16", "16_dim16", F23, None),
    ("fulldata", "17_fulldata", F23, None),
    ("noise", "03_robustness_v2", F23, None),
    ("bottleneck", "12_bottleneck", F23, None),
    ("hs7a", "14_readout", F23 + ["--arms", "quantum_rich", "quantum_rich_padded"], None),
    ("hs7b", "14_readout", F23 + ["--arms", "quantum_rich", "quantum_vqc"], None),
    ("ansatz", "21_ansatz", F23, None),
    ("encoder", "25_encoder", F23, None),
    ("capacity_tuned", "27_capacity_tuned", F23, None),
    ("angle", "19_angle", F23, ("as", 3.1416, 1.5708)),
    ("tanh", "20_tanh", F23, ("notanh", 1, 0)),
    ("depth_4v2", "18_depth", F23, ("L", 4, 2)),
    ("depth_1v2", "18_depth", F23, ("L", 1, 2)),
    ("rank", "10_capacity", F23, ("rank", 0, 8)),
    ("compression", "29_compression_frozen", [], ("dim", 256, 4)),
    # Added 23 Oct: with no arguments, 04 compares the primary pair, which these two
    # sweeps do not contain ("Nothing comparable found"). Their own contrasts:
    ("ansatz_arms", "21_ansatz", F23 + ["--arms", "quantum_vqc", "quantum_basic"], None),
    ("capacity_tuned_rank", "27_capacity_tuned", F23, ("rank", 0, 8)),
]


def stored_value(ns, key, target):
    """The value of `key` exactly as stored in the shards (04 matches strings)."""
    vals = {str(r["keys"][key]) for r in shards.load_all(ns) if key in r["keys"]}
    return min(vals, key=lambda v: abs(float(v) - float(target)))


def job04(ns, args, cross, tag):
    if cross:
        k, a, b = cross
        args = args + ["--compare-key", k, "--compare-values",
                       stored_value(ns, k, a), stored_value(ns, k, b)]
    return (ns, args, os.path.join(OUT, f"04f_{tag}.txt"))


def done(path):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        txt = f.read().rstrip()
    # finished AND successful: an interrupted run also wrote "#DONE" before 23 Oct
    return txt.endswith("#DONE") and "\nexit code 0\n" in txt + "\n"


def run_job(job):
    """One 04 run; written to a temporary file and renamed when complete, so a
    file on Drive is either finished or absent."""
    ns, args, path = job
    if done(path):
        return path, "already done"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    tmp = path + ".part"
    with open(tmp, "w") as f:
        f.write("args: --experiment " + ns + " " + " ".join(args) + "\n")
        f.flush()
        rc = subprocess.run([sys.executable, FAST04, "--experiment", ns] + args,
                            cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        f.write(f"\nexit code {rc}\n#DONE\n")
    os.replace(tmp, path)
    return path, f"finished (exit {rc})"


def part_a():
    out("=" * 100)
    out("A. EVERY OTHER REPORTED TABLE: recorded (or R_ for full data) vs unclipped view")
    out("=" * 100)
    views, plan = {}, []
    only = [x for x in os.environ.get("PART2_ONLY", "").split(",") if x]
    for label, ns, args, cross in RUNS:
        if only and label not in only:
            continue
        if ns not in views:
            rep = load("R_" + ns)
            ident = bool(rep) and all(str(r.get("replay_check", {}).get("verdict", "")).startswith("IDENTICAL")
                                      for r in rep.values())
            base = ns if ident else "R_" + ns
            views[ns] = (base,) + build_view(ns, base)
        base, view, n_u, n = views[ns]
        jobs = None
        if n_u and cross and not any(cross[0] in r["keys"] for r in shards.load_all(ns)):
            out(f"\n{label}: {ns} has no key '{cross[0]}': skipped")
            continue
        if n_u:
            # Keep a pair the original 04 already finished (the overnight run);
            # otherwise compute BOTH sides with fast04, so a pair is never mixed.
            orig = [os.path.join(OUT, f"04_{label}_{side}.txt") for side in ("clipped", "unclipped")]
            jobs = (job04(base, args, cross, f"{label}_clipped"),
                    job04(view, args, cross, f"{label}_unclipped"))
            if all(done(o) for o in orig):
                jobs = tuple((j[0], j[1], o) for j, o in zip(jobs, orig))
        plan.append((label, ns, base, view, n_u, n, jobs))

    todo = [(SIZE.get(label, 1), j) for label, *_, jobs in plan if jobs for j in jobs]
    todo = [j for _, j in sorted(todo, key=lambda t: -t[0])]
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"running {len(todo)} analyses, {workers} at a time "
          f"({sum(done(j[2]) for j in todo)} already finished)", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_job, j) for j in todo]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            path, status = fu.result()
            print(f"  [{k}/{len(todo)}] {os.path.basename(path)}: {status}", flush=True)

    for label, ns, base, view, n_u, n, jobs in plan:
        out(f"\n{label}: {ns}  clipped side {base}; view {view}: {n} runs, {n_u} unclipped")
        if not jobs:
            out("  no run had a step above 20: unchanged by construction")
            continue
        a, b = jobs[0][2], jobs[1][2]
        out(f"  -> {os.path.basename(a)} and {os.path.basename(b)}")
        out(f"  computed by {'the original 04' if '/04_' in a else 'fast04'}")
        for p in (a, b):
            txt = open(p).read()
            if "SEED-LEVEL FALLBACK IN USE" in txt:
                out(f"  !!! SEED-LEVEL FALLBACK IN USE in {os.path.basename(p)}: STOP and report")
            if "Nothing comparable found" in txt:
                out(f"  note: {os.path.basename(p)} has no table (04 found nothing to compare with these arguments)")
            if "Traceback" in txt:
                out(f"  !!! {os.path.basename(p)} ended with an error")
        # mean AUC per arm, configuration, dataset and n under each condition
        A, X = load(base), load(view)
        cells = collections.defaultdict(lambda: [[], []])
        for key, r in A.items():
            k = dict(key)
            cfg = " ".join(f"{a_}={b_}" for a_, b_ in key if a_ not in ("dataset", "regime", "seed", "arm",
                                                                      "fp", "aug", "lrh", "lrq"))
            c = (k.get("arm", "-"), cfg, k["dataset"], k["regime"])
            x = X.get(key)
            va, vx = auc_of(r), auc_of(x) if x else None
            if va is not None and vx is not None:
                cells[c][0].append(va)
                cells[c][1].append(vx)
        worst = sorted(((abs(sum(vx) / len(vx) - sum(va) / len(va)),
                         sum(vx) / len(vx) - sum(va) / len(va), c, len(va))
                        for c, (va, vx) in cells.items()), reverse=True)
        out(f"  mean-AUC cells: {len(worst)}; largest |change| in a cell mean:")
        for _, d, c, m in worst[:5]:
            if abs(d) > 0:
                out(f"    {d:+.4f}  arm={c[0]} {c[1]} dataset={c[2]} n={c[3]} ({m} seeds)")
        if worst and worst[0][0] == 0:
            out("    none: every cell mean is identical")
        out("  verdict changes (clipped -> unclipped):")
        for line in verdict_changes(a, b):
            out("    " + line)
        for ref, what in ((os.path.join(REPO, COMMITTED.get(label, "-")), "reported log"),
                          (a.replace("04f_", "04_"), "original 04, clipped"),
                          (b.replace("04f_", "04_"), "original 04, unclipped")):
            if os.path.exists(ref) and ref not in (a, b) and (what != "reported log" or base == ns):
                mine = b if "unclipped" in what else a
                same, diff, lines = compare_rows(ref, mine)
                out(f"  check against {what} ({os.path.basename(ref)}): {same} rows identical, "
                    f"{diff} different" + ("" if diff else "  OK"))
                for ln in lines[:6]:
                    out("    " + ln)


def table_rows(p):
    """
    Every result row of a 04 output, keyed by (table header, condition). The
    condition is the words before the first number; learning-rate labels
    (lrh=, lrq=) are dropped, because older versions of 04 left them out of
    the cell label while the numbers are the same cell.
    """
    import re
    sec, res = None, {}
    for ln in open(p):
        if ln.startswith("==="):
            sec = ln.strip()
        elif sec and ("[" in ln and "," in ln) and ln[:1] not in "!#":
            t = [w for w in ln.split() if not re.match(r"lr[hq]=", w)]
            i = next((j for j, w in enumerate(t) if re.match(r"^[+-]\d", w)), len(t))
            res[(sec, " ".join(t[:i]))] = " ".join(t[i:])
    return res


def compare_rows(ref, mine):
    """Rows present in both files: identical text, or the two versions."""
    R, M = table_rows(ref), table_rows(mine)
    same = diff = 0
    lines = []
    for k in R:
        if k in M:
            if R[k] == M[k]:
                same += 1
            else:
                diff += 1
                lines.append(f"{k[0][:50]} | {k[1]}\n      ref:  {R[k]}\n      here: {M[k]}")
    return same, diff, lines


def verdict_changes(pa, pb):
    """Rows of the two 04 outputs whose verdict differs; compared line by line
    inside each table (same order on both sides: same cells, same arguments)."""
    def rows(p):
        sec, res = None, {}
        for ln in open(p):
            if ln.startswith("==="):
                sec = ln.strip()
            elif sec and " n=" in ln and "[" in ln:
                cond = ln[:41].strip()
                res[(sec, cond)] = ln.rstrip()
        return res
    A, B = rows(pa), rows(pb)
    ch = []
    for k in A:
        va, vb = A[k], B.get(k)
        if vb is None:
            ch.append(f"{k[1]}: missing on unclipped side")
        elif ("difference" in va) != ("difference" in vb) or va.split()[-2:] != vb.split()[-2:]:
            ch.append(f"{k[0][4:40]} {k[1]}\n      clipped   {va[41:]}\n      unclipped {vb[41:]}")
    return ch or ["none"]


# ------------------------------------------------------------------ B
def history_of(r):
    h = r.get("history")
    if not isinstance(h, dict):
        h = (r.get("metrics") or {}).get("history")
    return h if isinstance(h, dict) else {}


def epochs_of(r):
    for d in (r, r.get("metrics") if isinstance(r.get("metrics"), dict) else {}, r.get("test_metrics") or {}):
        if isinstance(d, dict) and d.get("epochs_run") is not None:
            return int(d["epochs_run"])
    return None


def part_b():
    out("\n" + "=" * 100)
    out("B. FULL-DATA BUDGET RUNS: does each B_17 run repeat its R_17 run over the epochs R_17 ran?")
    out("=" * 100)
    R, B = load("R_17_fulldata"), load("B_17_fulldata")
    verdict = collections.Counter()
    for key, b in B.items():
        r = R.get(key)
        if r is None:
            verdict["no R_17 partner"] += 1
            continue
        E = epochs_of(r)
        if not E:
            verdict["R_17 has no epoch count"] += 1
            continue
        hr, hb = history_of(r), history_of(b)
        bad = None
        for k in sorted(set(hr) & set(hb)):
            a, c = hr[k], hb[k]
            if not isinstance(a, list) or not isinstance(c, list):
                continue
            if len(c) < E or len(a) < E:
                bad = f"{k}: too short"
                break
            for i in range(E):
                x, y = a[i], c[i]
                if not (x == y or (isinstance(x, float) and isinstance(y, float)
                                   and math.isnan(x) and math.isnan(y))):
                    bad = f"{k}[{i}]"
                    break
            if bad:
                break
        if bad:
            verdict["PREFIX DIFFERENT"] += 1
            out(f"  DIFFERENT {dict(key)['arm']} {dict(key)['dataset']} s={dict(key)['seed']} at {bad}")
        else:
            verdict["PREFIX IDENTICAL"] += 1
    out(f"  {dict(verdict)} of {len(B)} B_17 runs")


# ------------------------------------------------------------------ C
def part_c():
    out("\n" + "=" * 100)
    out("C. HARDWARE (ii): end-of-circuit noise, binary AUC change per run (33_hardware_tuned)")
    out("=" * 100)
    n_runs = n_changed = 0
    worst = 0.0
    for r in shards.load_all("33_hardware_tuned"):
        k = r["keys"]
        if k["dataset"] not in ("breastmnist", "pneumoniamnist"):
            continue
        base = r["depol_curve"]["0.000"]["auc"]
        diffs = {p: m["auc"] - base for p, m in r["depol_curve"].items()}
        mx = max(abs(v) for v in diffs.values())
        n_runs += 1
        n_changed += mx > 0
        worst = max(worst, mx)
        if mx > 0:
            out(f"  {k['dataset']} n={k['regime']} s={k['seed']}: " +
                "  ".join(f"p={p}: {v:+.2e}" for p, v in sorted(diffs.items())))
    out(f"  binary runs {n_runs}; runs with any change {n_changed}; largest |change| {worst:.2e}")


def save(name):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(LINES) + "\n")
    LINES.clear()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("PART2_SKIP_BC"):
        part_a()
        save("part_a.txt")
        print("\nALL DONE. Written to", OUT)
        sys.exit(0)
    if not os.path.exists(os.path.join(OUT, "part_b.txt")):
        part_b()
        save("part_b.txt")
    if not os.path.exists(os.path.join(OUT, "part_c.txt")):
        part_c()
        save("part_c.txt")
    part_a()
    save("part_a.txt")
    print("\nALL DONE. Written to", OUT)
