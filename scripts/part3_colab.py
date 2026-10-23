"""
Part 3 (23 October 2026): every number in the manuscript, recomputed with and
without gradient clipping by the scripts that produced it. No training.

WHY THIS REPLACES PART 2's SECTION A
------------------------------------
Amendment 21 requires every reported number to be recomputed on the unclipped
view "by the script that produced them". Part 2 used 04_statistical_analysis.py
for that, but 04's per-cell tables are not where the manuscript's numbers come
from. They come from:
  13_family_table.py        the 23 confirmatory tests (part 1 did these)
  14_exploratory_table.py   the exploratory and follow-up rows (part 1 did these)
  15_reporting_tables.py    absolute AUC, the F1-selected sensitivity analysis,
                            the sensor-noise table, effect sizes, and the nested
                            intervals for d = 8, d = 16 and full data
  generate_paper_plots.py   the figures, including the per-dataset breakdown and
                            the leave-one-dataset-out intervals (from 13's caches)
So this part runs exactly those four scripts, unchanged, on two copies of the
artifact folder that differ only in which runs they contain:
  clip    every recorded run; for 17_fulldata its GPU re-execution R_17, because
          the GPU re-executions cannot reproduce the CPU records bit for bit,
          and the clipping contrast must pair runs from the same device
  unclip  the same runs, with every run whose optimisation took a step above 20
          replaced by its unclipped re-execution (U_), i.e. the X_ views
The copies are folders of links, so nothing is duplicated and the published
files are never touched. Then every number in the two sets of outputs is
compared, and every change is listed, largest first, with any verdict change.

The "clip" outputs must also reproduce the published tables wherever 17_fulldata
is not involved: that is the check that this machine computes what the cluster did.

    python /content/part3.py            (from /content/repo, QML_ARTIFACT_ROOT set)
Resumable: a script whose output is complete (exit 0) is not run again.
"""
import json
import math
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("PART2_REPO", "/content/repo")
ART = os.environ.get("QML_ARTIFACT_ROOT", os.path.join(REPO, "artifacts"))
OUT = os.environ.get("PART3_OUT", os.path.join(REPO, "logs_part3"))
WORK = os.path.join(os.path.dirname(ART), "mirrors")
sys.path.insert(0, os.path.join(REPO, "src"))
import shards   # noqa: E402

SOURCES = ["01_frozen_tuned", "03_robustness_v2", "09_lr_selection", "10_capacity",
           "12_bottleneck", "14_readout", "15_dim8", "16_dim16", "17_fulldata", "18_depth",
           "19_angle", "20_tanh", "21_ansatz", "25_encoder", "26_bottleneck_tuned",
           "27_capacity_tuned", "29_compression_frozen"]
DERIVED = ("R_", "U_", "B_", "C_", "X_")
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def kname(k):
    return tuple(sorted((a, str(b)) for a, b in k.items()))


def load(ns):
    return {kname(r["keys"]): r for r in shards.load_all(ns)}


def build_view(ns, base):
    """X_<ns>: U_ run where one exists, else the run in `base` (hard links; as in analyse_replay.py)."""
    import config
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
    return n_u, len(rec)


def mirrors():
    out("=" * 100)
    out("1. THE TWO ARTIFACT FOLDERS")
    out("=" * 100)
    choice = {"clip": {}, "unclip": {}}
    for ns in SOURCES:
        rep = load("R_" + ns)
        rec = load(ns)
        ident = bool(rec) and all(k in rep and str(rep[k].get("replay_check", {}).get("verdict", ""))
                                  .startswith("IDENTICAL") for k in rec)
        base = ns if ident else "R_" + ns
        n_u, n = build_view(ns, base)
        choice["clip"][ns], choice["unclip"][ns] = base, "X_" + ns
        out(f"  {ns:24s} clip = {base:26s} unclip = X_{ns:24s} ({n_u} of {n} runs unclipped)")
    others = sorted(d for d in os.listdir(os.path.join(ART, "shards"))
                    if not d.startswith(DERIVED) and d not in SOURCES)
    out(f"  other namespaces, the same in both: {', '.join(others)}")
    for m in ("clip", "unclip"):
        root = os.path.join(WORK, m)
        if os.path.isdir(root):
            shutil.rmtree(root)
        for sub in ("shards", "predictions"):
            os.makedirs(os.path.join(root, sub))
            for ns in SOURCES + others:
                src = os.path.join(ART, sub, choice[m].get(ns, ns))
                if os.path.isdir(src):
                    os.symlink(src, os.path.join(root, sub, ns))
    return {m: os.path.join(WORK, m) for m in ("clip", "unclip")}


def finished(path):
    return os.path.exists(path) and open(path).read().rstrip().endswith("exit code 0")


def run(m, root, name, cmd):
    path = os.path.join(OUT, f"{m}_{name}.txt")
    if finished(path):
        out(f"  {m:6s} {name:10s} already done")
        return
    env = dict(os.environ, QML_ARTIFACT_ROOT=root, MPLBACKEND="Agg")
    with open(path + ".part", "w") as f:
        f.write("command: " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\nexit code {rc}\n")
    os.replace(path + ".part", path)
    out(f"  {m:6s} {name:10s} exit {rc}" + ("" if rc == 0 else f"   <-- FAILED, see {os.path.basename(path)}"))


def flatten(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, x


def label_rows(x):
    """Lists of table rows are keyed by their identifying fields, not by position."""
    if isinstance(x, dict):
        return {k: label_rows(v) for k, v in x.items()}
    if isinstance(x, list) and x and all(isinstance(r, dict) for r in x):
        ident = ("test", "group", "namespace", "arm", "dataset", "regime", "sigma", "tag", "name",
                 "contrast", "n", "cells")
        out_ = {}
        for i, r in enumerate(x):
            key = "|".join(f"{k}={r[k]}" for k in ident if k in r and not isinstance(r[k], (dict, list)))
            out_[key or str(i)] = label_rows(r)
        return out_
    if isinstance(x, list):
        return [label_rows(v) for v in x]
    return x


def compare(a_path, b_path, title, skip_full=False):
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        out(f"\n{title}: not compared ({'first' if not os.path.exists(a_path) else 'second'} file missing)")
        return
    A = dict(flatten(label_rows(json.load(open(a_path)))))
    B = dict(flatten(label_rows(json.load(open(b_path)))))
    num, txt, only = [], [], []
    for k in sorted(set(A) | set(B)):
        if skip_full and ("full" in k.lower() or "17_fulldata" in k):
            continue
        if any(s in k for s in ("seconds", "timestamp", "sha", "git", "time")):
            continue
        if k not in A or k not in B:
            only.append(k)
            continue
        x, y = A[k], B[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not (x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))):
                num.append((abs(y - x) if math.isfinite(y - x) else float("inf"), k, x, y))
        elif x != y:
            txt.append((k, x, y))
    out(f"\n{title}")
    out(f"  numbers compared {sum(1 for k in A if k in B)}; changed {len(num)}; "
        f"text changed {len(txt)}; present on one side only {len(only)}")
    for k, x, y in txt[:40]:
        out(f"  TEXT  {k}\n        {x!r}  ->  {y!r}")
    for d, k, x, y in sorted(num, reverse=True)[:40]:
        out(f"  {d:9.2e}  {k}\n        {x!r}  ->  {y!r}")
    for k in only[:10]:
        out(f"  one side only: {k}")


def main():
    os.makedirs(OUT, exist_ok=True)
    roots = mirrors()
    W = str(max(1, (os.cpu_count() or 2) - 1))
    py = sys.executable
    out("\n" + "=" * 100)
    out("2. THE FOUR SCRIPTS ON EACH FOLDER")
    out("=" * 100)
    for m, root in roots.items():
        run(m, root, "13_family", [py, "src/13_family_table.py", "--recompute", "--workers", W])
        run(m, root, "14_explor", [py, "src/14_exploratory_table.py", "--recompute", "--workers", W])
        run(m, root, "15_report", [py, "src/15_reporting_tables.py", "--artifacts", root,
                                   "--out-tables", os.path.join(root, "tables"), "--nested"])
        run(m, root, "plots", [py, "src/eval/generate_paper_plots.py", "--artifacts", root,
                               "--out", os.path.join(root, "figures")])
        keep = os.path.join(OUT, m)
        os.makedirs(keep, exist_ok=True)
        for rel in ("family_table.json", "exploratory_table.json", "reporting/reporting_tables.json",
                    "figures/figure_data.json"):
            p = os.path.join(root, rel)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(keep, os.path.basename(rel)))
        for d in ("tables", "figures"):
            if os.path.isdir(os.path.join(root, d)):
                shutil.copytree(os.path.join(root, d), os.path.join(keep, d), dirs_exist_ok=True)

    out("\n" + "=" * 100)
    out("3. EVERY NUMBER: CLIPPED (clip) -> UNCLIPPED (unclip)")
    out("=" * 100)
    for f in ("family_table.json", "exploratory_table.json", "reporting_tables.json", "figure_data.json"):
        compare(os.path.join(OUT, "clip", f), os.path.join(OUT, "unclip", f), f)

    out("\n" + "=" * 100)
    out("4. REPRODUCTION: this machine's clip outputs vs the published files (full data excluded)")
    out("=" * 100)
    for pub, mine in ((os.path.join(ART, "family_table.json"), "family_table.json"),
                      (os.path.join(ART, "exploratory_table.json"), "exploratory_table.json"),
                      (os.path.join(ART, "reporting", "reporting_tables.json"), "reporting_tables.json"),
                      (os.path.join(REPO, "paper", "figures", "figure_data.json"), "figure_data.json")):
        compare(pub, os.path.join(OUT, "clip", mine), "published " + os.path.relpath(pub, REPO), skip_full=True)
    for m in ("clip", "unclip"):
        for name in ("13_family", "14_explor", "15_report", "plots"):
            p = os.path.join(OUT, f"{m}_{name}.txt")
            if os.path.exists(p) and "SEED-LEVEL FALLBACK" in open(p).read():
                out(f"!!! SEED-LEVEL FALLBACK IN USE in {os.path.basename(p)}: STOP and report")
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nALL DONE. Written to", OUT)


if __name__ == "__main__":
    main()
