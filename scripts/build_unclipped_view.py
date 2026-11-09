"""
AMENDMENT 30 - the unclipped condition of a CPU-recorded check, as a view.

    python scripts/build_unclipped_view.py --list 32_squash > /content/cu32_list.txt
    python scripts/build_unclipped_view.py --build 31_entangling_range 32_squash 34_lr_inherited

WHAT A VIEW IS
--------------
X_<source> holds one record per recorded run of <source>: its unclipped CPU
re-execution CU_<source> where clipping acted in it, and the recorded run
itself everywhere else. A run in which no step exceeded 20 is the same
computation with and without clipping (clip_grad_norm_ scales by exactly 1.0
below the threshold), and its clipped CPU re-execution C_<source> reproduced
it bit for bit, so the recorded run IS its unclipped run. This is the rule of
Amendment 21 (analyse_replay.py), applied to the CPU sources with C_ and CU_ in
place of R_ and U_.

The view is made of symbolic links, under its own namespace, in the local
artifact folder: no recorded or re-executed file is copied, edited or moved.
Shards are linked under the view's shard name; prediction files keep their
names, which do not contain the namespace (shards.pred_tag).

WHY IT REFUSES
--------------
A view with a clipped run in it would be labelled "unclipped" and be wrong, so
--build stops unless, for every run of the source:
  - its C_ re-execution exists and is IDENTICAL to the record, and
  - where that C_ run shows a step above 20 or a non-finite step, a CU_ run
    exists and is PREFIX IDENTICAL (Amendment 30, P1).

--list prints the shard names of the runs that need a CU_ run, one per line,
in the form replay.py --from-list reads.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402

SOURCES = ("31_entangling_range", "32_squash", "34_lr_inherited")


def kname(keys):
    return json.dumps(keys, sort_keys=True)


def load(ns):
    return {kname(r["keys"]): r for r in shards.load_all(ns)}


def clipped(rec):
    rr = rec.get("recorded") or {}
    return (rr.get("above_threshold") or 0) > 0 or (rr.get("nonfinite") or 0) > 0


def needs_unclipped(source):
    """Records of C_<source> in which clipping acted."""
    return [r for r in load("C_" + source).values() if clipped(r)]


def clear(view):
    """Remove an earlier view, so a refused build never leaves a stale one behind."""
    for d in (os.path.join(config.SHARD_DIR, view), os.path.join(config.PREDICTION_DIR, view)):
        if os.path.islink(d):
            os.unlink(d)
        elif os.path.isdir(d):
            for fn in os.listdir(d):
                os.unlink(os.path.join(d, fn))
            os.rmdir(d)


def build(source):
    view = "X_" + source
    sdir = os.path.join(config.SHARD_DIR, view)
    pdir = os.path.join(config.PREDICTION_DIR, view)
    clear(view)
    rec, cpu, unc = load(source), load("C_" + source), load("CU_" + source)
    problems = []
    if not rec:
        problems.append(f"{source}: no recorded runs here")
    for key in rec:
        c = cpu.get(key)
        if c is None:
            problems.append(f"no C_ run for {key}")
            continue
        if not str(c.get("replay_check", {}).get("verdict", "")).startswith("IDENTICAL"):
            problems.append(f"C_ run not IDENTICAL: {key}")
        if clipped(c):
            u = unc.get(key)
            if u is None:
                problems.append(f"no CU_ run for clipped {key}")
            elif not str(u.get("replay_check", {}).get("verdict", "")).startswith("PREFIX IDENTICAL"):
                problems.append(f"CU_ run not PREFIX IDENTICAL ({u['replay_check'].get('verdict')}): {key}")
    extra = [k for k in unc if k not in rec or not clipped(cpu.get(k, {}))]
    if extra:
        problems.append(f"{len(extra)} CU_ runs for runs that were not clipped, e.g. {extra[0]}")
    if problems:
        print(f"{source}: VIEW NOT BUILT, {len(problems)} problem(s):")
        for p in problems[:10]:
            print("   ", p)
        return False

    os.makedirs(sdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    n_u = n_p = 0
    for key, r in rec.items():
        keys = r["keys"]
        use = "CU_" + source if key in unc else source
        n_u += use != source
        os.symlink(os.path.realpath(shards.shard_path(use, **keys)),
                   os.path.join(sdir, shards.shard_name(view, **keys)))
        pp = shards.pred_path(use, **keys)
        if os.path.exists(pp):
            os.symlink(os.path.realpath(pp), os.path.join(pdir, os.path.basename(pp)))
            n_p += 1
    print(f"{view}: {len(rec)} runs, {n_u} of them unclipped re-executions, "
          f"{n_p} prediction files. VIEW BUILT")
    return n_p == len(rec)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", choices=SOURCES, help="print the runs that need a CU_ run")
    g.add_argument("--build", nargs="+", choices=SOURCES, help="build X_<source> for each")
    args = ap.parse_args()
    if args.list:
        names = sorted(shards.shard_name(args.list, **r["keys"]) for r in needs_unclipped(args.list))
        print("\n".join(names))
        print(f"{len(names)} runs of {args.list} need an unclipped re-execution", file=sys.stderr)
        return
    ok = all([build(s) for s in args.build])
    print("ALL VIEWS BUILT" if ok else "STOP: a view was not built or lacks predictions")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
