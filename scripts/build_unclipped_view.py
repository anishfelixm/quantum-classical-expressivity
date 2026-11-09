"""
AMENDMENT 30 - the unclipped condition of a CPU-recorded check, as a view.

    python scripts/build_unclipped_view.py --list 32_squash > /content/cu32_list.txt
    python scripts/build_unclipped_view.py --check-steps 31_entangling_range 32_squash 34_lr_inherited
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

THE PREFIX CHECK, AT STEP LEVEL (added 9 November 2026, see Amendment 30 note)
-----------------------------------------------------------------------------
An unclipped run must be the same computation as its clipped run until
clipping first acts. replay.py checks this per EPOCH, on the per-epoch history
in the two records. The tuning runs of 34_lr_inherited (09_lr_selection.py)
store no per-epoch history, so replay.py could not compare 148 of them ("not
compared (no per-epoch history in the record)"), and for the 201 runs clipped
in their first epoch its check covers zero epochs.

Every re-execution also stores the pre-clip gradient norm of every
optimisation step (recorders.py). Until the first step whose norm exceeds 20
(or is not finite), the clipped and unclipped runs perform identical updates,
so their step norms up to AND INCLUDING that step - its norm is computed before
clipping acts - must be bit-identical. After it they may differ. This check is
finer than the epoch check (it covers the first clipped epoch up to the clipped
step) and needs no history, so it is applied to EVERY unclipped run of every
source, and the view requires it.

WHY IT REFUSES
--------------
A view with a clipped run in it would be labelled "unclipped" and be wrong, so
--build stops unless, for every run of the source:
  - its C_ re-execution exists and is IDENTICAL to the record, and
  - where that C_ run shows a step above 20 or a non-finite step, a CU_ run
    exists, its replay verdict is PREFIX IDENTICAL or "not compared (no
    per-epoch history in the record)", and its step norms pass the step-level
    prefix check above (Amendment 30, P1).

--list prints the shard names of the runs that need a CU_ run, one per line,
in the form replay.py --from-list reads.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402

SOURCES = ("31_entangling_range", "32_squash", "34_lr_inherited",
           "35_lr_nosquash", "36_nosquash", "37_rate_gaps")      # 35-37: Amendment 31
NO_PREDICTIONS = ("34_lr_inherited", "35_lr_nosquash")   # tuning runs store none (09_lr_selection.py)
THRESHOLD = 20.0          # recorders.THRESHOLD, the clip norm of every reported run
NO_HISTORY = "not compared (no per-epoch history in the record)"


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


def norms_of(rec):
    rr = rec.get("recorded") or {}
    if float(rr.get("threshold", THRESHOLD)) != THRESHOLD:
        raise RuntimeError(f"recorded threshold {rr.get('threshold')} is not {THRESHOLD}")
    path = os.path.join(config.ARTIFACT_ROOT, rr.get("step_norms_file", ""))
    if not rr.get("step_norms_file") or not os.path.exists(path):
        return None
    with np.load(path) as z:
        return np.asarray(z["norms"], np.float64)


def step_prefix(c_rec, u_rec):
    """
    (ok, detail): the CU_ run's step norms equal the C_ run's, bit for bit, over
    every step up to and including the first step above 20 or not finite.
    """
    a, b = norms_of(c_rec), norms_of(u_rec)
    if a is None or b is None:
        return False, "step record missing"
    bad = [i for i, x in enumerate(a) if not math.isfinite(x) or x > THRESHOLD]
    if not bad:
        return False, "no step above 20 in the C_ run"
    i = bad[0]
    if len(b) <= i:
        return False, f"CU_ run has {len(b)} steps, fewer than the {i + 1} to compare"
    if not np.array_equal(a[:i + 1], b[:i + 1], equal_nan=True):
        j = int(np.flatnonzero(~((a[:i + 1] == b[:i + 1]) | (np.isnan(a[:i + 1]) & np.isnan(b[:i + 1]))))[0])
        return False, f"step {j + 1} differs ({a[j]!r} vs {b[j]!r}); first clipped step {i + 1}"
    return True, f"{i + 1} steps identical"


def verify(source, quiet=False):
    """Every check the view needs, without building it. Returns (problems, counts)."""
    rec, cpu, unc = load(source), load("C_" + source), load("CU_" + source)
    problems, counts = [], {"runs": len(rec), "clipped": 0, "epoch_prefix": 0,
                            "no_history": 0, "step_prefix_ok": 0, "steps_compared": 0}
    if not rec:
        problems.append(f"{source}: no recorded runs here")
    for key in rec:
        c = cpu.get(key)
        if c is None:
            problems.append(f"no C_ run for {key}")
            continue
        if not str(c.get("replay_check", {}).get("verdict", "")).startswith("IDENTICAL"):
            problems.append(f"C_ run not IDENTICAL: {key}")
        if not clipped(c):
            continue
        counts["clipped"] += 1
        u = unc.get(key)
        if u is None:
            problems.append(f"no CU_ run for clipped {key}")
            continue
        v = str(u.get("replay_check", {}).get("verdict", ""))
        if v.startswith("PREFIX IDENTICAL"):
            counts["epoch_prefix"] += 1
        elif v == NO_HISTORY:
            counts["no_history"] += 1
        else:
            problems.append(f"CU_ verdict {v!r}: {key}")
        ok, detail = step_prefix(c, u)
        if ok:
            counts["step_prefix_ok"] += 1
            counts["steps_compared"] += int(detail.split()[0])
        else:
            problems.append(f"step-level prefix check failed ({detail}): {key}")
    extra = [k for k in unc if k not in rec or not clipped(cpu.get(k, {}))]
    if extra:
        problems.append(f"{len(extra)} CU_ runs for runs that were not clipped, e.g. {extra[0]}")
    if not quiet:
        print(f"{source}: {counts['runs']} runs, clipped {counts['clipped']}; CU_ verdicts: "
              f"epoch prefix identical {counts['epoch_prefix']}, no per-epoch history "
              f"{counts['no_history']}; step-level prefix identical {counts['step_prefix_ok']} "
              f"of {counts['clipped']} ({counts['steps_compared']} steps compared)"
              + ("" if not problems else f"; {len(problems)} PROBLEM(S)"))
        for p in problems[:10]:
            print("   ", p)
    return problems, counts


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
    problems, _ = verify(source)
    if problems:
        print(f"{source}: VIEW NOT BUILT")
        return False

    rec, unc = load(source), load("CU_" + source)
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
    # 09_lr_selection runs store no predictions; the selection reads val_auc from the shard.
    need_pred = source not in NO_PREDICTIONS
    print(f"{view}: {len(rec)} runs, {n_u} of them unclipped re-executions, "
          f"{n_p} prediction files. VIEW BUILT")
    return n_p == len(rec) or not need_pred


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", choices=SOURCES, help="print the runs that need a CU_ run")
    g.add_argument("--check-steps", nargs="+", choices=SOURCES,
                   help="Amendment 30 P1 at step level, for each source; builds nothing")
    g.add_argument("--build", nargs="+", choices=SOURCES, help="build X_<source> for each")
    args = ap.parse_args()
    if args.list:
        names = sorted(shards.shard_name(args.list, **r["keys"]) for r in needs_unclipped(args.list))
        print("\n".join(names))
        print(f"{len(names)} runs of {args.list} need an unclipped re-execution", file=sys.stderr)
        return
    if args.check_steps:
        bad = sum(len(verify(s)[0]) for s in args.check_steps)
        print("P1 (step level) HOLDS for every unclipped run" if bad == 0
              else f"P1 (step level) FAILS: {bad} problem(s) listed above")
        sys.exit(0 if bad == 0 else 1)
    ok = all([build(s) for s in args.build])
    print("ALL VIEWS BUILT" if ok else "STOP: a view was not built or lacks predictions")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
