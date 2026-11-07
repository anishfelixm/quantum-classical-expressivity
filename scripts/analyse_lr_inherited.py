"""
AMENDMENT 29 - WERE THE INHERITED LEARNING RATES THE ONES TUNING WOULD PICK?
Exploratory.

    python scripts/analyse_lr_inherited.py              # after the runs
    python scripts/analyse_lr_inherited.py --partial    # progress only: counts, no AUC

WHY
---
Four arms were tuned (Amendments 3 and 3a): the identity, matched, Fourier and
VQC heads. Every arm added later took the rate of the arm it is built from
(config.LR_INHERITANCE), and that choice was never checked. In the tuning sweep a
threefold change in rate moved the VQC's validation AUC by up to 0.013, about the
size of the re-uploading, readout and restriction effects. Amendment 29 runs the
same tuning procedure, unchanged, for every inherited arm, and also for the two
arms compared at d = 8 (tuned at d = 4 only).

THE PROCEDURE (src/09_lr_selection.py, unchanged)
------------------------------------------------
Grid {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}; tuning seeds 90001-90005 (disjoint from
the confirmatory seeds); n in {5, 20, 100}; the four datasets; frozen backbone,
learned projection; the selected rate is the grid point with the highest mean
VALIDATION AUC over all tuning runs. Test metrics are never consulted.

PREDICTIONS (Amendment 29, written before any run)
    P1  For every checked arm, the procedure selects the inherited rate (1e-2).
    P2  For every checked arm, the mean validation AUC at the inherited rate is
        within 0.005 of the best grid point.
    The per-n choices and the curves are descriptive.

WHAT IT CAN AND CANNOT DO
-------------------------
It can say whether each inherited rate is the one the tuning rule would have
chosen, and how much validation AUC the inherited rate gives up. It cannot change
a reported result: re-running a comparison at a rate chosen after seeing it would
be a choice made after the data. Any arm whose inherited rate is not selected is
named in the paper with the size of the gap.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402

NS = "34_lr_inherited"
GRID = ["3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01"]
REGIMES = ["5", "20", "100"]
SEEDS = [90001, 90002, 90003, 90004, 90005]
THRESHOLD = 0.005

# What each checked arm inherited: the rate tuning selected for its parent
# (Amendment 3a: quantum_vqc, matched_param_fullrank and fourier_rff 1e-2; linear
# 3e-2). At d = 8 the arms use the rate tuned at d = 4.
CHECKED = [
    # (arm, rank, d, parent, inherited rate)
    ("quantum_reupload",    None, 4, "quantum_vqc",            "1e-02"),
    ("quantum_vqc_range1",  None, 4, "quantum_vqc",            "1e-02"),
    ("quantum_rich",        None, 4, "quantum_vqc",            "1e-02"),
    ("quantum_rich_padded", None, 4, "quantum_vqc",            "1e-02"),
    ("quantum_basic",       None, 4, "quantum_vqc",            "1e-02"),
    ("low_rank",            0,    4, "matched_param_fullrank", "1e-02"),
    ("low_rank",            2,    4, "matched_param_fullrank", "1e-02"),
    ("low_rank",            8,    4, "matched_param_fullrank", "1e-02"),
    ("fourier_rff_r2",      None, 4, "fourier_rff",            "1e-02"),
    ("quantum_vqc",         None, 8, "quantum_vqc (tuned at d = 4)", "1e-02"),
    ("low_rank",            2,    8, "matched_param_fullrank (tuned at d = 4)", "1e-02"),
]
EXPECTED_RUNS = len(GRID) * len(REGIMES) * 4 * len(SEEDS)          # 360 per arm


def label(arm, rank, d):
    return f"{arm}{'' if rank is None else f' (rank {rank})'}, d = {d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="print run counts only (no AUC is shown while runs are going)")
    args = ap.parse_args()

    rows = shards.load_all(NS)
    by = defaultdict(list)
    for r in rows:
        k = r["keys"]
        rank = int(k["rank"]) if "rank" in k else None
        by[(k["arm"], rank, int(k["dim"]))].append(r)

    print(f"AMENDMENT 29 (exploratory) | {len(rows)} runs | sha={config.git_sha()[:8]}")
    shas = sorted({r.get("git_sha", "?")[:8] for r in rows})
    print(f"commits in {NS}: {shas}")
    complete = True
    for arm, rank, d, _, _ in CHECKED:
        n = len(by.get((arm, rank, d), []))
        complete &= (n == EXPECTED_RUNS)
        print(f"  {label(arm, rank, d):40s} {n:4d} of {EXPECTED_RUNS}")
    if args.partial:
        return
    if not complete:
        sys.exit("not complete: no selection is made on a partial grid")
    if len(shas) != 1:
        sys.exit(f"more than one commit in {NS}: {shas}")

    out, p1, p2 = [], True, True
    print("\nMean validation AUC by learning rate (all tuning runs pooled; test never used)")
    print(f"{'arm':40s} " + " ".join(f"{g:>8s}" for g in GRID) + "  selected inherited    gap")
    for arm, rank, d, parent, inherited in CHECKED:
        runs = by[(arm, rank, d)]
        curve = {}
        for g in GRID:
            v = [r["val_auc"] for r in runs if r["keys"]["lr"] == g and r.get("val_auc") is not None]
            curve[g] = float(np.mean(v)) if v else float("nan")
        best = max(GRID, key=lambda g: curve[g])
        gap = curve[best] - curve[inherited]
        p1 &= (best == inherited)
        p2 &= (gap < THRESHOLD)
        per_n = {}
        for n in REGIMES:
            m = {g: np.mean([r["val_auc"] for r in runs if r["keys"]["lr"] == g
                             and str(r["keys"]["regime"]) == n and r.get("val_auc") is not None])
                 for g in GRID}
            per_n[n] = max(m, key=m.get)
        edge = best in (GRID[0], GRID[-1])
        print(f"{label(arm, rank, d):40s} " + " ".join(f"{curve[g]:8.4f}" for g in GRID)
              + f"  {best:>8s} {inherited:>9s} {gap:+.4f}"
              + ("  EDGE" if edge else ""))
        out.append({"arm": arm, "rank": rank, "d": d, "parent": parent,
                    "inherited": inherited, "selected": best, "gap": gap,
                    "curve": curve, "per_n_selected": per_n, "edge": edge})

    print("\nPer-n choices (descriptive):")
    for o in out:
        print(f"  {label(o['arm'], o['rank'], o['d']):40s} "
              + "  ".join(f"n={n}:{o['per_n_selected'][n]}" for n in REGIMES))

    print("\n" + "=" * 78)
    print("PREDICTIONS (Amendment 29)")
    print("=" * 78)
    print(f"P1  every checked arm selects its inherited rate: {'HOLDS' if p1 else 'FAILS'}")
    for o in out:
        if o["selected"] != o["inherited"]:
            print(f"      {label(o['arm'], o['rank'], o['d'])}: selected {o['selected']} "
                  f"(inherited {o['inherited']})")
    print(f"P2  inherited rate within {THRESHOLD} validation AUC of the best: "
          f"{'HOLDS' if p2 else 'FAILS'}")
    for o in out:
        if o["gap"] >= THRESHOLD:
            print(f"      {label(o['arm'], o['rank'], o['d'])}: gap {o['gap']:+.4f}")

    path = os.path.join(config.ARTIFACT_ROOT, "lr_inherited.json")
    with open(path, "w") as f:
        json.dump({"threshold": THRESHOLD, "arms": out}, f, indent=1)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
