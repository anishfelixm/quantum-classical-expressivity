"""
Amendment 19 analysis: the 1,000-epoch runs against the primary 100-epoch runs.

    python scripts/analyse_budget_check.py                       (from the repo root)
    python scripts/analyse_budget_check.py --shards path/to/artifacts/shards

Standard library only. Reads shards, writes nothing except
logs_stats_budget_check.txt in the current directory.

PAIRING. A run in 30_budget_check is paired with the 01_frozen_tuned run that
has the same dataset, n, d, seed and arm. Same split, same initialisation, same
features, same learning rate; only MAX_EPOCHS and PATIENCE differ.

WHAT IS REPORTED (fixed in Amendment 19 before the run)
-------------------------------------------------------
1. Per dataset and arm: mean test AUC under each budget, the mean paired change
   (1,000-epoch minus 100-epoch), and the epochs run and selected.
2. Per arm, pooled over datasets with equal weight: the mean paired change, with
   a 95% percentile interval from a bootstrap over seeds (B = 2000, seeds
   resampled within each dataset). Prediction P1: |change| < 0.01 for each arm.
3. Delta(5) = AUC(vqc) - AUC(matched), pooled, under each budget, with the same
   seed bootstrap. Prediction P2: positive under the 1,000-epoch budget.

The intervals resample seeds only, not test images: this is an exploratory check
of a protocol constant, reported descriptively, not a member of any test family.
A cell with fewer than all planned seeds is reported with its seed count.
"""
import argparse
import collections
import json
import os
import random
import statistics

BUDGET_NS, PRIMARY_NS = "30_budget_check", "01_frozen_tuned"
ARMS = ("quantum_vqc", "matched_param_fullrank")
B = 2000
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def load(root, ns):
    d = os.path.join(root, ns)
    recs = {}
    if not os.path.isdir(d):
        return recs
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            r = json.load(f)
        k = r.get("keys", {})
        if str(k.get("regime")) != "5" or str(k.get("dim")) != "4" or k.get("arm") not in ARMS:
            continue
        if k.get("bn") or k.get("L") or k.get("as") or k.get("notanh"):
            continue
        m = r.get("metrics", {})
        key = (k["dataset"], k["arm"], int(k["seed"]))
        recs[key] = {"auc": m.get("auc"), "epochs": m.get("epochs_run"),
                     "best": (m.get("best_epoch") + 1) if m.get("best_epoch") is not None else None,
                     "lrh": m.get("lr_head"), "lrq": m.get("lr_quantum")}
    return recs


def pct(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def boot(per_ds, stat):
    """per_ds: dataset -> list of per-seed tuples. Resample seeds within dataset."""
    rng = random.Random(20261015)
    vals = []
    for _ in range(B):
        sample = {ds: [rows[rng.randrange(len(rows))] for _ in rows] for ds, rows in per_ds.items()}
        vals.append(stat(sample))
    return pct(vals, 0.025), pct(vals, 0.975)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default=os.path.join("artifacts", "shards"))
    args = p.parse_args()

    bud, pri = load(args.shards, BUDGET_NS), load(args.shards, PRIMARY_NS)
    out(f"budget runs: {len(bud)}   matching primary runs available: "
        f"{sum(1 for k in bud if k in pri)}")
    if not bud:
        out("No 30_budget_check shards found.")
        return
    lrs = collections.Counter((v["lrh"], v["lrq"]) for v in bud.values())
    out(f"learning rates in budget runs (lr_head, lr_quantum): {dict(lrs)}")

    datasets = sorted({k[0] for k in bud})
    out("\n=== 1. Per dataset and arm (n = 5, d = 4) ===")
    out(f"{'dataset':16s} {'arm':24s} {'seeds':>5s} {'AUC 100ep':>9s} {'AUC 1000ep':>10s} "
        f"{'change':>8s} {'epochs 100/1000':>16s} {'selected 100/1000':>18s}")
    paired = {}   # (ds, arm) -> list of (seed, auc_std, auc_ext)
    for ds in datasets:
        for arm in ARMS:
            rows = []
            for (d_, a_, s), v in sorted(bud.items()):
                if d_ == ds and a_ == arm and (d_, a_, s) in pri:
                    w = pri[(d_, a_, s)]
                    if v["auc"] is not None and w["auc"] is not None:
                        rows.append((s, w["auc"], v["auc"], w["epochs"], v["epochs"], w["best"], v["best"]))
            paired[(ds, arm)] = rows
            if not rows:
                out(f"{ds:16s} {arm:24s} {0:5d}  (no pairs)")
                continue
            m = statistics.mean
            out(f"{ds:16s} {arm:24s} {len(rows):5d} {m(r[1] for r in rows):9.4f} "
                f"{m(r[2] for r in rows):10.4f} {m(r[2] - r[1] for r in rows):+8.4f} "
                f"{m(r[3] for r in rows):7.1f}/{m(r[4] for r in rows):7.1f} "
                f"{m(r[5] for r in rows):8.1f}/{m(r[6] for r in rows):8.1f}")

    out("\n=== 2. Pooled change per arm (1000-epoch minus 100-epoch), equal weight per dataset ===")
    out("Prediction P1: |change| < 0.01 for each arm.")
    for arm in ARMS:
        per_ds = {ds: [(r[1], r[2]) for r in paired[(ds, arm)]] for ds in datasets if paired[(ds, arm)]}
        if not per_ds:
            continue
        stat = lambda smp: statistics.mean(statistics.mean(b - a for a, b in rows) for rows in smp.values())
        est = stat(per_ds)
        lo, hi = boot(per_ds, stat)
        verdict = "P1 holds" if abs(est) < 0.01 else "P1 FAILS"
        out(f"  {arm:24s} {est:+.4f}  95% seed-bootstrap [{lo:+.4f}, {hi:+.4f}]  "
            f"({len(per_ds)} datasets)  {verdict}")

    out("\n=== 3. Delta(5) = VQC - matched, pooled, under each budget ===")
    out("Prediction P2: positive under the 1,000-epoch budget.")
    per_ds = {}
    for ds in datasets:
        q = {r[0]: r for r in paired[(ds, "quantum_vqc")]}
        c = {r[0]: r for r in paired[(ds, "matched_param_fullrank")]}
        common = sorted(set(q) & set(c))
        if common:
            per_ds[ds] = [(q[s][1] - c[s][1], q[s][2] - c[s][2]) for s in common]
    if per_ds:
        for idx, label in ((0, "100 epochs (primary runs, same seeds)"), (1, "1000 epochs")):
            stat = lambda smp, i=idx: statistics.mean(statistics.mean(r[i] for r in rows) for rows in smp.values())
            est = stat(per_ds)
            lo, hi = boot(per_ds, stat)
            out(f"  {label:38s} Delta(5) = {est:+.4f}  95% seed-bootstrap [{lo:+.4f}, {hi:+.4f}]")
        ext = statistics.mean(statistics.mean(r[1] for r in rows) for rows in per_ds.values())
        out(f"  P2 {'holds' if ext > 0 else 'FAILS'} (point estimate under 1000 epochs: {ext:+.4f}); "
            f"seeds per dataset: { {ds: len(v) for ds, v in per_ds.items()} }")

    dst = os.path.join(os.getcwd(), "logs_stats_budget_check.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    out(f"\nwritten to {dst}")


if __name__ == "__main__":
    main()
