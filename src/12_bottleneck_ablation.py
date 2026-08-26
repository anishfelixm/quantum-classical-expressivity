"""
EXPERIMENT 12 - BOTTLENECK ABLATION.  IS THE HEAD DOING THE WORK?

THE PROBLEM THIS EXISTS TO FIX
-------------------------------
Freezing the backbone is not the same as isolating the head. At d=4 with two
classes, the trainable budget of the "frozen backbone" experiment is:

    bottleneck Linear(256, 4)   1,028      97%
    head                           24       2%
    classifier Linear(4, 2)        10       1%

The head is the smallest component in the model by a factor of forty. A
1,028-parameter learned projection has ample capacity to reshape the latent
space to suit whichever head follows - which is the same absorption effect
measured at the encoder in Q3, one layer further down, and it was never
controlled for.

So the honest reading of every frozen-backbone result so far is: "with a learned
256->4 projection free to adapt to each head, the heads perform about the same."
That is a weaker statement than the one the paper wants to make, and a reviewer
who counts parameters will notice.

THE FIX
-------
Freeze the projection too, and compare the head ordering across three policies:

    learned   trainable Linear(256, d). The status quo; head holds ~2%.
    pca       top-d principal directions of the TRAINING features, frozen.
              Optimal linear compression in the mean-squared sense.
    random    fixed Gaussian projection, frozen (Johnson-Lindenstrauss).
              Approximately distance-preserving, and arm-agnostic by
              construction.

Under either frozen policy the head holds 24 of 34 trainable parameters (~70%)
and is the dominant learner. This is the only configuration in which "the
difference is a property of the head's function class" is a claim the design
actually supports.

WHY BOTH pca AND random
------------------------
They fail in opposite directions, so agreement between them is informative.

    pca alone      invites "you chose a projection that happens to suit one
                   head" - it is fitted to the data, after all.
    random alone   invites "your projection is bad, so you are comparing heads
                   on degraded features."

If the head ordering is the SAME under an optimal projection and a random one,
neither objection survives, and the ordering is a property of the heads.

WHAT TO LOOK FOR
----------------
1. Does the primary contrast (quantum - classical) keep its SIGN across all
   three policies? If it flips when the bottleneck is frozen, the original
   result was about the projection, not the head.
2. Does the scarcity crossover survive? The crossover is the paper's headline;
   if it only exists with a learned projection, that must be said plainly.
3. How much absolute AUC is lost by freezing? Large loss means the learned
   projection was doing most of the work all along - itself a finding worth
   reporting, and one the conference paper implicitly relied on without knowing.

PREDICTION, RECORDED BEFORE THE RUN
------------------------------------
Freezing the bottleneck will lower absolute AUC for every arm. The QUESTION is
whether the between-arm ordering is preserved. If the crossover survives under
both frozen policies, the mechanism claim is substantially stronger. If it
disappears, the crossover was an artifact of a learned projection adapting
differently to different heads - which must then be reported as the finding.

SCOPE
-----
Frozen encoder, cached features, d=4. pca needs the training features, so it is
only defined on the cached path.

USAGE
-----
    python src/12_bottleneck_ablation.py --quick
    python src/12_bottleneck_ablation.py
    python src/12_bottleneck_ablation.py --summary-only
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm, QUANTUM_ARMS             # noqa: E402
from train.loop import train_model                              # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features
FROZEN = _exp1.FROZEN

EXPERIMENT = "12_bottleneck"

POLICIES = ["learned", "pca", "random"]
ARMS = list(config.PRIMARY_COMPARISON) + ["linear"]


def run_cell(dataset, n_per_class, seed, arm, policy, dim=4, force=False):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed,
                arm=arm, bn=policy)
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    C = num_classes_of(dataset)
    t0 = time.time()

    loaders, meta = get_cached_features(dataset, str(n_per_class), seed)
    model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                      seed=seed, build_backbone=False,
                      bottleneck_policy=policy)

    if policy == "pca":
        # TRAINING features only. Fitting on val or test would leak.
        model.fit_bottleneck(_exp1.get_pool_features(dataset))

    capacity = model.capacity_report()

    metrics, history, _, probs, labels = train_model(
        model, loaders["train"], loaders["val"], loaders["test"],
        num_classes=C, use_features=True,
        is_quantum=(arm in QUANTUM_ARMS), verbose=False, return_probs=True)

    payload = {
        "metrics": metrics,
        "capacity": capacity,
        "bottleneck_policy": policy,
        "pca_variance_retained": model.pca_variance_retained,
        "predictions_file": shards.save_predictions(
            EXPERIMENT, labels, probs, **keys),
        "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test", "regime")},
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del model
    torch.cuda.empty_cache()
    return metrics


# ------------------------------------------------------------------ summary
def _paired(a, b):
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common
                  if a[s] is not None and b[s] is not None])
    if len(d) < 2:
        return None
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, m - half, m + half, len(d)


def summarise(metric="auc"):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    # (dataset, regime, policy) -> arm -> seed -> metric
    tbl = defaultdict(lambda: defaultdict(dict))
    cap, pca_var = {}, defaultdict(list)
    for r in rows:
        k = r["keys"]
        tbl[(k["dataset"], int(k["regime"]), k["bn"])][k["arm"]][k["seed"]] = \
            r["metrics"].get(metric)
        if r.get("capacity"):
            cap[(k["arm"], k["bn"])] = r["capacity"]
        if r.get("pca_variance_retained") is not None:
            pca_var[k["dataset"]].append(r["pca_variance_retained"])

    policies = [p for p in POLICIES if any(c[2] == p for c in tbl)]
    arms = sorted({a for c in tbl.values() for a in c})
    regimes = sorted({reg for (_, reg, _) in tbl})

    # ---------------------------------------------------------- capacity
    print("=" * 78)
    print("WHERE THE TRAINABLE PARAMETERS LIVE")
    print("=" * 78)
    print("The point of the whole experiment: with a learned projection the head")
    print("holds ~2% of the trainable budget, so 'the head's function class' is")
    print("not what the frozen-backbone comparison was mainly measuring.")
    print(f"\n{'arm':24s} {'bottleneck':>10s} {'bneck':>7s} {'head':>6s} "
          f"{'clf':>5s} {'total':>7s} {'head %':>7s}")
    print("-" * 70)
    for (arm, pol) in sorted(cap):
        c = cap[(arm, pol)]
        print(f"{arm:24s} {pol:>10s} {c['bottleneck']:>7d} {c['head']:>6d} "
              f"{c['classifier']:>5d} {c['total']:>7d} "
              f"{100 * c['head_share']:>6.1f}%")
    if pca_var:
        print("\nPCA variance retained at d=4:")
        for ds in sorted(pca_var):
            print(f"  {ds:16s} {np.mean(pca_var[ds]):.4f}")

    # ---------------------------------------------------------- absolute
    print("\n" + "=" * 78)
    print(f"{metric.upper()} BY BOTTLENECK POLICY")
    print("=" * 78)
    print("Absolute loss from freezing measures how much work the learned")
    print("projection was doing. Large loss is itself a reportable finding.")
    print(f"\n{'dataset':16s} {'n/cls':>6s} {'arm':>24s}  " +
          "  ".join(p.rjust(9) for p in policies))
    print("-" * (50 + 11 * len(policies)))
    for ds in sorted({c[0] for c in tbl}):
        for reg in regimes:
            for arm in arms:
                cells, present = [], False
                for pol in policies:
                    v = [x for x in tbl[(ds, reg, pol)].get(arm, {}).values()
                         if x is not None]
                    if v:
                        present = True
                        cells.append(f"{np.mean(v):9.4f}")
                    else:
                        cells.append("        -")
                if present:
                    print(f"{ds:16s} {reg:>6d} {arm:>24s}  " + "  ".join(cells))

    # ------------------------------------------------- the primary contrast
    arm_a, arm_b = config.PRIMARY_COMPARISON
    print("\n" + "=" * 78)
    print(f"PRIMARY CONTRAST: {arm_a} - {arm_b}, BY POLICY")
    print("=" * 78)
    print("Does the sign survive freezing the projection? If it flips, the")
    print("original result was about the bottleneck, not the head.")
    print(f"\n{'n/cls':>6s}  " + "  ".join(p.rjust(11) for p in policies))
    print("-" * (8 + 13 * len(policies)))

    by_policy = {p: {} for p in policies}
    for reg in regimes:
        cells = []
        for pol in policies:
            deltas = []
            for ds in sorted({c[0] for c in tbl}):
                out = _paired(tbl[(ds, reg, pol)].get(arm_a, {}),
                              tbl[(ds, reg, pol)].get(arm_b, {}))
                if out:
                    deltas.append(out[0])
            if deltas:
                by_policy[pol][reg] = float(np.mean(deltas))
                cells.append(f"{by_policy[pol][reg]:+11.4f}")
            else:
                cells.append("          -")
        print(f"{reg:>6d}  " + "  ".join(cells))

    # ---------------------------------------------------------- crossover
    print("\n" + "=" * 78)
    print("DOES THE SCARCITY CROSSOVER SURVIVE?")
    print("=" * 78)
    print("Prediction recorded before the run: absolute AUC falls under both")
    print("frozen policies. The question is whether the ORDERING is preserved.")
    print(f"\n{'policy':>10s} {'scarce':>10s} {'abundant':>10s} {'slope':>11s}  verdict")
    print("-" * 62)

    agree = []
    for pol in policies:
        per_n = by_policy[pol]
        if len(per_n) < 3:
            print(f"{pol:>10s}  (need >=3 shot levels)")
            continue
        ns = sorted(per_n)
        xs, ys = np.log2(ns), np.array([per_n[n] for n in ns])
        slope = float(np.polyfit(xs, ys, 1)[0])
        scarce = float(np.mean([per_n[n] for n in ns[:2]]))
        abundant = float(np.mean([per_n[n] for n in ns[-2:]]))
        ok = scarce > 0 > abundant and slope < 0
        agree.append((pol, ok))
        print(f"{pol:>10s} {scarce:>+10.4f} {abundant:>+10.4f} {slope:>+11.5f}  "
              f"{'crossover present' if ok else 'no crossover'}")

    frozen_ok = [ok for pol, ok in agree if pol in ("pca", "random")]
    if frozen_ok and all(frozen_ok):
        print("\n  The crossover survives with the projection frozen, under BOTH")
        print("  an optimal and a random compression. It is a property of the")
        print("  heads, not of a learned projection adapting to each of them.")
    elif frozen_ok and not any(frozen_ok):
        print("\n  The crossover does NOT survive freezing the projection. It was")
        print("  an artifact of a learned bottleneck adapting differently to")
        print("  different heads. Report this as the finding - it supersedes the")
        print("  earlier reading rather than qualifying it.")
    elif frozen_ok:
        print("\n  The frozen policies DISAGREE. Neither 'you picked a friendly")
        print("  projection' nor 'your projection is degraded' can be ruled out")
        print("  yet; report both and do not claim the ordering is head-driven.")

    n_pred = sum(1 for r in rows if r.get("predictions_file"))
    print(f"\n{n_pred}/{len(rows)} runs recorded per-sample predictions.")
    print("Confirmatory intervals: 04_statistical_analysis.py "
          "--experiment 12_bottleneck")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+", type=int, default=config.N_PER_CLASS)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--policies", nargs="+", default=POLICIES,
                   choices=POLICIES)
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS[:5])
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--metric", default="auc")
    args = p.parse_args()

    if args.summary_only:
        summarise(args.metric)
        return

    if args.quick:
        args.datasets = ["breastmnist", "pneumoniamnist"]
        args.regimes = [5, 100]
        args.seeds = config.ALL_SEEDS[:3]

    # Preflight: the capacity table is the argument for running this at all.
    print(f"Trainable capacity at d={args.dim}:")
    for pol in args.policies:
        for arm in args.arms:
            m = build_arm(arm, d=args.dim, num_classes=2, seed=42,
                          build_backbone=False, bottleneck_policy=pol)
            c = m.capacity_report()
            print(f"  {pol:>8s} {arm:24s} bneck={c['bottleneck']:>5d} "
                  f"head={c['head']:>4d} total={c['total']:>5d} "
                  f"head={100 * c['head_share']:.1f}%")

    total = (len(args.datasets) * len(args.regimes) * len(args.arms)
             * len(args.policies) * len(args.seeds))
    n_quantum = (len([a for a in args.arms if a in QUANTUM_ARMS])
                 * len(args.datasets) * len(args.regimes)
                 * len(args.policies) * len(args.seeds))
    print(f"\nBottleneck ablation | {total} cells ({n_quantum} quantum, "
          f"~2.5 min each) | device={config.DEVICE} sha={config.git_sha()[:8]}")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for n in args.regimes:
            for seed in args.seeds:
                for pol in args.policies:
                    for arm in args.arms:
                        done += 1
                        m = run_cell(ds, n, seed, arm, pol, dim=args.dim,
                                     force=args.force)
                        if m is None:
                            continue
                        eta = (time.time() - t0) / done * (total - done) / 3600
                        print(f"[{done}/{total}] {ds} n={n} s={seed} "
                              f"bn={pol:<8s} {arm:24s} auc={m['auc']:.4f} "
                              f"ETA {eta:.1f}h", flush=True)

    summarise(args.metric)


if __name__ == "__main__":
    main()
