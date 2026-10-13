"""
E8 - DOES THE LEARNED PROJECTION SATURATE THE LATENT INPUTS, AND IS IT VQC-SPECIFIC?

    python src/17_latent_saturation.py --estimate       # runtime from the original records
    python src/17_latent_saturation.py --quick          # one cell, both arms, sanity check
    nohup python src/17_latent_saturation.py > logs_e8_saturation.txt 2>&1 &
    python src/17_latent_saturation.py --summary-only

STATUS: EXPLORATORY. Motivated by data, labelled so everywhere, outside the
confirmatory family, no correction, no verdict. Written 13 October 2026, BEFORE
any run of this script.

WHAT PROMPTED IT
----------------
Reproducing one confirmatory run for the function-class figure (quantum_vqc,
BloodMNIST, n = 5, seed 42) showed two of the four latent inputs pinned at the
tanh boundary for more than 99.8% of test images (interquartile range ~0.01, vs
0.6-0.8 for the other two). On those images the VQC effectively receives two
varying inputs, not four. One run cannot say whether this is general, whether
the matched classical head's projection does the same, or whether it depends on
the amount of data. This script asks exactly that and nothing more.

DESIGN - REPRODUCTION ONLY, NO NEW CONFIGURATION
------------------------------------------------
Every run here is a bit-identical re-run of an existing confirmatory run from
artifacts/shards/01_frozen_tuned, with every setting read from that run's own
record. Nothing is tuned or chosen here. The reproduced test AUC is compared
with the recorded one and the difference is stored; on the cluster GPU the
first reproduction matched exactly.

  arms      quantum_vqc, matched_param_fullrank (the primary pair)
  datasets  all four
  n         5 first, then 100 (so a partial sweep still answers the n = 5 case)
  seeds     the first 10 confirmatory seeds (config.CONFIRMATORY_SEEDS[:10])
  runs      2 x 4 x 2 x 10 = 160

MEASURES, DEFINED NOW
---------------------
On the test set, for each latent coordinate j of z_tilde = tanh(W h + b) * s:
  near_boundary_j = share of test images with |z_j| > 0.95 s
  iqr_j           = interquartile range of z_j over the test images
and per run
  n_active = number of coordinates with near_boundary_j < 0.5
(a coordinate pinned at the boundary for most images carries almost no
image-to-image variation into the head).

PREDICTIONS, WRITTEN BEFORE THE RUN (weak priors, reported whichever way)
--------------------------------------------------------------------------
P1  Saturation is NOT specific to the VQC: mean n_active differs between the two
    arms by less than one coordinate in every dataset x n cell. Basis: the
    projection and the tanh squashing are identical for both arms.
P2  Saturation is more common with less data: mean n_active is lower at n = 5
    than at n = 100, for both arms. Basis: with ten training images, separating
    them pushes projection weights large.
No significance test is attached to either; both are read off the summary.

OUTPUT
------
Shards in artifacts/shards/28_latent_saturation (keys: dataset, regime, seed,
arm; git SHA recorded), test latents in artifacts/latents/28_latent_saturation,
and reproduced test probabilities via shards.save_predictions. Resumable:
existing shards are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm                           # noqa: E402
from train.loop import train_model                              # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features

EXPERIMENT = "28_latent_saturation"
SOURCE = "01_frozen_tuned"
ARMS = ["quantum_vqc", "matched_param_fullrank"]
DATASETS = ["bloodmnist", "breastmnist", "pathmnist", "pneumoniamnist"]
REGIMES = ["5", "100"]
SEEDS = [str(s) for s in config.CONFIRMATORY_SEEDS[:10]]
BOUNDARY = 0.95
QUANTUM = {"quantum_vqc"}


def source_records():
    """{(dataset, regime, seed, arm): record} for the primary runs at d = 4."""
    d = os.path.join(config.SHARD_DIR, SOURCE)
    out = {}
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            r = json.load(f)
        k = {a: str(b) for a, b in r["keys"].items()}
        if k.get("arm") in ARMS and k.get("dim") == "4" and "bn" not in k:
            key = (k["dataset"], k["regime"], k["seed"], k["arm"])
            if key in out:
                raise RuntimeError(f"two source records for {key}")
            out[key] = r
    return out


def plan(quick=False):
    if quick:
        return [("bloodmnist", "5", SEEDS[0], a) for a in ARMS]
    return [(ds, n, s, a) for n in REGIMES for ds in DATASETS for s in SEEDS for a in ARMS]


def saturation(z, scale):
    near = (np.abs(z) > BOUNDARY * scale).mean(axis=0)
    q75, q25 = np.percentile(z, [75, 25], axis=0)
    return {"near_boundary": [float(x) for x in near],
            "iqr": [float(x) for x in (q75 - q25)],
            "std": [float(x) for x in z.std(axis=0)],
            "n_active": int((near < 0.5).sum())}


def run_one(rec):
    keys, cfg = rec["keys"], rec["config"]
    ds, regime, seed, arm = keys["dataset"], str(keys["regime"]), int(keys["seed"]), keys["arm"]
    t0 = time.time()
    config.set_determinism(seed)
    C = num_classes_of(ds)
    loaders, _ = get_cached_features(ds, regime, seed)
    # Exactly as 01's run_cell built it: optional axes only when the record's
    # keys show they were set explicitly.
    model = build_arm(arm, d=4, num_classes=C, n_layers=cfg["n_layers"], seed=seed,
                      build_backbone=False, use_tanh=True,
                      angle_scale=(cfg["angle_scale"] if "as" in keys else None),
                      bottleneck_policy="learned", head_rank=None)
    metrics, _, best_state, probs, labels = train_model(
        model, loaders["train"], loaders["val"], loaders["test"], num_classes=C,
        use_features=True, is_quantum=(arm in QUANTUM), verbose=False, return_probs=True,
        lr_head=cfg["lr_head"], lr_quantum=cfg["lr_quantum"])
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        z = model.latent(loaders["test"].feats.float()).cpu().numpy()
    pdiff = float("nan")
    pf = os.path.join(config.PREDICTION_DIR, SOURCE, rec.get("predictions_file", ""))
    if rec.get("predictions_file") and os.path.exists(pf):
        with np.load(pf) as zz:
            pdiff = float(np.abs(zz["probs"].astype(np.float64) - probs).max())
    scale = float(model.angle_scale)
    return {"z": z.astype(np.float32), "labels": labels, "probs": probs,
            "auc_recorded": float(rec["metrics"]["auc"]), "auc_reproduced": float(metrics["auc"]),
            "max_prob_diff": pdiff, "angle_scale": scale, "source_sha": rec["git_sha"],
            "stats": saturation(z, scale), "wall_time": time.time() - t0}


def estimate(recs, todo):
    """Projected runtime from the ORIGINAL runs' own epoch counts and epoch times,
    dominated by the largest dataset (PathMNIST), not extrapolated from a small one."""
    by_ds, missing = {}, 0
    for key in todo:
        r = recs.get(key)
        m = (r or {}).get("metrics") or {}
        if not r or m.get("epochs_run") is None or m.get("mean_epoch_time") is None:
            missing += 1
            continue
        by_ds[key[0]] = by_ds.get(key[0], 0.0) + m["epochs_run"] * m["mean_epoch_time"]
    total = sum(by_ds.values())
    print("Projected training time from the original runs' records (excludes the final "
          "test evaluation and feature loading):")
    for ds, sec in sorted(by_ds.items(), key=lambda x: -x[1]):
        print(f"  {ds:15s} {sec / 3600:6.2f} h")
    print(f"  total          {total / 3600:6.2f} h   ({missing} runs without timing)")
    print("Assumption: same GPU and code path as the originals. Treat as a lower bound.")


def summary():
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return
    by = {}
    for r in rows:
        k = r["keys"]
        by.setdefault((k["dataset"], str(k["regime"]), k["arm"]), []).append(r)
    exact = sum(1 for r in rows if abs(r["auc_reproduced"] - r["auc_recorded"]) < 1e-9)
    print(f"E8 latent saturation - EXPLORATORY | {len(rows)} runs | "
          f"{exact}/{len(rows)} reproduced the recorded AUC exactly")
    worst = max(abs(r["auc_reproduced"] - r["auc_recorded"]) for r in rows)
    print(f"largest |reproduced - recorded| AUC: {worst:.2e}\n")
    print(f"{'dataset':15s} {'n':>4s} {'arm':24s} {'runs':>4s} {'mean n_active':>13s} "
          f"{'share pinned, per coordinate (mean over runs)':>46s}")
    for (ds, n, arm), rs in sorted(by.items(), key=lambda x: (int(x[0][1]), x[0][0], x[0][2])):
        na = np.mean([r["stats"]["n_active"] for r in rs])
        nb = np.mean([r["stats"]["near_boundary"] for r in rs], axis=0)
        print(f"{ds:15s} {n:>4s} {arm:24s} {len(rs):4d} {na:13.2f}   "
              + " ".join(f"{x:5.2f}" for x in nb))

    print("\nP1 (not VQC-specific): VQC - matched mean n_active per cell")
    for ds in DATASETS:
        for n in REGIMES:
            a, b = by.get((ds, n, "quantum_vqc")), by.get((ds, n, "matched_param_fullrank"))
            if a and b:
                diff = np.mean([r["stats"]["n_active"] for r in a]) - \
                    np.mean([r["stats"]["n_active"] for r in b])
                print(f"  {ds:15s} n={n:>3s}  {diff:+.2f}  ({'within 1' if abs(diff) < 1 else 'NOT within 1'})")
    print("\nP2 (more saturation with less data): mean n_active at n=5 vs n=100")
    for arm in ARMS:
        for ds in DATASETS:
            a, b = by.get((ds, "5", arm)), by.get((ds, "100", arm))
            if a and b:
                x = np.mean([r["stats"]["n_active"] for r in a])
                y = np.mean([r["stats"]["n_active"] for r in b])
                print(f"  {arm:24s} {ds:15s} n=5 {x:.2f}  n=100 {y:.2f}  "
                      f"({'lower at n=5' if x < y else 'not lower at n=5'})")

    print("\nDescriptive only: per seed, VQC - matched AUC against VQC - matched n_active (n = 5)")
    for ds in DATASETS:
        pairs = {}
        for arm in ARMS:
            for r in by.get((ds, "5", arm), []):
                pairs.setdefault(str(r["keys"]["seed"]), {})[arm] = r
        both = [p for p in pairs.values() if len(p) == 2]
        if len(both) >= 3:
            da = np.array([p["quantum_vqc"]["auc_reproduced"] - p["matched_param_fullrank"]["auc_reproduced"]
                           for p in both])
            dn = np.array([p["quantum_vqc"]["stats"]["n_active"] - p["matched_param_fullrank"]["stats"]["n_active"]
                           for p in both])
            rho = (np.corrcoef(np.argsort(np.argsort(da)), np.argsort(np.argsort(dn)))[0, 1]
                   if da.std() > 0 and dn.std() > 0 else float("nan"))
            print(f"  {ds:15s} {len(both)} seeds  Spearman rho = {rho:+.2f}")
    print("\nEXPLORATORY. One reproduction per run; no test, no correction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summary()
        return
    recs = source_records()
    todo = plan(args.quick)
    missing = [k for k in todo if k not in recs]
    if missing:
        sys.exit(f"{len(missing)} planned runs have no source record, e.g. {missing[:3]}")
    if args.estimate:
        estimate(recs, todo)
        return
    print(f"E8 latent saturation | {len(todo)} runs | sha={config.git_sha()[:8]} | "
          f"device={config.DEVICE}", flush=True)
    t0 = time.time()
    for i, key in enumerate(todo, 1):
        ds, n, s, arm = key
        skeys = dict(dataset=ds, regime=n, seed=s, arm=arm)
        if not args.force and shards.exists(EXPERIMENT, **skeys):
            continue
        out = run_one(recs[key])
        os.makedirs(os.path.join(config.LATENT_DIR, EXPERIMENT), exist_ok=True)
        np.savez_compressed(os.path.join(config.LATENT_DIR, EXPERIMENT, shards.pred_tag(**skeys)),
                            z=out["z"], labels=out["labels"].astype(np.int16))
        pf = shards.save_predictions(EXPERIMENT, out["labels"], out["probs"], **skeys)
        shards.write(EXPERIMENT, {k: v for k, v in out.items() if k not in ("z", "labels", "probs")}
                     | {"predictions_file": pf}, **skeys)
        st = out["stats"]
        print(f"[{i}/{len(todo)}] {ds} n={n} s={s} {arm}: AUC {out['auc_reproduced']:.4f} "
              f"(diff {out['auc_reproduced'] - out['auc_recorded']:+.1e})  n_active {st['n_active']}  "
              f"pinned {[round(x, 2) for x in st['near_boundary']]}  {out['wall_time']:.0f}s  "
              f"elapsed {(time.time() - t0) / 3600:.2f}h", flush=True)
    summary()


if __name__ == "__main__":
    main()
