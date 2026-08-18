"""
EXPERIMENT 1 - HEAD COMPARISON, FROZEN AND ADAPTIVE ENCODER.

(The filename says "frozen ablation" for historical reasons; the script runs
both encoder regimes. --freeze-policies selects which.)

WHAT IT ISOLATES
----------------
With freeze_policy="all" the entire ResNet is immobilised, so every arm sees
identical, static ImageNet features. Whatever separates the arms is a property
of the head's function class alone - no confound from the backbone adapting
differently to different gradients.

With freeze_policy="layer3_only" the encoder can adapt, which is the end-to-end
setting. Running both from ONE script guarantees the two regimes cannot drift
apart, which is what happened when they lived in separate files.

WHY THE FROZEN PATH IS FAST
---------------------------
A frozen backbone in eval mode is a deterministic function: the same image
always produces the same 256-d vector (verified bit-exact). The backbone runs
ONCE per (dataset, regime, seed); the resulting vectors are cached and all arms
train on those. Mathematically identical, orders of magnitude cheaper.

This is also why augmentation is OFF here (config.AUGMENT_FROZEN = False):
augmentation would make features non-deterministic and caching invalid. It also
means the frozen-vs-adaptive comparison must run with augmentation off on BOTH
sides, or freezing and augmentation would vary together.

PER-SAMPLE PREDICTIONS
----------------------
Every run writes its test-set probabilities through shards.save_predictions(),
which derives the filename from the SAME keys that name the shard. 04 reads them
back through shards.load_predictions() using the keys stored in the shard, so
writer and reader cannot disagree.

They previously did disagree: 01 wrote one naming convention, 03 another, and 04
expected 03's - so 04 never found 01's files, silently fell back to seed-level
resampling, and printed results that looked fine. That fallback is exactly the
analysis the pre-registration forbids.

OPTIONAL AXES
-------------
--n-layers    circuit depth L (depth sweep: manifold dimension 3*L*d)
--no-tanh     feed raw z to the head; CLASSICAL ARMS ONLY (see registry.py)
--angle-scale the tanh scale; swept because it was never tuned
--lr-head     LR for non-quantum trainable parameters
--lr-quantum  LR for the VQC rotation angles

An axis enters the shard key when it is EXPLICITLY SET, not when it differs from
the current config value. Keying against a mutable constant was a latent bug: the
moment config.LR_HEAD changed to a tuned value, every old shard trained at the
old default (which carries no lr key) would have become a valid cache hit for the
new default and been silently reused as if it were tuned.

--experiment writes to a different shard namespace. 09_lr_selection.py uses it so
hyperparameter search never mixes with the results it informs; the confirmatory
sweep should use its own namespace too.

USAGE
    python src/01_frozen_backbone_ablation.py --summary-only
    python src/01_frozen_backbone_ablation.py --datasets breastmnist --dims 4 --seeds 42
    python src/01_frozen_backbone_ablation.py --n-layers 4 --arms quantum_vqc
    python src/01_frozen_backbone_ablation.py --no-tanh --arms linear mlp matched_param_fullrank
    python src/01_frozen_backbone_ablation.py --confirmatory --use-tuned-lr \
        --experiment 01_frozen_tuned
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                    # noqa: E402
import shards                                                    # noqa: E402
from data.medmnist_loader import get_loaders, num_classes_of     # noqa: E402
from models.backbone import TruncatedResNet18                    # noqa: E402
from models.registry import build_arm, QUANTUM_ARMS              # noqa: E402
from train.loop import train_model                               # noqa: E402
from train.metrics import compute_metrics                        # noqa: E402

EXPERIMENT = "01_frozen"

# Verified values (there is no 'frozen'):
#   'all'         -> backbone fully frozen, 1,038 trainable params
#   'layer3_only' -> 2,100,750 trainable params
FROZEN = "all"
ADAPTIVE = "layer3_only"

# The diagnostic contrast: capacity floor, function-class control, treatment.
DIAGNOSTIC_ARMS = ["linear", "fourier_rff", "quantum_vqc"]

LR_SELECTION_FILE = os.path.join(config.ARTIFACT_ROOT, "lr_selection.json")


# ------------------------------------------------------------------ features
class FeatureBatches:
    """Mirrors the GPUBatches interface, but yields cached 256-d vectors."""

    def __init__(self, feats, labels, batch_size, shuffle, seed):
        self.feats, self.labels = feats, labels
        self.batch_size, self.shuffle, self._seed = batch_size, shuffle, seed
        self.epoch = 0

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self._seed * 100_003 + self.epoch)
        n = len(self.labels)
        order = (torch.randperm(n, generator=g).to(self.feats.device)
                 if self.shuffle else torch.arange(n, device=self.feats.device))
        for i in range(0, n, self.batch_size):
            sel = order[i:i + self.batch_size]
            yield self.feats[sel], self.labels[sel]
        self.epoch += 1


@torch.no_grad()
def _extract(backbone, loader):
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x))
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def get_cached_features(dataset, regime, seed, force=False):
    """Run the frozen backbone once; reuse for every arm and dimension."""
    tag = f"{dataset}__r{regime}__s{seed}.pt"
    path = os.path.join(config.FEATURE_CACHE, tag)

    if os.path.exists(path) and not force:
        blob = torch.load(path, map_location=config.DEVICE, weights_only=True)
    else:
        full = (regime == "full")
        train, val, test, meta = get_loaders(
            dataset,
            n_per_class=None if full else int(regime),
            seed=seed,
            augment=config.AUGMENT_FROZEN,      # must be False for caching
            full_data=full)
        backbone = TruncatedResNet18(freeze_policy=FROZEN).to(config.DEVICE)
        backbone.eval()
        blob = {"meta": meta}
        for name, ld in (("train", train), ("val", val), ("test", test)):
            f, y = _extract(backbone, ld)
            blob[f"{name}_x"], blob[f"{name}_y"] = f.cpu(), y.cpu()
        torch.save(blob, path)
        del backbone
        torch.cuda.empty_cache()

    dev = config.DEVICE
    loaders = {}
    for name, shuffle in (("train", True), ("val", False), ("test", False)):
        loaders[name] = FeatureBatches(
            blob[f"{name}_x"].to(dev), blob[f"{name}_y"].to(dev),
            config.BATCH_SIZE, shuffle, seed)
    return loaders, blob["meta"]


# ------------------------------------------------------------------ pca+svm
def run_pca_svm(blob_loaders, num_classes, dim, seed):
    """Non-neural reference. Same cached features, so essentially free."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    tr, te = blob_loaders["train"], blob_loaders["test"]
    Xtr, ytr = tr.feats.cpu().numpy(), tr.labels.cpu().numpy()
    Xte, yte = te.feats.cpu().numpy(), te.labels.cpu().numpy()

    if len(np.unique(ytr)) < 2:
        return None, None, None

    sc = StandardScaler().fit(Xtr)
    n_comp = min(dim, Xtr.shape[0], Xtr.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(sc.transform(Xtr))
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True,
              random_state=seed).fit(pca.transform(sc.transform(Xtr)), ytr)

    P = svm.predict_proba(pca.transform(sc.transform(Xte)))
    m = compute_metrics(yte, P.argmax(1), P, num_classes)
    m["variance_retained"] = float(pca.explained_variance_ratio_.sum())
    m["n_components"] = int(n_comp)
    return m, P, yte


# ------------------------------------------------------------------ keys
def _shard_keys(dataset, regime, dim, seed, arm, freeze_policy, augment,
                n_layers, use_tanh, angle_scale, lr_head, lr_quantum):
    """
    Optional axes enter the key when EXPLICITLY SET (not None / not default
    behaviour), so shards produced before those axes existed remain addressable,
    and no later change to a config constant can turn an old shard into a false
    cache hit for a new setting.

    lr_quantum is keyed only for quantum arms - classical arms have no quantum
    parameter group, so keying it there would fabricate distinct names for runs
    that are byte-identical.
    """
    keys = dict(dataset=dataset, regime=regime, dim=dim, seed=seed, arm=arm,
                fp=freeze_policy, aug=int(augment))
    if n_layers is not None and n_layers != config.VQC_LAYERS:
        keys["L"] = n_layers
    if not use_tanh:
        keys["notanh"] = 1
    if angle_scale is not None:
        keys["as"] = f"{float(angle_scale):.4f}"
    if lr_head is not None:
        keys["lrh"] = f"{float(lr_head):.0e}"
    if lr_quantum is not None and arm in QUANTUM_ARMS:
        keys["lrq"] = f"{float(lr_quantum):.0e}"
    return keys


# ------------------------------------------------------------------ cell
def run_cell(dataset, regime, dim, seed, arm, freeze_policy=FROZEN,
             augment=False, force=False, n_layers=None, use_tanh=True,
             angle_scale=None, save_predictions=True,
             lr_head=None, lr_quantum=None, experiment=EXPERIMENT):
    """
    freeze_policy=FROZEN   -> cached features, no backbone constructed at all
    freeze_policy=ADAPTIVE -> full end-to-end training through layer3

    augment defaults to False on BOTH sides. Feature caching requires
    deterministic features, so the frozen arm cannot augment; if the adaptive arm
    did, freezing and augmentation would vary together and the encoder-adaptation
    result would be uninterpretable.
    """
    keys = _shard_keys(dataset, regime, dim, seed, arm, freeze_policy, augment,
                       n_layers, use_tanh, angle_scale, lr_head, lr_quantum)
    if not force and shards.exists(experiment, **keys):
        return None

    effective_layers = config.VQC_LAYERS if n_layers is None else n_layers
    config.set_determinism(seed)
    C = num_classes_of(dataset)
    cached = (freeze_policy == FROZEN and not augment)

    if cached:
        loaders, meta = get_cached_features(dataset, regime, seed)
        train, val, test = loaders["train"], loaders["val"], loaders["test"]

        if arm == "pca_svm":
            metrics, probs, labels = run_pca_svm(loaders, C, dim, seed)
            if metrics is None:
                return None
            payload = {"metrics": metrics, "meta": meta}
            if save_predictions:
                payload["predictions_file"] = shards.save_predictions(
                    experiment, labels, probs, **keys)
            shards.write(experiment, payload, **keys)
            return metrics

        # build_backbone=False: the head trains on cached features, so no ResNet
        # is constructed at all.
        model = build_arm(arm, d=dim, num_classes=C, n_layers=effective_layers,
                          seed=seed, build_backbone=False,
                          use_tanh=use_tanh, angle_scale=angle_scale)
    else:
        if arm == "pca_svm":
            return None          # only defined on static frozen features
        full = (regime == "full")
        train, val, test, meta = get_loaders(
            dataset, n_per_class=None if full else int(regime),
            seed=seed, augment=augment, full_data=full)
        model = build_arm(arm, d=dim, num_classes=C, n_layers=effective_layers,
                          seed=seed, freeze_policy=freeze_policy,
                          use_tanh=use_tanh, angle_scale=angle_scale)

    out = train_model(
        model, train, val, test,
        num_classes=C, use_features=cached,
        is_quantum=(arm in QUANTUM_ARMS), verbose=False,
        return_probs=save_predictions,
        lr_head=lr_head, lr_quantum=lr_quantum)

    if save_predictions:
        metrics, history, _, probs, labels = out
    else:
        metrics, history, _ = out
        probs = labels = None

    payload = {
        "metrics": metrics,
        "meta": meta,
        # The REALISED configuration, recorded regardless of whether it entered
        # the key. This is what the manuscript's settings table is built from.
        "config": {"n_layers": effective_layers,
                   "use_tanh": bool(use_tanh),
                   "angle_scale": float(model.angle_scale),
                   "lr_head": metrics.get("lr_head"),
                   "lr_quantum": metrics.get("lr_quantum")},
        "history": {k: history[k] for k in
                    ("train_f1", "val_f1", "val_auc", "val_ece", "val_prob_std",
                     "pre_clip_grad_norm", "quantum_grad_var")},
    }
    if save_predictions and probs is not None:
        payload["predictions_file"] = shards.save_predictions(
            experiment, labels, probs, **keys)

    shards.write(experiment, payload, **keys)
    del model
    torch.cuda.empty_cache()
    return metrics


# ------------------------------------------------------------------ summary
def _paired_delta(a_by_seed, b_by_seed):
    """
    Mean paired difference over seeds present in BOTH arms, with a normal-
    approximation 95% interval.

    Pairing on seed matters: both arms saw identical splits and identical
    initialisation seeds, so seed-level variance largely cancels.

    DIAGNOSTIC ONLY. It ignores test-set sampling variance. Confirmatory
    intervals come from the nested bootstrap in 04_statistical_analysis.py.
    """
    common = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in common
                  if a_by_seed[s] is not None and b_by_seed[s] is not None])
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    mean = float(d.mean())
    half = 1.96 * float(d.std(ddof=1) / np.sqrt(len(d)))
    return mean, mean - half, mean + half, len(d)


def _contrast_table(tbl, arm_a, arm_b, title, note, metric):
    print(f"\n=== {title}: {arm_a} - {arm_b} ({metric}) ===")
    print(note)
    print(f"{'dataset':15s} {'reg':>5s} {'d':>3s} {'encoder':>9s} {'delta':>9s} "
          f"{'95% CI':>21s} {'n':>4s}  verdict")
    print("-" * 100)
    any_row = False
    for cell in sorted(tbl):
        a = tbl[cell].get(arm_a, {})
        b = tbl[cell].get(arm_b, {})
        if not a or not b:
            continue
        m, lo, hi, n = _paired_delta(a, b)
        if np.isnan(m):
            continue
        any_row = True
        verdict = (f"{arm_a} better" if lo > 0 else
                   f"{arm_b} better" if hi < 0 else "no difference")
        enc = "frozen" if cell[3] == FROZEN else "adaptive"
        print(f"{cell[0]:15s} {cell[1]:>5s} {cell[2]:>3d} {enc:>9s} {m:+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {n:>4d}  {verdict}")
    if not any_row:
        print("  (no cells with both arms present)")


def summarise(metric="auc", experiment=EXPERIMENT):
    rows = shards.load_all(experiment)
    if not rows:
        print(f"No shards found for '{experiment}'.")
        return

    tbl, n_preds, n_found = {}, 0, 0
    for r in rows:
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]), k["dim"], k.get("fp", FROZEN))
        tbl.setdefault(cell, {}).setdefault(k["arm"], {})[k["seed"]] = \
            r["metrics"].get(metric)
        if r.get("predictions_file"):
            n_preds += 1
            # Verify the file is actually retrievable under the shard's own keys,
            # rather than trusting that a name was recorded.
            if shards.pred_exists(experiment, **k):
                n_found += 1

    present = [a for a in (["pca_svm"] + config.ARMS)
               if any(a in v for v in tbl.values())]

    print(f"\n=== {metric.upper()} by cell (mean +- sd over seeds) ===")
    print(f"{'dataset':15s} {'reg':>5s} {'d':>3s} {'encoder':>9s} " +
          " ".join(f"{a[:12]:>14s}" for a in present))
    print("-" * (36 + 15 * len(present)))
    for (ds, reg, d, fp) in sorted(tbl):
        cells = []
        for a in present:
            v = [x for x in tbl[(ds, reg, d, fp)].get(a, {}).values() if x is not None]
            cells.append(f"{np.mean(v):.4f}+-{np.std(v):.3f}" if v else "       -      ")
        enc = "frozen" if fp == FROZEN else "adaptive"
        print(f"{ds:15s} {reg:>5s} {d:>3d} {enc:>9s} " +
              " ".join(f"{c:>14s}" for c in cells))

    # Contrasts are read from config, not hardcoded, so the labels can never
    # drift from what the analysis plan declares.
    _contrast_table(tbl, *config.PRIMARY_COMPARISON,
                    "PRIMARY (Q1, efficiency at equal parameters)",
                    "Both arms have 24 head parameters and full rank at d=4.",
                    metric)
    _contrast_table(tbl, *config.SECONDARY_COMPARISON,
                    "SECONDARY (Q2, dequantization)",
                    "Basis-matched, NOT parameter-matched (324 vs 24). Asks whether\n"
                    "the VQC exploits its own function class as well as a direct fit.",
                    metric)
    if hasattr(config, "DIAGNOSTIC_COMPARISON"):
        _contrast_table(tbl, *config.DIAGNOSTIC_COMPARISON,
                        "DIAGNOSTIC (rank-limited control)",
                        "matched_param is rank-limited to width 3; superseded by\n"
                        "matched_param_fullrank. Retained for reproducibility only.",
                        metric)

    # ---- Q3: does the encoder absorb the bottleneck?
    print(f"\n=== Q3: frozen - adaptive, per arm ({metric}) ===")
    print("The premise check showed compression is nearly free WITH an adaptive")
    print("encoder. A large negative delta here means the encoder was absorbing")
    print("the constraint - the reframed 'Latent Reshaping' claim.")
    print(f"{'dataset':15s} {'reg':>5s} {'d':>3s} {'arm':>24s} {'delta':>9s} {'n':>4s}")
    print("-" * 72)
    for (ds, reg, d, fp) in sorted(tbl):
        if fp != FROZEN or (ds, reg, d, ADAPTIVE) not in tbl:
            continue
        for a in present:
            fz = tbl[(ds, reg, d, FROZEN)].get(a, {})
            ad = tbl[(ds, reg, d, ADAPTIVE)].get(a, {})
            if not fz or not ad:
                continue
            m, _, _, n = _paired_delta(fz, ad)
            if not np.isnan(m):
                print(f"{ds:15s} {reg:>5s} {d:>3d} {a:>24s} {m:+9.4f} {n:>4d}")

    print(f"\n{n_preds}/{len(rows)} runs recorded predictions; "
          f"{n_found} are retrievable on disk.")
    if n_found < len(rows):
        print("Runs without retrievable predictions can only support seed-level")
        print("resampling, NOT the pre-registered nested bootstrap.")
    print("\nDIAGNOSTIC OUTPUT - every interval above is a normal approximation")
    print("over seeds. Confirmatory statistics come from 04_statistical_analysis.py.")


# ------------------------------------------------------------------ driver
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostic", action="store_true",
                   help="4 datasets x n{5,10,20,50,100} x d=4 x "
                        "{frozen,adaptive} x 3 arms x 10 seeds")
    p.add_argument("--confirmatory", action="store_true",
                   help="primary comparison, frozen, d=4, CONFIRMATORY_SEEDS (40)")
    p.add_argument("--datasets", nargs="+", default=config.DATASETS)
    p.add_argument("--regimes", nargs="+",
                   default=[str(n) for n in config.N_PER_CLASS])
    p.add_argument("--dims", nargs="+", type=int, default=config.BOTTLENECKS)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--freeze-policies", nargs="+", default=[FROZEN])
    p.add_argument("--augment", action="store_true",
                   help="off by default so freezing is the only difference (Q3)")
    p.add_argument("--n-layers", type=int, default=None,
                   help="VQC depth L; manifold dimension is 3*L*d (depth sweep)")
    p.add_argument("--no-tanh", action="store_true",
                   help="feed raw z to the head; CLASSICAL ARMS ONLY")
    p.add_argument("--angle-scale", type=float, default=None,
                   help="tanh scale; default config.ANGLE_SCALE")
    p.add_argument("--lr-head", type=float, default=None,
                   help="LR for non-quantum trainable parameters")
    p.add_argument("--lr-quantum", type=float, default=None,
                   help="LR for the VQC rotation angles")
    p.add_argument("--use-tuned-lr", action="store_true",
                   help="load per-arm LRs written by 09_lr_selection.py")
    p.add_argument("--experiment", default=EXPERIMENT,
                   help="shard namespace; use a fresh one for tuned runs")
    p.add_argument("--no-predictions", action="store_true",
                   help="skip saving per-sample probabilities (not recommended)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--metric", default="auc")
    args = p.parse_args()

    if args.summary_only:
        summarise(args.metric, experiment=args.experiment)
        return

    if args.diagnostic:
        args.datasets = config.DATASETS
        args.regimes = [str(n) for n in config.N_PER_CLASS]
        args.dims = [4]
        args.arms = DIAGNOSTIC_ARMS
        args.freeze_policies = [FROZEN, ADAPTIVE]
        args.augment = False
        args.seeds = args.seeds or config.ALL_SEEDS

    if args.confirmatory:
        args.datasets = config.DATASETS
        args.regimes = [str(n) for n in config.N_PER_CLASS]
        args.dims = [4]
        args.arms = list(config.PRIMARY_COMPARISON)
        args.freeze_policies = [FROZEN]
        args.augment = False
        args.seeds = args.seeds or config.CONFIRMATORY_SEEDS

    use_tanh = not args.no_tanh
    save_preds = not args.no_predictions
    arms = args.arms or (["pca_svm"] + config.arms_for(args.dims[0]))

    # Per-arm tuned learning rates. Loaded from disk rather than hardcoded so the
    # selection stays auditable and reversible.
    tuned = {}
    if args.use_tuned_lr:
        if not os.path.exists(LR_SELECTION_FILE):
            print(f"--use-tuned-lr: {LR_SELECTION_FILE} not found. "
                  f"Run 09_lr_selection.py first (it writes this file).")
            return
        with open(LR_SELECTION_FILE) as f:
            blob = json.load(f)
        tuned = {a: float(v) for a, v in blob["selected"].items()}
        print(f"using tuned LRs from {blob.get('timestamp', '?')}: {tuned}")
        if args.experiment == EXPERIMENT:
            print("WARNING: writing tuned runs into the default namespace. Use "
                  "--experiment 01_frozen_tuned to keep them separate.")

    # The no-tanh ablation is mathematically invalid for quantum arms: RY is
    # 2*pi-periodic, so unbounded z destroys injectivity. Drop them loudly rather
    # than letting build_arm raise 900 runs in.
    if not use_tanh:
        dropped = [a for a in arms if a in QUANTUM_ARMS]
        if dropped:
            print(f"--no-tanh: dropping quantum arms {dropped} "
                  f"(RY encoding is 2*pi-periodic; unbounded z is invalid)")
            arms = [a for a in arms if a not in QUANTUM_ARMS]
        if not arms:
            print("nothing left to run")
            return

    # Preflight: construct one model of every kind now rather than discovering a
    # broken combination 900 runs in.
    preflight_layers = config.VQC_LAYERS if args.n_layers is None else args.n_layers
    for arm in arms:
        if arm == "pca_svm":
            continue
        build_arm(arm, d=args.dims[0], num_classes=2, seed=42,
                  build_backbone=False, n_layers=preflight_layers,
                  use_tanh=use_tanh, angle_scale=args.angle_scale)
        if ADAPTIVE in args.freeze_policies:
            build_arm(arm, d=args.dims[0], num_classes=2, seed=42,
                      freeze_policy=ADAPTIVE, n_layers=preflight_layers,
                      use_tanh=use_tanh, angle_scale=args.angle_scale)

    total = sum(len(args.seeds or config.seeds_for(dim)) * len(arms)
                for _ in args.datasets for _ in args.regimes
                for dim in args.dims for _ in args.freeze_policies)

    print(f"Experiment 1 | {total} cells | ns={args.experiment} | "
          f"device={config.DEVICE} | sha={config.git_sha()[:8]}")
    print(f"  augment={args.augment} L={preflight_layers} use_tanh={use_tanh} "
          f"angle_scale={args.angle_scale or float(config.ANGLE_SCALE):.4f} "
          f"save_predictions={save_preds}")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for regime in args.regimes:
            for fp in args.freeze_policies:
                for dim in args.dims:
                    seeds = args.seeds or config.seeds_for(dim)
                    for seed in seeds:
                        for arm in arms:
                            done += 1
                            lrh = tuned.get(arm, args.lr_head)
                            lrq = tuned.get(arm, args.lr_quantum)
                            m = run_cell(ds, regime, dim, seed, arm,
                                         freeze_policy=fp, augment=args.augment,
                                         force=args.force, n_layers=args.n_layers,
                                         use_tanh=use_tanh,
                                         angle_scale=args.angle_scale,
                                         save_predictions=save_preds,
                                         lr_head=lrh, lr_quantum=lrq,
                                         experiment=args.experiment)
                            if m is None:
                                continue
                            eta = (time.time() - t0) / done * (total - done) / 3600
                            print(f"[{done}/{total}] {ds} r={regime} d={dim} "
                                  f"{'froz' if fp == FROZEN else 'adap'} s={seed} "
                                  f"{arm:24s} auc={m['auc']:.4f} "
                                  f"f1={m['macro_f1']:.4f} ETA {eta:.1f}h",
                                  flush=True)

    summarise(args.metric, experiment=args.experiment)


if __name__ == "__main__":
    main()
