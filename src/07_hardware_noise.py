"""
EXPERIMENT 7 - QUANTUM HARDWARE NOISE (shot noise and depolarizing).

WHY THIS IS A SEPARATE, QUANTUM-ONLY EXPERIMENT
------------------------------------------------
Experiment 3 injects noise on the IMAGES. Both arms see bit-identical corrupted
tensors, so it is a fair head-to-head and it belongs in the main results.

The noise here has no classical counterpart. Finite measurement sampling and
depolarizing decoherence are properties of a quantum processor; there is nothing
to compare a classical MLP against. Presenting these as a "quantum vs classical"
comparison would be dishonest. They answer a different and narrower question:

    WOULD THIS SURVIVE ON REAL HARDWARE?

That belongs in a feasibility / limitations section, and stating the distinction
explicitly is what keeps the main comparison clean.

Every QML reviewer asks this. A paper that reports only noise-free state-vector
simulation gets the standard objection that its numbers are over-optimistic.

WHAT IS SIMULATED
-----------------
SHOT NOISE. Real hardware estimates <X_i> from a finite number of measurements,
so each expectation value carries sampling error ~ 1/sqrt(shots). With 1024
shots the standard error is ~0.03 on a quantity bounded in [-1, 1] - not
negligible relative to the effect sizes in this study.

DEPOLARIZING NOISE. With probability p a qubit's state is replaced by the
maximally mixed state. This is the standard first-order model of decoherence and
gate infelicity. Applied after every variational block, so deeper circuits
accumulate more of it - which is the realistic behaviour, and the reason circuit
depth is not free on hardware.

Requires default.mixed (density-matrix simulation): a pure state-vector
simulator cannot represent a mixed state, so noise cannot be modelled on
default.qubit at all.

METHOD
------
Evaluation only. Heads are trained noise-free on cached features - which is what
would actually happen, since you train on a simulator and deploy on hardware -
then the SAME trained weights are evaluated through noisy circuits. Any
degradation is therefore attributable to the hardware model, not to a different
optimisation path.

COST
----
Shot-based and density-matrix simulation are both far slower than state-vector
evaluation, and the density matrix is 4^d rather than 2^d. Feasible at d=4;
d=8 is already 65,536 entries per sample. Defaults are set accordingly.

USAGE
-----
    python src/07_hardware_noise.py --quick
    python src/07_hardware_noise.py
    python src/07_hardware_noise.py --summary-only
"""
import argparse
import os
import sys
import time

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm                           # noqa: E402
from train.loop import train_model                              # noqa: E402
from train.metrics import compute_metrics                       # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features

EXPERIMENT = "07_hardware_noise"

SHOT_LEVELS = [None, 8192, 1024, 256, 64]        # None = exact expectation
DEPOLARIZING_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05]


def noisy_circuit(d, n_layers, n_uploads, shots=None, p_depol=0.0):
    """
    Rebuilds the trained circuit on a device that models hardware imperfection.

    default.mixed is required for depolarizing noise - a state-vector simulator
    cannot represent a mixed state. It is used for shot noise too so that the
    two sweeps share one code path and cannot diverge.
    """
    dev = qml.device("default.mixed", wires=d, shots=shots)
    layers_per_block = n_layers // n_uploads

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circuit(inputs, weights):
        for r in range(n_uploads):
            qml.AngleEmbedding(inputs, wires=range(d), rotation="Y")
            lo = r * layers_per_block
            qml.StronglyEntanglingLayers(weights[lo:lo + layers_per_block],
                                         wires=range(d))
            if p_depol > 0:
                # after every block, so depth costs fidelity - as on hardware
                for w in range(d):
                    qml.DepolarizingChannel(p_depol, wires=w)
        return [qml.expval(qml.PauliX(i)) for i in range(d)]

    return circuit


@torch.no_grad()
def evaluate_noisy(model, features, labels, num_classes, d, n_layers,
                   n_uploads, shots, p_depol, batch=64):
    """
    Runs the trained model with the quantum head replaced by a noisy circuit.
    The classical bottleneck and classifier are untouched - only the quantum
    evaluation changes, which is exactly the hardware substitution being modelled.
    """
    circuit = noisy_circuit(d, n_layers, n_uploads, shots=shots, p_depol=p_depol)
    weights = model.head.q_layer.weights.detach().cpu()

    probs = []
    for i in range(0, len(labels), batch):
        h = features[i:i + batch]
        z = model.latent(h).cpu()                       # bottleneck + tanh
        v = torch.stack([torch.as_tensor(circuit(z[j], weights), dtype=torch.float32)
                         for j in range(z.shape[0])])
        logits = model.classifier(v.to(model.classifier.weight.device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(probs)
    return compute_metrics(labels, probs.argmax(axis=1), probs, num_classes), probs


def run_cell(dataset, n_per_class, seed, arm="quantum_vqc", dim=4,
             n_shot_eval=None, force=False):
    keys = dict(dataset=dataset, regime=n_per_class, dim=dim, seed=seed, arm=arm)
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    C = num_classes_of(dataset)
    t0 = time.time()

    # --- train noise-free, exactly as one would before deploying ----------
    loaders, _ = get_cached_features(dataset, str(n_per_class), seed)
    model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                      seed=seed, build_backbone=False)
    out = train_model(model, loaders["train"], loaders["val"], loaders["test"],
                      num_classes=C, use_features=True, is_quantum=True,
                      verbose=False)
    best_state = out[2]
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model = model.to(config.DEVICE).eval()

    te = loaders["test"]
    feats, labels = te.feats, te.labels.cpu().numpy()
    if n_shot_eval:                                   # subsample: shots are slow
        idx = np.random.default_rng(seed).choice(len(labels),
                                                 min(n_shot_eval, len(labels)),
                                                 replace=False)
        feats, labels = feats[idx], labels[idx]

    head = model.head
    n_layers, n_uploads = head.n_layers, head.n_uploads

    shot_curve = {}
    for s in SHOT_LEVELS:
        m, _ = evaluate_noisy(model, feats, labels, C, dim, n_layers,
                              n_uploads, shots=s, p_depol=0.0)
        shot_curve["exact" if s is None else str(s)] = m

    depol_curve = {}
    for p in DEPOLARIZING_LEVELS:
        m, _ = evaluate_noisy(model, feats, labels, C, dim, n_layers,
                              n_uploads, shots=None, p_depol=p)
        depol_curve[f"{p:.3f}"] = m

    shards.write(EXPERIMENT,
                 {"shot_curve": shot_curve, "depol_curve": depol_curve,
                  "n_eval": int(len(labels)), "wall_time": time.time() - t0},
                 **keys)

    del model
    torch.cuda.empty_cache()
    return shot_curve, depol_curve


def summarise(metric="auc"):
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found.")
        return

    shot, depol = {}, {}
    for r in rows:
        k = r["keys"]
        cell = (k["dataset"], str(k["regime"]))
        for s, m in r["shot_curve"].items():
            shot.setdefault((cell, s), []).append(m.get(metric))
        for p, m in r["depol_curve"].items():
            depol.setdefault((cell, p), []).append(m.get(metric))

    cells = sorted({c for (c, _) in shot})

    print(f"\n=== SHOT NOISE: {metric.upper()} vs measurements per expectation ===")
    print("Real hardware estimates <X> from finite samples; error ~ 1/sqrt(shots).")
    cols = ["exact"] + [str(s) for s in SHOT_LEVELS if s]
    print(f"\n{'dataset':16s} {'n/cls':>6s} " + " ".join(f"{c:>9s}" for c in cols))
    print("-" * (24 + 10 * len(cols)))
    for cell in cells:
        vals = []
        for c in cols:
            v = [x for x in shot.get((cell, c), []) if x is not None]
            vals.append(f"{np.mean(v):9.4f}" if v else "        -")
        print(f"{cell[0]:16s} {cell[1]:>6s} " + " ".join(vals))

    print(f"\n=== DEPOLARIZING NOISE: {metric.upper()} vs error rate per qubit ===")
    print("Applied after every variational block, so depth accumulates error.")
    cols = [f"{p:.3f}" for p in DEPOLARIZING_LEVELS]
    print(f"\n{'dataset':16s} {'n/cls':>6s} " + " ".join(f"{c:>9s}" for c in cols))
    print("-" * (24 + 10 * len(cols)))
    for cell in cells:
        vals = []
        for c in cols:
            v = [x for x in depol.get((cell, c), []) if x is not None]
            vals.append(f"{np.mean(v):9.4f}" if v else "        -")
        print(f"{cell[0]:16s} {cell[1]:>6s} " + " ".join(vals))

    print(f"\n=== Retention (noisy / exact) ===")
    for cell in cells:
        ex = [x for x in shot.get((cell, "exact"), []) if x is not None]
        if not ex:
            continue
        base = np.mean(ex)
        s1k = [x for x in shot.get((cell, "1024"), []) if x is not None]
        d01 = [x for x in depol.get((cell, "0.010"), []) if x is not None]
        print(f"  {cell[0]:16s} n={cell[1]:>4s}  exact={base:.4f}  "
              f"1024 shots={np.mean(s1k)/base:.3f}x  "
              f"p=0.01 depol={np.mean(d01)/base:.3f}x"
              if s1k and d01 else f"  {cell[0]:16s} n={cell[1]:>4s} partial")

    print("\nFEASIBILITY SECTION ONLY. There is no classical counterpart to these")
    print("noise models, so none of this is a quantum-vs-classical comparison.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["breastmnist", "pneumoniamnist"])
    p.add_argument("--regimes", nargs="+", type=int, default=[5, 100])
    p.add_argument("--seeds", nargs="+", type=int, default=config.ALL_SEEDS[:5])
    p.add_argument("--arm", default="quantum_vqc")
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--n-eval", type=int, default=500,
                   help="test images per cell; shot simulation is slow")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--metric", default="auc")
    args = p.parse_args()

    if args.summary_only:
        summarise(args.metric)
        return

    if args.quick:
        args.datasets = ["breastmnist"]
        args.regimes = [100]
        args.seeds = config.ALL_SEEDS[:2]
        args.n_eval = 156

    total = len(args.datasets) * len(args.regimes) * len(args.seeds)
    print(f"Q-hardware noise | {total} cells | {len(SHOT_LEVELS)} shot levels "
          f"x {len(DEPOLARIZING_LEVELS)} depol levels | sha={config.git_sha()[:8]}")
    print("density-matrix simulation is slow; --n-eval subsamples the test set")

    done, t0 = 0, time.time()
    for ds in args.datasets:
        for n in args.regimes:
            for seed in args.seeds:
                done += 1
                out = run_cell(ds, n, seed, arm=args.arm, dim=args.dim,
                               n_shot_eval=args.n_eval, force=args.force)
                if out is None:
                    continue
                sc, dc = out
                eta = (time.time() - t0) / done * (total - done) / 3600
                print(f"[{done}/{total}] {ds} n={n} s={seed} "
                      f"exact={sc['exact']['auc']:.4f} "
                      f"1024sh={sc['1024']['auc']:.4f} "
                      f"p.01={dc['0.010']['auc']:.4f} ETA {eta:.1f}h", flush=True)

    summarise(args.metric)


if __name__ == "__main__":
    main()
