"""
REPRODUCE ONE CONFIRMATORY RUN, SO THE FUNCTION-CLASS FIGURE SHOWS A TRAINED MODEL.

    python src/16_reproduce_for_figure.py

WHY THIS EXISTS
---------------
The paper's central theoretical result is that the VQC head's output is exactly
a classical sum of 3^d sines and cosines. The figure that shows it must be drawn
from a model that was actually trained in this study, not from circuit
parameters chosen for the picture. The confirmatory runs saved predictions and
metrics but not weights (best_state was discarded to keep the sweep small), so
this script retrains ONE run, identically configured, and keeps its weights.

THE RUN, FIXED BEFORE THIS SCRIPT WAS FIRST EXECUTED (13 October 2026)
----------------------------------------------------------------------
quantum_vqc, BloodMNIST, n = 5 labelled images per class, seed 42, d = 4,
frozen backbone, learned projection - the first confirmatory seed in the cell
where the exploratory per-dataset advantage is largest. Every setting (learning
rates, depth, tanh, angle scale) is read from that run's own record in
artifacts/shards/01_frozen_tuned, not restated here.

VERIFICATION, PRINTED EVERY TIME
--------------------------------
The reproduced test AUC and per-image probabilities are compared with the
recorded run. The original ran on a GPU; this may run on a CPU, where floating-
point order differs, so exact equality is not expected. The differences are
written into the output file and must be quoted in the figure caption. If the
AUC differs by more than 0.01 the script stops: the weights would then not be
those of the reported run in any useful sense.

WHAT IS SAVED (artifacts/reproduced/function_class.npz)
-------------------------------------------------------
The trained head evaluated by PennyLane itself, in float64:
  - on a 61 x 61 grid over EVERY pair of latent coordinates (i, j) in the
    encoder's range (-pi/2, pi/2), the other two held at the median of the real
    test latents. The plotting script shows the pair along which the real test
    latents spread most (largest product of interquartile ranges), so the
    surface is drawn where the data actually are. (Added 13 Oct: in this run two
    of the four latent coordinates sit near the tanh boundary for most test
    images, so a fixed (z1, z2) slice showed the surface mostly where there are
    no data.)
  - at 2,000 random latents in the same range, from which the plotting script
    fits the 81-term classical Fourier series;
  - the real test latents and labels, so the figure can show where the data lie.

Nothing here writes into any experiment namespace.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
from data.medmnist_loader import num_classes_of                 # noqa: E402
from models.registry import build_arm                           # noqa: E402
from train.loop import train_model                              # noqa: E402

_exp1 = __import__("01_frozen_backbone_ablation")
get_cached_features = _exp1.get_cached_features

CELL = {"dataset": "bloodmnist", "regime": "5", "seed": "42", "arm": "quantum_vqc", "dim": "4"}
GRID = 61
N_FIT = 2000
MAX_AUC_DIFF = 0.01


def find_record():
    d = os.path.join(config.SHARD_DIR, "01_frozen_tuned")
    hits = []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            r = json.load(f)
        k = {a: str(b) for a, b in r["keys"].items()}
        if all(k.get(a) == b for a, b in CELL.items()) and "bn" not in k:
            hits.append(r)
    if len(hits) != 1:
        sys.exit(f"expected exactly one recorded run for {CELL}, found {len(hits)}")
    return hits[0]


def main():
    rec = find_record()
    cfg, keys = rec["config"], rec["keys"]
    seed, ds, regime = int(keys["seed"]), keys["dataset"], str(keys["regime"])
    print(f"reproducing {CELL} | recorded on commit {rec['git_sha'][:8]} at {rec['timestamp']}")
    print(f"  settings from the record: {cfg}")
    print(f"  device here: {config.DEVICE}")

    config.set_determinism(seed)
    C = num_classes_of(ds)
    loaders, _ = get_cached_features(ds, regime, seed)
    # Passed exactly as 01's run_cell passed them: optional axes only when the
    # record's keys show they were set explicitly.
    model = build_arm("quantum_vqc", d=4, num_classes=C, n_layers=cfg["n_layers"],
                      seed=seed, build_backbone=False, use_tanh=True,
                      angle_scale=(cfg["angle_scale"] if "as" in keys else None),
                      bottleneck_policy="learned", head_rank=None)
    out = train_model(model, loaders["train"], loaders["val"], loaders["test"],
                      num_classes=C, use_features=True, is_quantum=True, verbose=False,
                      return_probs=True, lr_head=cfg["lr_head"], lr_quantum=cfg["lr_quantum"])
    metrics, _, best_state, probs, labels = out
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()

    # ---- verification against the recorded run
    auc_rec, auc_new = rec["metrics"]["auc"], metrics["auc"]
    pdiff = float("nan")
    pf = os.path.join(config.PREDICTION_DIR, "01_frozen_tuned", rec["predictions_file"])
    if os.path.exists(pf):
        with np.load(pf) as z:
            pdiff = float(np.abs(z["probs"].astype(np.float64) - probs).max())
    print(f"  test AUC recorded {auc_rec:.6f} | reproduced {auc_new:.6f} | "
          f"diff {auc_new - auc_rec:+.2e}")
    print(f"  max |probability difference| over the test set: {pdiff:.2e}")
    if not abs(auc_new - auc_rec) <= MAX_AUC_DIFF:
        sys.exit(f"reproduced AUC differs by more than {MAX_AUC_DIFF}; not saving weights.")

    # ---- the trained head, evaluated by PennyLane in float64
    head = model.head.double()
    feats = loaders["test"].feats
    with torch.no_grad():
        z_test = model.latent(feats.float()).double().cpu().numpy()
        lim = float(model.angle_scale) * (1 - 1e-6)
        g = np.linspace(-lim, lim, GRID)
        G1, G2 = np.meshgrid(g, g)
        fixed = np.median(z_test[:, 2:], axis=0)
        Zg = np.stack([G1.ravel(), G2.ravel(),
                       np.full(G1.size, fixed[0]), np.full(G1.size, fixed[1])], 1)
        dev = next(head.parameters()).device
        out_grid = head(torch.from_numpy(Zg).to(dev)).cpu().numpy().reshape(GRID, GRID, -1)
        med = np.median(z_test, axis=0)
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        pair_grids = []
        for i, j in pairs:
            Zp = np.tile(med, (G1.size, 1))
            Zp[:, i], Zp[:, j] = G1.ravel(), G2.ravel()
            pair_grids.append(head(torch.from_numpy(Zp).to(dev)).cpu().numpy().reshape(GRID, GRID, -1))
        pair_grids = np.stack(pair_grids)
        rng = np.random.default_rng(0)
        Zf = rng.uniform(-lim, lim, (N_FIT, 4))
        out_fit = head(torch.from_numpy(Zf).to(dev)).cpu().numpy()

    dest = os.path.join(config.ARTIFACT_ROOT, "reproduced")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "function_class.npz")
    np.savez_compressed(
        path, grid_z1=G1, grid_z2=G2, grid_out=out_grid, fixed_z34=fixed,
        pairs=np.array(pairs), pair_grids=pair_grids, median_z=med,
        fit_z=Zf, fit_out=out_fit, test_z=z_test, test_y=labels.astype(np.int16),
        auc_recorded=auc_rec, auc_reproduced=auc_new, max_prob_diff=pdiff,
        angle_scale=float(model.angle_scale), cell=json.dumps(CELL),
        record_sha=rec["git_sha"], device=str(config.DEVICE),
        head_weights=head.q_layer.weights.detach().cpu().numpy())
    print(f"written: {path}")


if __name__ == "__main__":
    main()
