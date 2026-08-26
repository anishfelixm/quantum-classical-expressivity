"""
EXPERIMENT 11 - FREEZING AND GRADIENT-FLOW VERIFICATION.

WHAT THIS EXISTS TO PROVE
--------------------------
The study makes two structural claims that were, until this script, ASSERTED
rather than measured:

    PROOF A.  With freeze_policy="all" the backbone is genuinely untouched.
              Setting requires_grad=False and calling set_bn_eval() is HOW the
              backbone is frozen; it is not EVIDENCE that it stayed frozen. If a
              future edit dropped the set_bn_eval() call, BatchNorm running
              statistics would drift under the heads, every "frozen" result would
              be quietly contaminated, and nothing in the pipeline would notice.

    PROOF B.  With freeze_policy="layer3_only" gradients flow back into the
              encoder FROM EVERY HEAD, including the quantum one.
              "backprop through a PennyLane TorchLayer propagates input
              gradients" is a statement about the library. A reviewer asking
              "does the VQC actually train your encoder?" wants a number
              measured on THIS model, on THIS data.

Both are cheap. Both close objections that are currently unanswerable.

WHAT IS MEASURED
----------------
PROOF A - bit-exact comparison.
    Every backbone PARAMETER and every backbone BUFFER (BatchNorm running_mean,
    running_var, num_batches_tracked) is snapshotted before training and
    compared with torch.equal afterwards. Buffers matter more than parameters
    here: parameters are protected by requires_grad=False, but running
    statistics update on the FORWARD pass and are immune to that flag. eval()
    mode is the only thing stopping them, which is exactly the fragile part.

    A NEGATIVE CONTROL runs the same model without set_bn_eval(). Its buffers
    MUST change - otherwise the test proves nothing, because a comparison that
    can never fail is not evidence.

PROOF B - three independent signals, per head.
    1. gradient reaches the backbone: mean per-epoch L2 norm > 0
    2. the backbone MOVES: ||W_after - W_before|| over layer3 > 0
    3. gradient reaches the head's INPUT: d(loss)/d(z) is non-zero, which for
       the quantum arm is the specific thing in doubt

    Signal 3 is measured directly rather than inferred, because it is the one a
    sceptical reader will not take on trust: it requires the derivative to pass
    through the simulated circuit.

COST
----
Minutes. A handful of short runs on one small dataset - this verifies plumbing,
not performance, so it does not need the full grid.

USAGE
-----
    python src/11_flow_verification.py
    python src/11_flow_verification.py --arms quantum_vqc linear
    python src/11_flow_verification.py --summary-only
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import shards                                                   # noqa: E402
from data.medmnist_loader import get_loaders, num_classes_of    # noqa: E402
from models.registry import build_arm, QUANTUM_ARMS             # noqa: E402
from train.loop import train_model, module_grad_norms           # noqa: E402

EXPERIMENT = "11_flow"

FROZEN = "all"
ADAPTIVE = "layer3_only"

# One arm of each kind. The quantum arms are the ones actually in question; the
# classical arms are the reference that shows the measurement works.
DEFAULT_ARMS = ["linear", "matched_param_fullrank", "low_rank",
                "fourier_rff", "quantum_vqc", "quantum_reupload"]


# ------------------------------------------------------------------ snapshots
def snapshot(module):
    """Clone every parameter and buffer, detached from the graph."""
    return (
        {n: p.detach().clone() for n, p in module.named_parameters()},
        {n: b.detach().clone() for n, b in module.named_buffers()},
    )


def diff_report(before, after):
    """
    Which tensors changed, and by how much.

    torch.equal is bit-exact on purpose: "changed by 1e-9" is still changed, and
    a frozen module has no mechanism that would produce a small change rather
    than none at all.
    """
    b_par, b_buf = before
    a_par, a_buf = after
    out = {"params_changed": [], "buffers_changed": [],
           "max_param_delta": 0.0, "max_buffer_delta": 0.0,
           "n_params": len(b_par), "n_buffers": len(b_buf)}

    for n, t in b_par.items():
        if not torch.equal(t, a_par[n]):
            out["params_changed"].append(n)
            out["max_param_delta"] = max(
                out["max_param_delta"], float((a_par[n] - t).abs().max()))
    for n, t in b_buf.items():
        if not torch.equal(t.float(), a_buf[n].float()):
            out["buffers_changed"].append(n)
            out["max_buffer_delta"] = max(
                out["max_buffer_delta"], float((a_buf[n].float() - t.float()).abs().max()))

    out["n_params_changed"] = len(out["params_changed"])
    out["n_buffers_changed"] = len(out["buffers_changed"])
    out["unchanged"] = (out["n_params_changed"] == 0 and out["n_buffers_changed"] == 0)
    # Only the first few names; the counts carry the information.
    out["params_changed"] = out["params_changed"][:5]
    out["buffers_changed"] = out["buffers_changed"][:5]
    return out


# ------------------------------------------------------------------ proof C
def input_gradient_check(model, num_classes, device, batch=8):
    """
    Does d(loss)/d(z) survive the head?

    For the quantum arm this is the derivative passing through the simulated
    circuit - the step a sceptical reader will not accept on the strength of a
    library's documentation. Measured on random features so it depends on
    nothing but the model.
    """
    model = model.to(device).eval()
    h = torch.randn(batch, 256, device=device, requires_grad=True)
    y = torch.randint(0, num_classes, (batch,), device=device)

    logits = model.forward_from_features(h)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    g = h.grad
    return {
        "input_grad_norm": float(g.norm()) if g is not None else None,
        "input_grad_max": float(g.abs().max()) if g is not None else None,
        "input_grad_flows": bool(g is not None and float(g.abs().max()) > 0.0),
    }


# ------------------------------------------------------------------ cell
def run_cell(arm, freeze_policy, dataset, n_per_class, seed, dim=4,
             max_epochs=3, force=False, no_bn_eval=False):
    """
    One verification run. Short by design: this checks plumbing, and plumbing
    either works on epoch 1 or it does not.

    no_bn_eval=True is the NEGATIVE CONTROL for Proof A. It skips set_bn_eval(),
    which must make BatchNorm buffers drift. If they do not drift, the positive
    test is vacuous and the whole check is worthless.
    """
    keys = dict(arm=arm, fp=freeze_policy, dataset=dataset,
                regime=n_per_class, dim=dim, seed=seed)
    if no_bn_eval:
        keys["nobneval"] = 1
    if not force and shards.exists(EXPERIMENT, **keys):
        return None

    config.set_determinism(seed)
    device = config.DEVICE
    C = num_classes_of(dataset)
    t0 = time.time()

    model = build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                      seed=seed, freeze_policy=freeze_policy,
                      build_backbone=True).to(device)

    # Static structural facts, before anything is trained.
    n_bb_trainable = sum(p.numel() for p in model.backbone.parameters()
                         if p.requires_grad)
    capacity = model.capacity_report()
    grad_in = input_gradient_check(
        build_arm(arm, d=dim, num_classes=C, n_layers=config.VQC_LAYERS,
                  seed=seed, build_backbone=False), C, device)

    if no_bn_eval:
        # Disarm the guard that keeps frozen BatchNorm in eval mode.
        model.set_bn_eval = lambda: None

    before = snapshot(model.backbone)

    train, val, test, meta = get_loaders(dataset, n_per_class=n_per_class,
                                         seed=seed, augment=False)
    metrics, history, _ = train_model(
        model, train, val, test, num_classes=C, use_features=False,
        is_quantum=(arm in QUANTUM_ARMS), verbose=False,
        max_epochs=max_epochs, patience=max_epochs)

    after = snapshot(model.backbone)
    diff = diff_report(before, after)

    # Displacement restricted to layer3 - the only block that is ever unfrozen,
    # so it is where movement is expected under the adaptive policy.
    l3 = [float((after[0][n] - t).norm())
          for n, t in before[0].items() if n.startswith("features.6")]
    diff["layer3_displacement"] = float(np.sqrt(sum(x ** 2 for x in l3))) if l3 else 0.0

    flow = metrics.get("grad_flow") or {}
    payload = {
        "frozen_check": diff,
        "grad_flow": flow,
        "input_gradient": grad_in,
        "backbone_trainable_params": n_bb_trainable,
        "capacity": capacity,
        "auc": metrics.get("auc"),
        "epochs_run": metrics.get("epochs_run"),
        "no_bn_eval": bool(no_bn_eval),
        "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test", "regime")},
        "wall_time": time.time() - t0,
    }
    shards.write(EXPERIMENT, payload, **keys)

    del model
    torch.cuda.empty_cache()
    return payload


# ------------------------------------------------------------------ summary
def summarise():
    rows = shards.load_all(EXPERIMENT)
    if not rows:
        print("No shards found. Run without --summary-only first.")
        return

    normal = [r for r in rows if not r.get("no_bn_eval")]
    control = [r for r in rows if r.get("no_bn_eval")]

    # ---------------------------------------------------------- PROOF A
    print("=" * 78)
    print("PROOF A - IS THE FROZEN BACKBONE ACTUALLY FROZEN?")
    print("=" * 78)
    print("Bit-exact comparison of every backbone parameter AND buffer before")
    print("and after training. Buffers are the fragile part: BatchNorm running")
    print("statistics update on the FORWARD pass, so requires_grad=False does")
    print("not protect them - only eval() mode does.")
    print(f"\n{'arm':24s} {'encoder':>9s} {'bb train':>9s} {'par chg':>8s} "
          f"{'buf chg':>8s} {'max delta':>11s}  verdict")
    print("-" * 92)

    a_pass = True
    for r in sorted(normal, key=lambda x: (x["keys"]["fp"], x["keys"]["arm"])):
        k, d = r["keys"], r["frozen_check"]
        enc = "frozen" if k["fp"] == FROZEN else "adaptive"
        md = max(d["max_param_delta"], d["max_buffer_delta"])
        if k["fp"] == FROZEN:
            ok = d["unchanged"]
            verdict = "UNCHANGED" if ok else "VIOLATION: backbone moved"
            a_pass &= ok
        else:
            ok = not d["unchanged"]
            verdict = "changed (expected)" if ok else "VIOLATION: nothing moved"
            a_pass &= ok
        print(f"{k['arm']:24s} {enc:>9s} {r['backbone_trainable_params']:>9d} "
              f"{d['n_params_changed']:>8d} {d['n_buffers_changed']:>8d} "
              f"{md:>11.2e}  {verdict}")

    # ---------------------------------------------------- NEGATIVE CONTROL
    print(f"\n--- Negative control: same runs WITHOUT set_bn_eval() ---")
    print("BatchNorm buffers MUST drift here. If they do not, the check above")
    print("is vacuous - it would pass even on a broken pipeline.")
    if not control:
        print("  NOT RUN. Use --negative-control; without it Proof A is untested.")
        c_pass = False
    else:
        c_pass = True
        for r in sorted(control, key=lambda x: x["keys"]["arm"]):
            d = r["frozen_check"]
            drifted = d["n_buffers_changed"] > 0
            c_pass &= drifted
            print(f"  {r['keys']['arm']:24s} buffers changed="
                  f"{d['n_buffers_changed']:>3d} "
                  f"max delta={d['max_buffer_delta']:.2e}  "
                  f"{'OK (test is discriminative)' if drifted else 'TEST IS VACUOUS'}")

    # ---------------------------------------------------------- PROOF B
    print("\n" + "=" * 78)
    print("PROOF B - DO GRADIENTS REACH THE ENCODER, FROM EVERY HEAD?")
    print("=" * 78)
    print("Three independent signals. All three must hold for the adaptive")
    print("regime, and the first must be ABSENT for the frozen regime.")
    print(f"\n{'arm':24s} {'encoder':>9s} {'bb grad':>10s} {'l3 move':>10s} "
          f"{'d(loss)/dz':>11s}  verdict")
    print("-" * 92)

    b_pass = True
    for r in sorted(normal, key=lambda x: (x["keys"]["fp"], x["keys"]["arm"])):
        k = r["keys"]
        enc = "frozen" if k["fp"] == FROZEN else "adaptive"
        bb = (r.get("grad_flow") or {}).get("backbone")
        bb_s = f"{bb['mean']:10.4f}" if bb else "         -"
        move = r["frozen_check"].get("layer3_displacement", 0.0)
        gin = r.get("input_gradient") or {}
        gz = gin.get("input_grad_norm")
        gz_s = f"{gz:11.4e}" if gz else "          -"

        if k["fp"] == FROZEN:
            ok = bb is None
            verdict = "no gradient (correct)" if ok else "VIOLATION: gradient reached backbone"
        else:
            ok = bool(bb) and move > 0 and gin.get("input_grad_flows")
            verdict = ("flows + encoder moves" if ok else
                       "VIOLATION: " + ("no backbone gradient" if not bb else
                                        "encoder did not move" if move <= 0 else
                                        "no gradient into head input"))
        b_pass &= ok
        print(f"{k['arm']:24s} {enc:>9s} {bb_s} {move:10.4f} {gz_s}  {verdict}")

    # ---------------------------------------------------------- capacity
    print("\n" + "=" * 78)
    print("WHERE THE TRAINABLE PARAMETERS LIVE")
    print("=" * 78)
    print("Relevant to how much of the 'frozen backbone' result is the head at all.")
    seen = {}
    for r in normal:
        c = r.get("capacity")
        if c:
            seen[(r["keys"]["arm"], r["keys"]["fp"])] = c
    print(f"\n{'arm':24s} {'encoder':>9s} {'backbone':>9s} {'bneck':>7s} "
          f"{'head':>6s} {'total':>9s} {'head %':>7s}")
    print("-" * 78)
    for (arm, fp) in sorted(seen):
        c = seen[(arm, fp)]
        enc = "frozen" if fp == FROZEN else "adaptive"
        print(f"{arm:24s} {enc:>9s} {c['backbone']:>9d} {c['bottleneck']:>7d} "
              f"{c['head']:>6d} {c['total']:>9d} {100 * c['head_share']:>6.1f}%")

    # ---------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print(f"PROOF A (freezing)          {'PASS' if a_pass else 'FAIL'}")
    print(f"  negative control          {'PASS' if c_pass else 'NOT RUN / FAIL'}")
    print(f"PROOF B (gradient flow)     {'PASS' if b_pass else 'FAIL'}")
    print("=" * 78)
    if a_pass and b_pass and c_pass:
        print("Both structural claims are now measured, not asserted. These tables")
        print("belong in the methods section - they answer 'how do you know the")
        print("backbone was frozen?' and 'does the VQC actually train the encoder?'")
    else:
        print("At least one claim is unsupported. Do NOT run further experiments")
        print("until this passes: every downstream result depends on it.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    p.add_argument("--dataset", default="breastmnist",
                   help="smallest dataset; this verifies plumbing, not accuracy")
    p.add_argument("--regime", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--negative-control", action="store_true", default=True,
                   help="also run without set_bn_eval(); buffers must then drift")
    p.add_argument("--no-negative-control", dest="negative_control",
                   action="store_false")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if args.summary_only:
        summarise()
        return

    plan = [(arm, fp) for fp in (FROZEN, ADAPTIVE) for arm in args.arms]
    total = len(plan) + (1 if args.negative_control else 0)
    print(f"Flow verification | {total} runs | {args.dataset} n={args.regime} "
          f"d={args.dim} epochs={args.max_epochs}")
    print(f"  device={config.DEVICE} sha={config.git_sha()[:8]}\n")

    for i, (arm, fp) in enumerate(plan, 1):
        enc = "frozen" if fp == FROZEN else "adaptive"
        print(f"[{i}/{total}] {arm} / {enc} ...", flush=True)
        run_cell(arm, fp, args.dataset, args.regime, args.seed,
                 dim=args.dim, max_epochs=args.max_epochs, force=args.force)

    if args.negative_control:
        # One arm suffices: set_bn_eval() is a property of the backbone, not of
        # the head attached to it.
        print(f"[{total}/{total}] negative control (no set_bn_eval) ...", flush=True)
        run_cell(args.arms[0], FROZEN, args.dataset, args.regime, args.seed,
                 dim=args.dim, max_epochs=args.max_epochs, force=args.force,
                 no_bn_eval=True)

    print()
    summarise()


if __name__ == "__main__":
    main()
