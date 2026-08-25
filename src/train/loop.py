"""
The single training loop. Every experiment calls this.

WHY ONE LOOP
------------
The previous codebase duplicated ~150 lines between scripts 01 and 02. The two
copies had already drifted: the `>=` checkpoint bug and the unreachable
early-stopping clamp existed in one form in each. Any behaviour that differs
between the frozen and end-to-end experiments must be an explicit argument, not
an accident of copy-paste.

DUAL CHECKPOINT SELECTION - AND WHY IT MATTERS HERE
----------------------------------------------------
The old version selected the best epoch by validation Macro-F1 while the paper
reports AUC. Normally a minor mismatch; in this study it is a confound.

Macro-F1 depends on the argmax threshold. The VQC has a documented calibration
failure - probability mass collapsing toward a point - so its validation F1 goes
nearly FLAT across epochs. Selection then becomes close to arbitrary, and the
reported AUC comes from an arbitrarily chosen epoch. Because that pathology
affects the arms unequally, the selection criterion silently becomes part of the
comparison.

Both are tracked. `test` reports the AUC-selected model (the primary endpoint),
`test["test_f1_selected"]` the F1-selected one. Reporting both turns a hidden
confound into a stated sensitivity analysis, and material disagreement between
them is itself a finding about calibration.

The LR scheduler follows validation AUC, consistent with the primary endpoint.

PER-MODULE GRADIENT NORMS - THE FLOW PROOF
-------------------------------------------
Every epoch records the L2 gradient norm of each component separately:

    grad_norm_backbone     does the encoder actually receive gradient?
    grad_norm_bottleneck   is the 256->d projection learning?
    grad_norm_head         is the head learning?
    grad_norm_classifier

This exists because the study makes two claims it could not previously
substantiate with a number:

    1. With freeze_policy="all" the backbone is untouched.
       grad_norm_backbone must be None (no parameter carries a gradient).
    2. With freeze_policy="layer3_only" gradients flow back into the encoder
       FROM EVERY HEAD, including the quantum one.
       grad_norm_backbone must be non-zero for quantum arms too.

"backprop through a PennyLane TorchLayer propagates input gradients" is a
statement about the library. These numbers are a statement about THIS model, on
THIS data, and they are what a reviewer asking "does the VQC train the encoder?"
actually wants to see.

They are measured BEFORE clipping, in the same place as pre_clip_grad_norm -
clip_grad_norm_ rescales gradients in place, so measuring afterwards would
report the clipped values.

PER-EPOCH METRICS ARE COMPUTED IN LIGHT MODE
---------------------------------------------
compute_metrics(..., light=True) returns only AUC, Macro-F1, ECE and probability
spread - everything selection and the training curves consume. Average
precision, sensitivity, specificity and balanced accuracy are skipped.

Those are wanted on the FINAL test evaluation, not ~200 times per run on a
validation set of 20 images. On 9-class datasets the one-hot average-precision
call is the most expensive operation in the epoch. `light` existed but was never
passed, so the cost was being paid for numbers nothing read.

Consequence: `val_bal_acc` is no longer in `history`. It is still reported on
every test evaluation, which is the only place it is used.

LEARNING-RATE OVERRIDES
-----------------------
lr_head / lr_quantum / lr_backbone default to config but can be overridden per
run. A fixed shared LR is NOT neutral: measured gradient norms at d=4 span
0.48-0.76 for quantum_vqc against 0.97-4.79 for the classical arms, so an
identical LR gives the quantum arm systematically smaller effective steps. "The
VQC underperforms" and "the VQC was under-stepped" are then indistinguishable -
the cheapest way for a reviewer to dismiss a negative result.

Passing them as ARGUMENTS rather than mutating config globals matters: the shard
key is derived from what was explicitly requested, so a later change to
config.LR_HEAD cannot retroactively make old shards look like tuned ones.

best_val_auc is returned so 09_lr_selection.py can choose an LR on VALIDATION
performance without ever touching test.

INSTRUMENTATION THAT EXISTS FOR THE MANUSCRIPT
----------------------------------------------
pre_clip_grad_norm is the GLOBAL norm over all trainable parameters. Measured
values at d=4 span 0.5 (quantum_vqc) to 15.9 (deep_funnel), so a clip threshold
of 1.0 would have bound on classical arms while never touching the quantum arm -
a per-arm learning-rate multiplier disguised as a safety net. GRAD_CLIP_NORM is
20.0 for that reason, and the norms are logged so the choice stays auditable.

quantum_grad_var is logged for quantum arms. Gradient variance falls ~62x from
d=4 to d=16, so a d=16 failure must be attributable to trainability (barren
plateau) or to expressivity, and the two are distinguishable only with this
number in hand.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from train.metrics import compute_metrics, class_weights

# Components whose gradient norms are tracked separately. Order matters only for
# display; membership is decided by the parameter-name prefix.
GRAD_GROUPS = ("backbone", "bottleneck", "head", "classifier")


def _forward(model, x, use_features: bool):
    return model.forward_from_features(x) if use_features else model(x)


def module_grad_norms(model):
    """
    L2 gradient norm per component, measured BEFORE clipping.

    Returns None for a component with no gradient-carrying parameter, which is
    the signature of a correctly frozen module - distinct from 0.0, which would
    mean "receives gradient, and it happens to vanish".
    """
    sq = {g: [] for g in GRAD_GROUPS}
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        for g in GRAD_GROUPS:
            if name.startswith(g):
                sq[g].append(float(p.grad.detach().norm().item()) ** 2)
                break
    return {g: (float(np.sqrt(sum(v))) if v else None) for g, v in sq.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device,
             use_features=False, return_probs=False, light=False):
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels, probs = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long().to(device, non_blocking=True)
        logits = _forward(model, x, use_features)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    m = compute_metrics(labels, preds, probs, num_classes, light=light)
    m["loss"] = total_loss / max(n, 1)
    if return_probs:
        return m, probs, labels
    return m


def train_model(model, train_loader, val_loader, test_loader, *,
                num_classes, device=None, use_features=False,
                weight_decay=None, max_epochs=None, patience=None,
                is_quantum=False, verbose=True, return_probs=False,
                lr_head=None, lr_quantum=None, lr_backbone=None):
    """
    Trains, selects on BOTH validation AUC and validation Macro-F1, and returns
    (test_metrics, history, best_state) - or five values when return_probs=True:
    (test_metrics, history, best_state, test_probs, test_labels).

    test_metrics["best_val_auc"] is the selection score, used by LR tuning so
    hyperparameters are never chosen on test.

    test_metrics["grad_flow"] summarises the per-module gradient norms over
    training - the empirical evidence that the encoder is or is not being
    trained by this particular head.

    best_state holds the AUC-selected TRAINABLE parameters only; full state dicts
    across thousands of runs would be tens of GB of identical ImageNet weights.
    """
    device = device or config.DEVICE
    weight_decay = config.WEIGHT_DECAY if weight_decay is None else weight_decay
    max_epochs = max_epochs or config.MAX_EPOCHS
    patience = patience or config.PATIENCE
    lr_head = config.LR_HEAD if lr_head is None else float(lr_head)
    lr_quantum = config.LR_QUANTUM if lr_quantum is None else float(lr_quantum)
    lr_backbone = config.LR_BACKBONE if lr_backbone is None else float(lr_backbone)
    model = model.to(device)

    groups = model.param_groups(lr_backbone, lr_head, lr_quantum, weight_decay)
    optimizer = optim.Adam(groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15)

    y_train = []
    for _, y in train_loader:
        y_train.extend(y.view(-1).tolist())
    w = torch.from_numpy(
        class_weights(y_train, num_classes, config.CLASS_WEIGHT_CLIP)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    trainable = [p for p in model.parameters() if p.requires_grad]
    min_epochs = config.min_epochs_for(len(train_loader))

    # val_bal_acc is intentionally absent: light mode does not compute it, and
    # nothing consumes it per-epoch. It is still reported on test.
    history = {k: [] for k in
               ["train_loss", "train_f1", "train_auc", "val_loss", "val_f1",
                "val_auc", "val_ece", "val_prob_std",
                "pre_clip_grad_norm", "quantum_grad_var", "epoch_time"]
               + [f"grad_norm_{g}" for g in GRAD_GROUPS]}

    best_auc, best_auc_state, best_auc_epoch = -1.0, None, -1
    best_f1, best_f1_state, best_f1_epoch = -1.0, None, -1
    stale = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        model.set_bn_eval()          # must follow .train(); it resets submodules

        run_loss, n = 0.0, 0
        preds, labels, probs, gnorms, qvars = [], [], [], [], []
        mod_norms = {g: [] for g in GRAD_GROUPS}

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = _forward(model, x, use_features)
            loss = criterion(logits, y)
            loss.backward()

            # Per-module norms FIRST: clip_grad_norm_ rescales in place, so
            # measuring after it would report clipped values.
            for g, v in module_grad_norms(model).items():
                if v is not None:
                    mod_norms[g].append(v)

            # returns the GLOBAL norm BEFORE clipping - the parity diagnostic
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(
                trainable, config.GRAD_CLIP_NORM)))
            if is_quantum:
                gv = model.head.grad_variance()
                if gv:
                    qvars.append(gv["var"])
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            n += x.size(0)
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                probs.append(p.cpu().numpy())
                preds.append(p.argmax(dim=1).cpu().numpy())
                labels.append(y.cpu().numpy())

        # RUNNING training metrics - collected while weights were still changing
        # within the epoch. Fine for the overfitting plot, not to be quoted as
        # end-of-epoch training performance.
        tr = compute_metrics(np.concatenate(labels), np.concatenate(preds),
                             np.concatenate(probs), num_classes, light=True)
        va = evaluate(model, val_loader, criterion, num_classes, device,
                      use_features, light=True)
        scheduler.step(va["auc"] or 0.0)

        history["train_loss"].append(run_loss / max(n, 1))
        history["train_f1"].append(tr["macro_f1"])
        history["train_auc"].append(tr["auc"])
        history["val_loss"].append(va["loss"])
        history["val_f1"].append(va["macro_f1"])
        history["val_auc"].append(va["auc"])
        history["val_ece"].append(va["ece"])
        history["val_prob_std"].append(va["prob_std"])
        history["pre_clip_grad_norm"].append(float(np.mean(gnorms)) if gnorms else None)
        history["quantum_grad_var"].append(float(np.mean(qvars)) if qvars else None)
        for g in GRAD_GROUPS:
            history[f"grad_norm_{g}"].append(
                float(np.mean(mod_norms[g])) if mod_norms[g] else None)
        history["epoch_time"].append(time.time() - t0)

        cur_auc = va["auc"] or 0.0
        cur_f1 = va["macro_f1"] or 0.0

        # strict > : >= saved the LAST tied epoch, so degenerate models with a
        # constant validation score checkpointed an arbitrary late epoch.
        if cur_auc > best_auc:
            best_auc, best_auc_epoch, stale = cur_auc, epoch, 0
            best_auc_state = copy.deepcopy(model.trainable_state_dict())
            if verbose:
                print(f"      ep {epoch+1:03d} val_auc={cur_auc:.4f} "
                      f"val_f1={cur_f1:.4f} *")
        else:
            stale += 1

        if cur_f1 > best_f1:
            best_f1, best_f1_epoch = cur_f1, epoch
            best_f1_state = copy.deepcopy(model.trainable_state_dict())

        if stale >= patience and epoch >= min_epochs:
            if verbose:
                print(f"      early stop at epoch {epoch+1}")
            break

    # --- F1-selected model: the sensitivity check -------------------------
    f1_selected = None
    if best_f1_state is not None:
        model.load_state_dict(best_f1_state, strict=False)
        f1_selected = evaluate(model, test_loader, criterion, num_classes,
                               device, use_features)

    # --- AUC-selected model: the primary report ---------------------------
    if best_auc_state is not None:
        model.load_state_dict(best_auc_state, strict=False)

    if return_probs:
        test, probs, labels = evaluate(model, test_loader, criterion, num_classes,
                                       device, use_features, return_probs=True)
    else:
        test = evaluate(model, test_loader, criterion, num_classes, device, use_features)
        probs = labels = None

    # Compact flow summary. `None` means the component carried no gradient at
    # all, which is what a correctly frozen module looks like.
    def _summ(key):
        vals = [v for v in history[key] if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
                "first": float(vals[0]), "last": float(vals[-1]),
                "n_epochs": len(vals)}

    test["grad_flow"] = {g: _summ(f"grad_norm_{g}") for g in GRAD_GROUPS}
    test["backbone_received_gradient"] = test["grad_flow"]["backbone"] is not None

    test["best_epoch"] = best_auc_epoch
    test["best_val_auc"] = best_auc
    test["best_epoch_f1_selected"] = best_f1_epoch
    test["best_val_f1"] = best_f1
    test["epochs_run"] = len(history["epoch_time"])
    test["mean_epoch_time"] = float(np.mean(history["epoch_time"]))
    test["selection_metric"] = "val_auc"
    test["test_f1_selected"] = f1_selected
    test["lr_head"] = lr_head
    test["lr_quantum"] = lr_quantum
    test["lr_backbone"] = lr_backbone

    if return_probs:
        return test, history, best_auc_state, probs, labels
    return test, history, best_auc_state
