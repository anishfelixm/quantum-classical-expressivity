"""
The single training loop. Every experiment calls this.

WHY ONE LOOP
------------
The previous codebase duplicated ~150 lines between scripts 01 and 02. The two
copies had already drifted: the `>=` checkpoint bug and the unreachable
early-stopping clamp existed in one form in each. Any behaviour that differs
between the frozen and end-to-end experiments must now be an explicit argument,
not an accident of copy-paste.

DUAL CHECKPOINT SELECTION - AND WHY IT MATTERS HERE
----------------------------------------------------
The old version selected the best epoch by validation Macro-F1 and the paper
reports AUC. Normally a minor mismatch; in this study it is a confound.

Macro-F1 depends on the argmax threshold. The VQC has a documented calibration
failure - probability mass collapsing toward a point - so its validation F1 goes
nearly FLAT across epochs. Selection then becomes close to arbitrary, and the
AUC reported comes from an arbitrarily chosen epoch. Because that pathology
affects the arms unequally, the selection criterion silently becomes part of the
comparison.

So both are tracked:

    best_by_f1   - the decision-boundary view (was the only one before)
    best_by_auc  - the ranking view, matching the primary endpoint

`test` reports the AUC-selected model, `test_f1_selected` the F1-selected one.
Reporting both turns a hidden confound into a stated sensitivity analysis, and
if the two disagree materially that is itself a finding about calibration.

The LR scheduler follows validation AUC, consistent with the primary endpoint.

INSTRUMENTATION THAT EXISTS FOR THE MANUSCRIPT
----------------------------------------------
pre_clip_grad_norm is logged every epoch. Measured global L2 norms at d=4 span
0.5 (quantum_vqc) to 15.9 (deep_funnel), so a clip threshold of 1.0 would have
bound on classical arms while never touching the quantum arm - a per-arm
learning-rate multiplier disguised as a safety net. GRAD_CLIP_NORM is 20.0 for
that reason, and the norms are logged so the choice stays auditable.

quantum_grad_var is logged for the VQC. Gradient variance falls ~62x from d=4 to
d=16, so a d=16 failure must be attributable to trainability (barren plateau) or
to expressivity, and the two are distinguishable only with this number in hand.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from train.metrics import compute_metrics, class_weights


def _forward(model, x, use_features: bool):
    return model.forward_from_features(x) if use_features else model(x)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device,
             use_features=False, return_probs=False):
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

    m = compute_metrics(labels, preds, probs, num_classes)
    m["loss"] = total_loss / max(n, 1)
    if return_probs:
        return m, probs, labels
    return m


def train_model(model, train_loader, val_loader, test_loader, *,
                num_classes, device=None, use_features=False,
                weight_decay=None, max_epochs=None, patience=None,
                is_quantum=False, verbose=True, return_probs=False):
    """
    Trains, selects on BOTH validation AUC and validation Macro-F1, and returns
    (test_metrics, history, best_state).

    test_metrics["..."]              -> AUC-selected model (primary)
    test_metrics["test_f1_selected"] -> F1-selected model (sensitivity check)

    best_state is the AUC-selected trainable parameters. Only trainable
    parameters are stored - full state dicts across thousands of runs would be
    tens of GB of identical frozen ImageNet weights.
    """
    device = device or config.DEVICE
    weight_decay = config.WEIGHT_DECAY if weight_decay is None else weight_decay
    max_epochs = max_epochs or config.MAX_EPOCHS
    patience = patience or config.PATIENCE
    model = model.to(device)

    groups = model.param_groups(config.LR_BACKBONE, config.LR_HEAD,
                                config.LR_QUANTUM, weight_decay)
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

    history = {k: [] for k in
               ["train_loss", "train_f1", "train_auc", "val_loss", "val_f1",
                "val_auc", "val_bal_acc", "val_ece", "val_prob_std",
                "pre_clip_grad_norm", "quantum_grad_var", "epoch_time"]}

    best_auc, best_auc_state, best_auc_epoch = -1.0, None, -1
    best_f1, best_f1_state, best_f1_epoch = -1.0, None, -1
    stale = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        model.set_bn_eval()          # must follow .train(); it resets submodules

        run_loss, n = 0.0, 0
        preds, labels, probs, gnorms, qvars = [], [], [], [], []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = _forward(model, x, use_features)
            loss = criterion(logits, y)
            loss.backward()

            # returns the norm BEFORE clipping - this is the parity diagnostic
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

        # NOTE: these are RUNNING training metrics - collected while weights
        # were still changing within the epoch. Fine for the overfitting plot,
        # not to be quoted as end-of-epoch training performance.
        tr = compute_metrics(np.concatenate(labels), np.concatenate(preds),
                             np.concatenate(probs), num_classes)
        va = evaluate(model, val_loader, criterion, num_classes, device, use_features)
        scheduler.step(va["auc"] or 0.0)

        history["train_loss"].append(run_loss / max(n, 1))
        history["train_f1"].append(tr["macro_f1"])
        history["train_auc"].append(tr["auc"])
        history["val_loss"].append(va["loss"])
        history["val_f1"].append(va["macro_f1"])
        history["val_auc"].append(va["auc"])
        history["val_bal_acc"].append(va["bal_acc"])
        history["val_ece"].append(va["ece"])
        history["val_prob_std"].append(va["prob_std"])
        history["pre_clip_grad_norm"].append(float(np.mean(gnorms)) if gnorms else None)
        history["quantum_grad_var"].append(float(np.mean(qvars)) if qvars else None)
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

    test["best_epoch"] = best_auc_epoch
    test["best_val_auc"] = best_auc
    test["best_epoch_f1_selected"] = best_f1_epoch
    test["best_val_f1"] = best_f1
    test["epochs_run"] = len(history["epoch_time"])
    test["mean_epoch_time"] = float(np.mean(history["epoch_time"]))
    test["selection_metric"] = "val_auc"
    test["test_f1_selected"] = f1_selected

    if return_probs:
        return test, history, best_auc_state, probs, labels
    return test, history, best_auc_state
