"""
The single training loop. Experiments 1 and 2 both call this.

WHY ONE LOOP
------------
The previous codebase duplicated ~150 lines between scripts 01 and 02. The two
copies had already drifted: the `>=` checkpoint bug and the unreachable
early-stopping clamp existed in one form in each. Any behaviour that differs
between the frozen and end-to-end experiments must now be an explicit argument,
not an accident of copy-paste.

INSTRUMENTATION THAT EXISTS FOR THE MANUSCRIPT
----------------------------------------------
pre_clip_grad_norm is logged every epoch. Measured layer3 gradient norms are
~665 against a clip threshold of 1.0 - a 665x rescale - while the quantum
arm's natural scale is ~6.4. If clipping binds differentially across arms then
"the quantum model trains more stably" is partly an artefact of the threshold,
not a property of the architecture. Logging it makes that checkable instead of
assumable.

quantum_grad_var is logged for the VQC. Gradient variance falls ~62x from d=4
to d=16, so a d=16 failure must be attributable to trainability (barren
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


def _forward(model, x, use_features: bool):
    return model.forward_from_features(x) if use_features else model(x)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device, use_features=False):
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels, probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.view(-1).long().to(device)
        logits = _forward(model, x, use_features)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        p = torch.softmax(logits, dim=1)
        probs.extend(p.cpu().numpy())
        preds.extend(p.argmax(dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    m = compute_metrics(labels, preds, probs, num_classes)
    m["loss"] = total_loss / max(n, 1)
    return m


def train_model(model, train_loader, val_loader, test_loader, *,
                num_classes, device=None, use_features=False,
                weight_decay=None, max_epochs=None, patience=None,
                is_quantum=False, verbose=True):
    """
    Trains, selects on validation Macro-F1, returns (test_metrics, history,
    best_state). Only trainable parameters are returned in best_state - saving
    full state dicts across ~4800 runs would be ~40 GB of frozen ImageNet
    weights repeated over and over.
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
                "val_auc", "val_bal_acc", "val_ece", "pre_clip_grad_norm",
                "quantum_grad_var", "epoch_time"]}

    best_f1, best_state, best_epoch, stale = -1.0, None, -1, 0

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        model.set_bn_eval()          # must follow .train(); it resets submodules

        run_loss, n = 0.0, 0
        preds, labels, probs, gnorms, qvars = [], [], [], [], []

        for x, y in train_loader:
            x = x.to(device)
            y = y.view(-1).long().to(device)

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
                probs.extend(p.cpu().numpy())
                preds.extend(p.argmax(dim=1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        tr = compute_metrics(labels, preds, probs, num_classes)
        va = evaluate(model, val_loader, criterion, num_classes, device, use_features)
        scheduler.step(va["macro_f1"] or 0.0)

        history["train_loss"].append(run_loss / max(n, 1))
        history["train_f1"].append(tr["macro_f1"])
        history["train_auc"].append(tr["auc"])
        history["val_loss"].append(va["loss"])
        history["val_f1"].append(va["macro_f1"])
        history["val_auc"].append(va["auc"])
        history["val_bal_acc"].append(va["bal_acc"])
        history["val_ece"].append(va["ece"])
        history["pre_clip_grad_norm"].append(float(np.mean(gnorms)) if gnorms else None)
        history["quantum_grad_var"].append(float(np.mean(qvars)) if qvars else None)
        history["epoch_time"].append(time.time() - t0)

        cur = va["macro_f1"] or 0.0
        # strict > : >= saved the LAST tied epoch, so degenerate models with a
        # constant validation F1 checkpointed an arbitrary late epoch, and the
        # robustness experiment then evaluated those weights.
        if cur > best_f1:
            best_f1, best_epoch, stale = cur, epoch, 0
            best_state = copy.deepcopy(model.trainable_state_dict())
            if verbose:
                print(f"      ep {epoch+1:03d} val_f1={cur:.4f} "
                      f"val_auc={va['auc']:.4f} *")
        else:
            stale += 1

        if stale >= patience and epoch >= min_epochs:
            if verbose:
                print(f"      early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test = evaluate(model, test_loader, criterion, num_classes, device, use_features)
    test["best_epoch"] = best_epoch
    test["best_val_f1"] = best_f1
    test["epochs_run"] = len(history["epoch_time"])
    test["mean_epoch_time"] = float(np.mean(history["epoch_time"]))
    return test, history, best_state
