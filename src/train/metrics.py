"""
Metrics.

WHY CALIBRATION IS MEASURED HERE
--------------------------------
The conference "Precision Paradox" reported F1 collapsing to 0.145 under noise
while AUC held at 0.612 with a probability standard deviation of 0.0057. That
combination is the signature of a CALIBRATION failure - the probability mass
collapsing toward a single point so argmax stops discriminating - not of the
decision boundary degrading. Those are different claims with different
implications, and the paper cannot tell them apart without reporting AUC,
Macro-F1, ECE and probability spread at every noise level.

So every evaluation returns all of them. If AUC holds while F1 craters, the
finding is threshold drift and the manuscript says threshold drift.

WHY PR-AUC, SENSITIVITY AND SPECIFICITY ARE ALSO HERE
------------------------------------------------------
ROC-AUC is the right primary endpoint for a threshold-free comparison, but it is
not what a clinical reader interprets, and it is known to look optimistic under
class imbalance. Medical imaging reviewers expect:

    average precision (PR-AUC) - dominated by the minority class, so it exposes
                                 failures ROC-AUC can hide
    sensitivity / specificity  - the numbers a clinician can act on

They cost two lines and their absence is a predictable reviewer objection.
For multi-class, sensitivity is macro-averaged recall (= balanced accuracy) and
specificity is macro-averaged one-vs-rest true-negative rate.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)


def clean_val(v):
    """NaN -> None, for RFC 8259 compliant JSON."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    return float(v)


def expected_calibration_error(labels, probs, n_bins: int = 15) -> float:
    """
    Standard binned ECE over the predicted-class confidence.
    Rises when a model becomes overconfident or underconfident relative to its
    realised accuracy.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _sensitivity_specificity(labels, preds, num_classes):
    """
    Binary: sensitivity/specificity for the positive class (label 1).
    Multi-class: macro-averaged one-vs-rest.
    """
    try:
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    except ValueError:
        return np.nan, np.nan

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        return sens, spec

    sens_c, spec_c = [], []
    total = cm.sum()
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        if tp + fn:
            sens_c.append(tp / (tp + fn))
        if tn + fp:
            spec_c.append(tn / (tn + fp))
    return (float(np.mean(sens_c)) if sens_c else np.nan,
            float(np.mean(spec_c)) if spec_c else np.nan)


def _average_precision(labels, probs, num_classes):
    """PR-AUC. Binary uses the positive column; multi-class is macro OVR."""
    try:
        if num_classes == 2:
            return average_precision_score(labels, probs[:, 1])
        onehot = np.eye(num_classes)[np.asarray(labels)]
        return average_precision_score(onehot, probs, average="macro")
    except ValueError:
        return np.nan


def compute_metrics(labels, preds, probs, num_classes: int) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            # average is pinned explicitly rather than left to the default
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan   # a class absent from this evaluation split

    sens, spec = _sensitivity_specificity(labels, preds, num_classes)

    return {
        "acc": clean_val(accuracy_score(labels, preds)),
        "bal_acc": clean_val(balanced_accuracy_score(labels, preds)),
        "macro_f1": clean_val(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": clean_val(auc),
        # PR-AUC: minority-class dominated, exposes what ROC-AUC hides
        "ap": clean_val(_average_precision(labels, probs, num_classes)),
        # Clinically interpretable operating point
        "sensitivity": clean_val(sens),
        "specificity": clean_val(spec),
        "ece": clean_val(expected_calibration_error(labels, probs)),
        # Near-zero spread with a non-trivial AUC is the "collapsed probability"
        # signature that distinguishes calibration failure from boundary failure.
        "prob_std": clean_val(float(probs.max(axis=1).std())),
    }


def class_weights(labels, num_classes: int, clip=(0.1, 10.0)) -> np.ndarray:
    """
    Inverse-frequency weights, clipped.

    Unclipped, a class with a single training example produced a weight in the
    hundreds of thousands, which destabilised exactly the scarce regimes under
    study. Note this is a no-op on the balanced scarcity grid by construction -
    it only does work in the full-data reference row.
    """
    counts = np.bincount(np.asarray(labels).ravel(), minlength=num_classes)
    total = counts.sum()
    w = total / (num_classes * np.maximum(counts, 1))
    return np.clip(w, clip[0], clip[1]).astype(np.float32)
