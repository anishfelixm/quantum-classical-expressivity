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

So every evaluation returns all four. If AUC holds while F1 craters, the
finding is threshold drift and the manuscript says threshold drift.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
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

    return {
        "acc": clean_val(accuracy_score(labels, preds)),
        "bal_acc": clean_val(balanced_accuracy_score(labels, preds)),
        "macro_f1": clean_val(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": clean_val(auc),
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
