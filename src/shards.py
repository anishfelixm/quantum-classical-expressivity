"""
Result shard I/O.

WHY SHARDS AND NOT ONE BIG JSON
-------------------------------
The previous scripts held everything in memory and rewrote a single JSON at the
end of each dataset loop. A crash at hour 200 of a multi-day sweep lost
everything since the last dataset boundary, and there was no way to resume.

One shard per (experiment, dataset, regime, dim, seed, arm, ...). The driver
skips shards that already exist, so an interrupted sweep resumes by re-running
the same command.

PROVENANCE
----------
Every shard records the git commit that produced it. When a reviewer asks which
code version generated a given table - and for a journal with a code
availability statement, someone will - the answer is exact rather than
approximate.

PREDICTIONS ARE WRITTEN AND READ **ONLY** THROUGH THIS MODULE
--------------------------------------------------------------
The pre-registered statistic is a NESTED bootstrap that resamples test indices
as well as seeds, which is impossible from scalar metrics alone. So every run
also writes its per-sample test probabilities.

Those files were previously named independently by 01, by 03, and by the reader
in 04 - three conventions, none of which agreed. 04 therefore found 03's files
and never found 01's, silently fell back to seed-level resampling (the weaker
analysis the plan forbids), and printed results that looked fine.

save_predictions() and load_predictions() are now the only way in or out. Both
derive the filename from the SAME shard keys via pred_path(), and both use the
same internal array names, so writer and reader cannot drift apart again.
"""
import json
import os
import time

import numpy as np

import config


# ------------------------------------------------------------------ naming
def _stringify(v):
    return str(v).replace("/", "-").replace(" ", "")


def shard_name(experiment, **keys):
    """Deterministic, order-independent shard filename."""
    parts = [experiment] + [f"{k}{_stringify(v)}" for k, v in sorted(keys.items())]
    return "__".join(parts) + ".json"


def shard_path(experiment, **keys):
    d = os.path.join(config.SHARD_DIR, experiment)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, shard_name(experiment, **keys))


def pred_tag(**keys):
    """
    Filename for a run's per-sample predictions, derived from the SAME keys that
    name its shard. Sorted, so it does not depend on the caller's argument order.
    """
    return "__".join(f"{k}{_stringify(v)}" for k, v in sorted(keys.items())) + ".npz"


def pred_dir(experiment):
    d = os.path.join(config.PREDICTION_DIR, experiment)
    os.makedirs(d, exist_ok=True)
    return d


def pred_path(experiment, **keys):
    return os.path.join(pred_dir(experiment), pred_tag(**keys))


def pred_exists(experiment, **keys):
    return os.path.exists(pred_path(experiment, **keys))


# ------------------------------------------------------------------ predictions
PROBS_KEY = "probs"          # single-condition runs (01, 09)
LABELS_KEY = "labels"

# sklearn's roc_auc_score requires multi-class scores to sum to 1 within roughly
# 1e-5. Anything looser than that is a storage artefact worth repairing;
# anything tighter is float noise that must be left alone, because touching it
# perturbs the ranking AUC depends on.
RENORM_TOLERANCE = 1e-6


def save_predictions(experiment, labels, probs, **keys) -> str:
    """
    probs may be:
        ndarray [N, C]                  -> stored under PROBS_KEY
        dict {condition: ndarray[N,C]}  -> one array per condition (03's noise sweep)

    STORED AS float32, NOT float16.

    float16 was chosen to halve the file size, on the reasoning that AUC needs
    only ~3 decimal places. Both halves of that reasoning were wrong.

    1. It broke multi-class AUC outright. float16 rounding makes rows sum to
       0.999647-1.000352 instead of 1.0, and sklearn's roc_auc_score rejects
       multi-class scores that do not sum to one (tolerance ~1e-5). Every
       multi-class cell raised ValueError, _auc returned NaN, and
       04_statistical_analysis.py fell back to seed-level resampling - the
       weaker analysis the pre-registration forbids - while printing a table
       that looked correct.

    2. AUC depends on RANKING, not on absolute probability. float16 spacing near
       0.9 is ~5e-4, so on a 3,421-image test set many samples collapse onto
       identical stored values. Those artificial ties are broken by averaged
       ranks, biasing the recomputed AUC away from the value computed at
       training time from float32 - so a shard's own metric and the bootstrap's
       recomputation would disagree.

    float32 costs ~110 KB per run before compression. That is not worth trading
    against the integrity of the headline statistic.
    """
    labels = np.asarray(labels).astype(np.int16)
    arrays = ({PROBS_KEY: np.asarray(probs, dtype=np.float32)}
              if not isinstance(probs, dict)
              else {str(k): np.asarray(v, dtype=np.float32) for k, v in probs.items()})

    path = pred_path(experiment, **keys)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **{LABELS_KEY: labels}, **arrays)
    os.replace(tmp, path)
    return os.path.basename(path)


def load_predictions(experiment, condition=None, **keys):
    """
    Returns (probs[N, C] float64, labels[N] int) or (None, None) if absent.

    condition selects one array from a multi-condition file (e.g. "0.20" for a
    noise level). None takes PROBS_KEY when present, otherwise the single
    non-label array - so single-condition files load without the caller needing
    to know how they were written.

    ROWS ARE RENORMALISED ONLY WHEN THEY NEED IT.

    float16 storage perturbed row sums to 0.999647-1.000352, outside sklearn's
    multi-class tolerance, so roc_auc_score rejected the scores outright.
    Renormalising fixes that for every reader at once.

    But renormalising UNCONDITIONALLY is not free. Dividing by a float32 row sum
    moves each probability by ~1e-7, which is enough to re-break exact ties
    differently than they were broken at training time. AUC is a ranking
    statistic, so on a saturated model - where many samples share an identical
    predicted probability - a few hundred flipped ties shift AUC by ~1e-3. That
    showed up as one run in 1,600 failing the integrity check by 1.62e-03 on a
    file that was already correct.

    So: renormalise only when the worst row sum is outside TOLERANCE. Files
    written in float32 pass untouched and re-score bit-identically; legacy
    float16 files are repaired as before.
    """
    path = pred_path(experiment, **keys)
    if not os.path.exists(path):
        return None, None

    with np.load(path) as z:
        labels = z[LABELS_KEY].astype(int)
        if condition is not None:
            name = str(condition)
        elif PROBS_KEY in z.files:
            name = PROBS_KEY
        else:
            others = [k for k in z.files if k != LABELS_KEY]
            if len(others) != 1:
                return None, None       # ambiguous; caller must name a condition
            name = others[0]
        if name not in z.files:
            return None, None

        probs = z[name].astype(np.float64)
        row = probs.sum(axis=1, keepdims=True)

        # Degenerate all-zero rows cannot be normalised; leave them for the
        # metric to reject rather than dividing by zero.
        bad = (row <= 0).ravel()
        if bad.any():
            row[bad] = 1.0

        if np.abs(row - 1.0).max() > RENORM_TOLERANCE:
            probs = probs / row
        return probs, labels


# ------------------------------------------------------------------ shards
def exists(experiment, **keys) -> bool:
    return os.path.exists(shard_path(experiment, **keys))


def write(experiment, payload: dict, **keys) -> str:
    path = shard_path(experiment, **keys)
    record = {
        "experiment": experiment,
        "keys": keys,
        "git_sha": config.git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **payload,
    }
    tmp = path + ".tmp"                       # atomic: never leave a partial shard
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)
    return path


def load_all(experiment):
    d = os.path.join(config.SHARD_DIR, experiment)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                out.append(json.load(f))
    return out
