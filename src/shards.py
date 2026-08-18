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


def save_predictions(experiment, labels, probs, **keys) -> str:
    """
    probs may be:
        ndarray [N, C]                  -> stored under PROBS_KEY
        dict {condition: ndarray[N,C]}  -> one array per condition (03's noise sweep)

    float16 is deliberate: probabilities need ~3 decimal places for AUC and F1 to
    be identical to float64, and it halves ~250 KB per run across thousands of runs.
    """
    labels = np.asarray(labels).astype(np.int16)
    arrays = ({PROBS_KEY: np.asarray(probs, dtype=np.float16)}
              if not isinstance(probs, dict)
              else {str(k): np.asarray(v, dtype=np.float16) for k, v in probs.items()})

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
        return z[name].astype(np.float64), labels


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
