"""
In-process recorders for Amendments 20, 21 and 23: keep, for every training
run, the norm of every optimisation step and the trained weights.

Imported by scripts/replay.py and scripts/run_budget_check.py. Not run directly.

WHY THIS EXISTS
---------------
Two things the original runs did not keep, and that the manuscript needs:

1. The pre-clip gradient norm of every STEP. The shards store only the mean
   over each epoch's steps, which cannot show whether a single step ever
   exceeded the clip threshold. Whether clipping acted in a run is a question
   about single steps.
2. The trained weights. Only metrics and per-sample predictions were stored, so
   any question about the trained models themselves (their sensitivity, their
   behaviour on a hardware noise model) needs the model back. Training is
   deterministic on the study's machine (Amendment 21 tests this run by run),
   so re-executing a run regenerates its weights exactly; this keeps them.

HOW, WITHOUT CHANGING ANY STUDY SCRIPT
--------------------------------------
Three wrappers, installed in this process before the study script is
imported, so the script binds to them when it does `from train.loop import
train_model`:

  torch.nn.utils.clip_grad_norm_   records the norm it returns (the pre-clip
                                   norm). The clipping itself is unchanged.
  train.loop.train_model           starts a fresh step list for each run and
                                   keeps the run's AUC-selected trainable
                                   parameters (the third value it returns).
  shards.write                     when the study script writes the run's
                                   shard, saves the step norms and the weights
                                   next to it and adds a "recorded" field to
                                   the shard naming both files and summarising
                                   the steps.

None of the three changes a number the study script computes. The weights are
saved exactly as train_model returned them (float32, on CPU).

FILES
-----
artifacts/step_norms/<namespace>/<shard name>.npz   key "norms": float64 array, one per step
artifacts/weights/<namespace>/<shard name>.pt       dict parameter name -> tensor
"""
import math
import os

import numpy as np
import torch

import config
import shards
import train.loop as loop

THRESHOLD = 20.0          # the clip norm of every reported run
STATE = {"norms": None, "best_state": None, "extra": {}}
_INSTALLED = {"done": False}


def _stem(experiment, keys):
    return shards.shard_name(experiment, **keys)[:-len(".json")]


def install(save_weights=True, extra=None):
    """Install the three wrappers once per process. `extra` is merged into every
    shard's "recorded" field (for example the clip norm and budget in force)."""
    STATE["extra"] = dict(extra or {})
    if _INSTALLED["done"]:
        return
    _INSTALLED["done"] = True

    orig_clip = torch.nn.utils.clip_grad_norm_

    def clip_grad_norm_(parameters, max_norm, *a, **k):
        n = orig_clip(parameters, max_norm, *a, **k)
        if STATE["norms"] is not None:
            STATE["norms"].append(float(n))
        return n
    torch.nn.utils.clip_grad_norm_ = clip_grad_norm_

    orig_train = loop.train_model

    def train_model(*a, **k):
        STATE["norms"], STATE["best_state"] = [], None
        out = orig_train(*a, **k)
        if save_weights and isinstance(out, tuple) and len(out) >= 3 and isinstance(out[2], dict):
            STATE["best_state"] = {n: t.detach().to("cpu").clone() for n, t in out[2].items()}
        return out
    loop.train_model = train_model

    orig_write = shards.write

    def write(experiment, payload, **keys):
        norms = STATE["norms"]
        if norms is not None:
            stem = _stem(experiment, keys)
            sdir = os.path.join(config.ARTIFACT_ROOT, "step_norms", experiment)
            os.makedirs(sdir, exist_ok=True)
            spath = os.path.join(sdir, stem + ".npz")
            np.savez_compressed(spath + ".tmp.npz", norms=np.asarray(norms, np.float64))
            os.replace(spath + ".tmp.npz", spath)
            finite = [x for x in norms if math.isfinite(x)]
            rec = {"step_norms_file": os.path.relpath(spath, config.ARTIFACT_ROOT),
                   "steps": len(norms),
                   "threshold": THRESHOLD,
                   "above_threshold": sum(1 for x in finite if x > THRESHOLD),
                   "max_step_norm": max(finite) if finite else None,
                   "nonfinite": len(norms) - len(finite),
                   "clip_norm_in_force": str(config.GRAD_CLIP_NORM),
                   **STATE["extra"]}
            if STATE["best_state"] is not None:
                wdir = os.path.join(config.ARTIFACT_ROOT, "weights", experiment)
                os.makedirs(wdir, exist_ok=True)
                wpath = os.path.join(wdir, stem + ".pt")
                torch.save(STATE["best_state"], wpath + ".tmp")
                os.replace(wpath + ".tmp", wpath)
                rec["weights_file"] = os.path.relpath(wpath, config.ARTIFACT_ROOT)
                rec["weights_numel"] = int(sum(t.numel() for t in STATE["best_state"].values()))
            payload = dict(payload)
            payload["recorded"] = rec
            STATE["norms"], STATE["best_state"] = None, None
        return orig_write(experiment, payload, **keys)
    shards.write = write
