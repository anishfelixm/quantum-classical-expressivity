"""
Amendments 21 and 23: re-execute recorded training runs, keeping every step's
gradient norm and the trained weights, and check each re-execution against the
run it repeats.

    python -u scripts/replay.py --source 01_frozen_tuned --quick           # one small run per arm: must be identical
    python -u scripts/replay.py --source 01_frozen_tuned --clip 20 \
        --where arm=quantum_vqc dataset=pathmnist regime=100 --seeds 42 123
    python -u scripts/replay.py --source 01_frozen_tuned --clip inf --from-list runs.txt
    python -u scripts/replay.py --hardware-tuned --where dataset=breastmnist

WHAT A REPLAY IS
----------------
Every shard records the keys of its run (dataset, n, d, seed, arm and every
optional axis), and every study script derives a run entirely from those keys
plus its code. A replay reads a recorded shard, calls the SAME function of the
SAME study script that produced it (`run_cell`), with arguments reconstructed
from the keys, and writes the result to a separate namespace:

    --clip 20   ->  R_<source>   the original protocol. Must reproduce the
                                 recorded run bit for bit; adds the per-step
                                 norms and the weights.
    --clip inf  ->  U_<source>   clipping removed (Amendment 21), only for runs
                                 whose R_ record shows a step above 20.
    --budget 1000:200 -> B_<source>   Amendment 20: MAX_EPOCHS and PATIENCE
                                 raised in-process; clipping stays at 20.

--code-at <commit> runs the code of that commit, from a git worktree in
../replay_trees/, and selects only the runs that commit recorded. Every shard
records its commit, so each run is re-executed by the code that first ran it;
the launcher groups jobs by commit. Without --code-at the current code is used.

After every run the new shard is compared with the recorded one: every number
in the two records (except timings, the commit and the new "recorded" field)
and every stored test probability. The verdict is printed per run and kept in
the new shard under "replay_check". A recorded namespace is never written to:
the target always carries the R_ / U_ prefix, and the script refuses otherwise.

WHY RE-EXECUTION AT THE ORIGINAL PROTOCOL, AND NOT ONLY WITHOUT CLIPPING
-----------------------------------------------------------------------
The question "did clipping act in a reported run?" is about that run's own
trajectory. Re-executing at the original protocol reproduces that trajectory
exactly (if the machine is deterministic, which this checks for every run), so
its step record answers the question for the reported run itself. Only runs in
which clipping did act need an unclipped counterpart. The same re-execution
regenerates the weights of every reported model, which were never stored.

SOURCES AND HOW EACH IS CALLED
------------------------------
  01_frozen_tuned, 14_readout, 15_dim8, 16_dim16, 17_fulldata, 18_depth,
  19_angle, 20_tanh, 21_ansatz, 25_encoder, 26_bottleneck_tuned,
  27_capacity_tuned, 29_compression_frozen
        src/01_frozen_backbone_ablation.py  run_cell(..., experiment=target)
        Keys are inverted exactly: fp, aug, L, notanh, as, lrh, lrq, bn, rank.
        The angle scale is taken from the run's recorded config (the key holds
        it to four decimals only).
  03_robustness_v2   src/03_robustness_evaluation.py  run_cell(..., experiment=target)
  09_lr_selection    src/09_lr_selection.py           run_cell; module EXPERIMENT set to target
  10_capacity        src/10_capacity_sweep.py          run_cell; module EXPERIMENT set to target
  12_bottleneck      src/12_bottleneck_ablation.py     run_cell; module EXPERIMENT set to target

17_fulldata ran on CPU; this machine trains on GPU, so its replay is NOT
expected to be bit-identical (different floating-point order). It is marked
"cross-device" and its identity verdict is reported but not used as a
determinism test.

FEATURE-CACHE STATE (added 16 October 2026, before wave 1)
-------------------------------------------------------------
A run on the cached path that found no feature cache built it first, and
building it constructs a torchvision ResNet-18, whose random initialisation
draws from the same global stream the head is initialised from next. So the
first run to touch a (dataset, n, seed) cache started from a different point
of that stream than a run that found the cache on disk. The original logs show
which runs did (scripts/predict_cold_runs.py -> docs/cold_cache_prediction.json,
written before this re-execution): 699 of the logged runs, among them every
quantum_vqc run of the 30 confirmatory seeds outside ALL_SEEDS (600).

Each run is therefore re-executed first in the cache state the logs predict
("cold": the cache is rebuilt in a private temporary folder for that run, then
deleted; "warm": the cache on disk is read). If that execution is not
identical, the other state is tried once. The record keeps which state
reproduced the run and whether the logs predicted it. On a cold execution the
features rebuilt on this machine are compared bit for bit with the cache file;
if they differed, the file's features (the ones the original run computed and
wrote) are used, and the record says so. The unclipped (U_) re-execution of a
run always uses the state its R_ re-execution established; a budget (B_)
re-execution is checked against the recorded run epoch by epoch over the epochs
both ran (the first E epochs are the same computation), and also falls back to
the other state once. 17_fulldata (CPU original) is always re-executed warm.

--hardware-tuned (Amendment 23): src/07_hardware_noise.py run_cell for every
cell of 07_hardware_noise, with config.LR_HEAD and config.LR_QUANTUM set to the
tuned 1e-2 in-process (07 passes no rate, so it trained at the 1e-3 default),
writing to 33_hardware_tuned. No identity check: the rate differs by design.
"""
import argparse
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TREES = os.path.join(os.path.dirname(REPO), "replay_trees")


def worktree_for(sha):
    """
    A checkout of the commit that produced the runs, created once with
    `git worktree add` beside the repository. The replay imports the study
    scripts from there, so every run is re-executed by the code that first ran
    it. The working tree of the repository itself is never touched.
    """
    path = os.path.join(TREES, sha[:12])
    if not os.path.isdir(os.path.join(path, "src")):
        os.makedirs(TREES, exist_ok=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", path, sha], check=True)
    return path


def _code_root():
    if "--code-at" in sys.argv:
        return worktree_for(sys.argv[sys.argv.index("--code-at") + 1])
    return REPO


# Must happen before config is imported: config derives every artifact path from
# QML_ARTIFACT_ROOT, and a worktree's own default would point into the worktree.
os.environ.setdefault("QML_ARTIFACT_ROOT", os.path.join(REPO, "artifacts"))
CODE_ROOT = _code_root()
SRC = os.path.join(CODE_ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                 # noqa: E402

import config                      # noqa: E402
import shards                      # noqa: E402
import train.loop as loop          # noqa: E402
import recorders                   # noqa: E402
import predict_cold_runs as pcr    # noqa: E402

QUANTUM = ("quantum_vqc", "quantum_reupload", "quantum_rich", "quantum_rich_padded", "quantum_basic")
SCRIPT_01 = "01_frozen_backbone_ablation.py"
SOURCES = {ns: SCRIPT_01 for ns in (
    "01_frozen_tuned", "14_readout", "15_dim8", "16_dim16", "17_fulldata", "18_depth",
    "19_angle", "20_tanh", "21_ansatz", "25_encoder", "26_bottleneck_tuned",
    "27_capacity_tuned", "29_compression_frozen")}
SOURCES.update({"03_robustness_v2": "03_robustness_evaluation.py",
                "09_lr_selection": "09_lr_selection.py",
                "10_capacity": "10_capacity_sweep.py",
                "12_bottleneck": "12_bottleneck_ablation.py"})
CROSS_DEVICE = {"17_fulldata"}
HW_SOURCE, HW_TARGET, TUNED_RATE = "07_hardware_noise", "33_hardware_tuned", 1e-2
VOLATILE = ("time", "timestamp", "git_sha", "experiment", "recorded", "replay_check",
            "wall")


# ------------------------------------------------------------------ helpers
def load_module(fname):
    """Import a study script AFTER the recorders are installed, so it binds to them."""
    name = "study_" + fname.split("_")[0]
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRC, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def strip(x):
    """Record without the fields that legitimately differ between executions."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items()
                if not any(t in k.lower() for t in VOLATILE)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def first_difference(a, b, path=""):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) & set(b)):           # fields both records carry
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (u, v) in enumerate(zip(a, b)):
            d = first_difference(u, v, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def compare_predictions(src_ns, tgt_ns, keys):
    p0 = shards.pred_path(src_ns, **keys)
    p1 = shards.pred_path(tgt_ns, **keys)
    if not os.path.exists(p0) and not os.path.exists(p1):
        return "none stored", 0.0
    if not (os.path.exists(p0) and os.path.exists(p1)):
        return "stored in only one", float("nan")
    with np.load(p0) as a, np.load(p1) as b:
        if sorted(a.files) != sorted(b.files):
            return "different arrays", float("nan")
        worst = 0.0
        for k in a.files:
            x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
            if x.shape != y.shape:
                return f"shape differs in {k}", float("nan")
            worst = max(worst, float(np.abs(x - y).max()) if x.size else 0.0)
    return ("identical" if worst == 0.0 else "different"), worst


def _angle_scale(key_value, cfg):
    """The exact scale used: the recorded config value, else the sweep value the
    four-decimal key stands for (config.ANGLE_SCALE_SWEEP = [pi/2, pi])."""
    if cfg.get("angle_scale") is not None:
        return float(cfg["angle_scale"])
    for v in (math.pi / 2, math.pi):
        if f"{v:.4f}" == str(key_value):
            return v
    raise RuntimeError(f"cannot recover the exact angle scale for key as={key_value}")


def call_args_01(rec):
    """Invert 01's _shard_keys exactly (see its docstring for the rules)."""
    k, m, c = rec["keys"], rec.get("metrics", {}), rec.get("config", {})
    quantum = k["arm"] in QUANTUM
    lrq = float(k["lrq"]) if "lrq" in k else (None if quantum else m.get("lr_quantum"))
    return dict(dataset=k["dataset"], regime=str(k["regime"]), dim=int(k["dim"]),
                seed=int(k["seed"]), arm=k["arm"], freeze_policy=k["fp"],
                augment=bool(int(k["aug"])), force=False,
                n_layers=int(k["L"]) if "L" in k else None,
                use_tanh=(not bool(int(k["notanh"]))) if "notanh" in k else True,
                angle_scale=(_angle_scale(k["as"], c) if "as" in k else None),
                save_predictions=True,
                lr_head=float(k["lrh"]) if "lrh" in k else None,
                lr_quantum=lrq,
                bottleneck=k.get("bn"),
                head_rank=int(k["rank"]) if "rank" in k else None,
                tanh_explicit="notanh" in k)


def run_one(mod, source, target, rec):
    k = rec["keys"]
    fname = SOURCES.get(source)
    if source == HW_SOURCE:
        mod.EXPERIMENT = target
        mod.run_cell(k["dataset"], int(k["regime"]), int(k["seed"]), arm=k["arm"],
                     dim=int(k["dim"]), n_shot_eval=500)
    elif fname == SCRIPT_01:
        args = call_args_01(rec)
        accepted = inspect.signature(mod.run_cell).parameters
        if args["tanh_explicit"] and "tanh_explicit" not in accepted:
            raise RuntimeError(f"{k} carries a notanh key but this commit's run_cell has no tanh_explicit")
        args = {a: v for a, v in args.items() if a in accepted}
        if hasattr(mod, "_shard_keys"):
            kp = inspect.signature(mod._shard_keys).parameters
            ka = {"regime": args["regime"], "dataset": args["dataset"], "dim": args["dim"],
                  "seed": args["seed"], "arm": args["arm"], "freeze_policy": args["freeze_policy"],
                  "augment": args["augment"], "n_layers": args["n_layers"],
                  "use_tanh": args["use_tanh"], "angle_scale": args["angle_scale"],
                  "lr_head": args["lr_head"], "lr_quantum": args["lr_quantum"],
                  "bottleneck": args.get("bottleneck"), "head_rank": args.get("head_rank"),
                  "tanh_explicit": args.get("tanh_explicit", False)}
            expect = mod._shard_keys(**{a: v for a, v in ka.items() if a in kp})
            if shards.shard_name("x", **expect) != shards.shard_name("x", **k):
                raise RuntimeError(f"key inversion failed for {k}: would write {expect}")
        mod.run_cell(**args, experiment=target)
    elif fname == "03_robustness_evaluation.py":
        # 03's main() passed lr_head=lr AND lr_quantum=lr for every arm, classical
        # ones included (the quantum group is empty there, so the value has no
        # effect on training), and the record stores both under "lr". The key
        # carries lrq for circuit arms only, so the rates are taken from the
        # record's "lr" field: exactly what the original call received.
        lr = rec.get("lr") or {}
        lrh = lr.get("head") if "head" in lr else (float(k["lrh"]) if "lrh" in k else None)
        lrq = lr.get("quantum") if "quantum" in lr else (float(k["lrq"]) if "lrq" in k else None)
        mod.run_cell(k["dataset"], int(k["regime"]), int(k["seed"]), k["arm"], dim=int(k["dim"]),
                     lr_head=lrh, lr_quantum=lrq, experiment=target)
    elif fname == "09_lr_selection.py":
        mod.EXPERIMENT = target
        mod.run_cell(k["dataset"], int(k["regime"]), int(k["seed"]), k["arm"], float(k["lr"]),
                     dim=int(k["dim"]))
    elif fname == "10_capacity_sweep.py":
        mod.EXPERIMENT = target
        mod.run_cell(k["dataset"], int(k["regime"]), int(k["seed"]), int(k["rank"]),
                     dim=int(k["dim"]),
                     weight_decay=float(k["wd"]) if "wd" in k else None)
    elif fname == "12_bottleneck_ablation.py":
        mod.EXPERIMENT = target
        mod.run_cell(k["dataset"], int(k["regime"]), int(k["seed"]), k["arm"], k["bn"],
                     dim=int(k["dim"]))
    else:
        raise RuntimeError(f"no adapter for {source}")
    path = shards.shard_path(target, **k)
    if not os.path.exists(path):
        raise RuntimeError(f"replay of {k} did not write {path}: keys differ from the record")
    return path


# ------------------------------------------------------------------ feature-cache state
COLD = {"target": None, "info": None}


def install_cold_cache(mod):
    """
    Wrap get_cached_features so that, when COLD["target"] names the run's own
    (dataset, regime, seed), that one call behaves as it did for the run that
    built the cache: the cache is built from scratch (into a private temporary
    folder, so the shared cache is never written or deleted), which constructs
    the ResNet and so advances the random stream exactly as the original call
    did. The rebuilt features are then compared bit for bit with the shared
    cache file; if they differ, the file's features - the ones the original run
    computed and saved - replace them. Every other call is passed through.
    """
    import torch
    targets = [mod]
    m01 = sys.modules.get("01_frozen_backbone_ablation")
    if m01 is not None and m01 is not mod:
        targets.append(m01)
    targets = [t for t in targets if hasattr(t, "get_cached_features")]
    if not targets:
        return False
    orig = targets[0].get_cached_features

    def get_cached_features(dataset, regime, seed, force=False):
        want = COLD["target"]
        if want is None or (dataset, str(regime), int(seed)) != want:
            return orig(dataset, regime, seed, force)
        COLD["target"] = None                         # only the run's own first call
        tag = f"{dataset}__r{regime}__s{seed}.pt"
        shared = config.FEATURE_CACHE
        stored = os.path.join(shared, tag)
        tmp_root = os.path.join(config.ARTIFACT_ROOT, "tmp_cold")
        os.makedirs(tmp_root, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="cold_", dir=tmp_root)
        config.FEATURE_CACHE = tmp
        try:
            loaders, meta = orig(dataset, regime, seed, force)
        finally:
            config.FEATURE_CACHE = shared
            shutil.rmtree(tmp, ignore_errors=True)
        info = {"cache_file": tag, "cache_file_exists": os.path.exists(stored)}
        if info["cache_file_exists"]:
            blob = torch.load(stored, map_location=config.DEVICE, weights_only=True)
            same = True
            for name in ("train", "val", "test"):
                ld = loaders[name]
                fx, fy = blob[f"{name}_x"].to(ld.feats.device), blob[f"{name}_y"].to(ld.labels.device)
                if not (ld.feats.shape == fx.shape and torch.equal(ld.feats, fx)
                        and torch.equal(ld.labels, fy)):
                    same = False
            info["features_rebuilt_identical"] = same
            if not same:
                for name in ("train", "val", "test"):
                    ld = loaders[name]
                    ld.feats = blob[f"{name}_x"].to(ld.feats.device)
                    ld.labels = blob[f"{name}_y"].to(ld.labels.device)
                info["features_used"] = "cache file (rebuilt features differed)"
            else:
                info["features_used"] = "rebuilt (identical to the cache file)"
        else:
            info["features_used"] = "rebuilt (no cache file to compare)"
        COLD["info"] = info
        return loaders, meta

    for t in targets:
        t.get_cached_features = get_cached_features
    return True


def history_of(r):
    h = r.get("history")
    if not isinstance(h, dict):
        h = (r.get("metrics") or {}).get("history") if isinstance(r.get("metrics"), dict) else None
    return h if isinstance(h, dict) else {}


def epochs_of(r):
    for d in (r, r.get("metrics") if isinstance(r.get("metrics"), dict) else {}, r.get("test_metrics") or {}):
        if isinstance(d, dict) and d.get("epochs_run") is not None:
            return int(d["epochs_run"])
    return None


def prefix_compare(ref, new, n_epochs):
    """Every per-epoch series both records carry must agree bit for bit over the
    first n_epochs epochs."""
    hr, hn = history_of(ref), history_of(new)
    if not hr or not hn:
        return "not compared (no per-epoch history in the record)", None
    for key in sorted(set(hr) & set(hn)):
        a, b = hr[key], hn[key]
        if not isinstance(a, list) or not isinstance(b, list):
            continue
        if len(a) < n_epochs or len(b) < n_epochs:
            return "PREFIX DIFFERENT", f"/{key}: length {len(a)} vs {len(b)}, need {n_epochs}"
        d = first_difference(a[:n_epochs], b[:n_epochs], f"/{key}")
        if d:
            return "PREFIX DIFFERENT", d
    return f"PREFIX IDENTICAL ({n_epochs} epochs)", None


def check_budget(source, rec, new):
    """B_: the first E epochs of a 1,000-epoch run repeat the E-epoch run exactly
    (same seed, data order and initialisation; the scheduler's patience is fixed
    at 15 and does not depend on the budget). E = the reference run's epochs."""
    ref = rec
    if source in CROSS_DEVICE:
        path = shards.shard_path("R_" + source, **rec["keys"])
        if not os.path.exists(path):
            return {"verdict": "not compared (no R_ reference on this machine yet)"}
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
    E = epochs_of(ref)
    if not E:
        return {"verdict": "not compared (reference has no epoch count)"}
    v, d = prefix_compare(ref, new, E)
    return {"verdict": v, "first_difference": d,
            "reference": "R_" + source if ref is not rec else source}


def check_unclipped(source, rec, new):
    """U_: identical to the R_ re-execution up to the first epoch in which a step
    exceeded 20 (clip_grad_norm_ scales by exactly 1.0 below the threshold, so the
    two computations coincide until then)."""
    import numpy as np_
    path = shards.shard_path("R_" + source, **rec["keys"])
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)
    rr = ref.get("recorded", {})
    sfile = os.path.join(config.ARTIFACT_ROOT, rr.get("step_norms_file", ""))
    E = epochs_of(ref)
    if not rr.get("step_norms_file") or not os.path.exists(sfile) or not E:
        return {"verdict": "not compared (R_ step record or epoch count missing)"}
    with np_.load(sfile) as z:
        norms = z["norms"]
    if len(norms) % E:
        return {"verdict": f"not compared ({len(norms)} steps is not a multiple of {E} epochs)"}
    spe = len(norms) // E
    bad = [i for i, x in enumerate(norms) if not np_.isfinite(x) or x > recorders.THRESHOLD]
    if not bad:
        return {"verdict": "not compared (no step above 20 in the R_ run)"}
    e_star = bad[0] // spe
    if e_star == 0:
        return {"verdict": "PREFIX IDENTICAL (0 epochs: a step above 20 in epoch 1)",
                "first_clipped_epoch": 0}
    v, d = prefix_compare(ref, new, e_star)
    return {"verdict": v, "first_difference": d, "first_clipped_epoch": e_star,
            "reference": "R_" + source}


def check(source, target, rec, path):
    with open(path, encoding="utf-8") as f:
        new = json.load(f)
    if source == HW_SOURCE:
        return new, {"verdict": "not compared (different learning rate by design)"}
    if target.startswith("B_"):
        v = check_budget(source, rec, new)
        v.update({"replayed_at": config.git_sha(), "recorded_at": rec.get("git_sha")})
        return new, v
    if target.startswith("U_"):
        v = check_unclipped(source, rec, new)
        v.update({"replayed_at": config.git_sha(), "recorded_at": rec.get("git_sha")})
        return new, v
    a, b = strip(rec), strip(new)
    diff = first_difference(a, b)
    only = sorted(set(a) ^ set(b))
    pred, worst = compare_predictions(source, target, rec["keys"])
    same = diff is None and pred in ("identical", "none stored")
    verdict = "IDENTICAL" if same else "DIFFERENT"
    if source in CROSS_DEVICE:
        verdict += " (cross-device: CPU original, GPU replay)"
    return new, {"verdict": verdict, "first_difference": diff, "predictions": pred,
                 "max_abs_prob_diff": worst, "top_level_fields_in_only_one": only,
                 "replayed_at": config.git_sha(), "recorded_at": rec.get("git_sha")}


def select(source, where, seeds, from_list, code_at=None):
    rs = [r for r in shards.load_all(source)]
    if code_at:
        rs = [r for r in rs if str(r.get("git_sha", "")).startswith(code_at)]
    if from_list:
        with open(from_list, encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        rs = [r for r in rs if shards.shard_name(source, **r["keys"]) in wanted]
    for cond in where:
        key, val = cond.split("=", 1)
        rs = [r for r in rs if str(r["keys"].get(key)) == val]
    if seeds:
        rs = [r for r in rs if int(r["keys"]["seed"]) in set(seeds)]
    return sorted(rs, key=lambda r: shards.shard_name(source, **r["keys"]))


def quick_set(rs):
    """One cheap run per configuration: smallest n, smallest dataset, first seed."""
    order_ds = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
    groups = {}
    for r in rs:
        k = r["keys"]
        g = tuple(sorted((a, str(b)) for a, b in k.items() if a not in ("dataset", "regime", "seed")))
        n = int(k["regime"]) if str(k["regime"]).isdigit() else 10 ** 9
        rank = (n, order_ds.index(k["dataset"]) if k["dataset"] in order_ds else 9,
                config.ALL_SEEDS.index(int(k["seed"])) if int(k["seed"]) in config.ALL_SEEDS else 99)
        if g not in groups or rank < groups[g][0]:
            groups[g] = (rank, r)
    return [v[1] for v in groups.values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=sorted(SOURCES))
    p.add_argument("--hardware-tuned", action="store_true")
    p.add_argument("--clip", choices=["20", "inf"], default="20")
    p.add_argument("--budget", default=None, metavar="MAX_EPOCHS:PATIENCE",
                   help="Amendment 20: re-execute with a larger training budget "
                        "(clip stays 20); target B_<source>")
    p.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--from-list", default=None, help="file of shard names (one per line)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--gate", action="store_true",
                   help="print the QUICK CHECK verdict line for the selected runs")
    p.add_argument("--cache", choices=["auto", "warm", "cold"], default="auto",
                   help="feature-cache state (auto: as the original logs predict, then the other)")
    p.add_argument("--code-at", default=None,
                   help="re-execute with the code of this commit (a git worktree); only runs "
                        "recorded at this commit are selected")
    args = p.parse_args()

    if args.hardware_tuned:
        source, target = HW_SOURCE, HW_TARGET
        config.LR_HEAD = config.LR_QUANTUM = TUNED_RATE
        assert loop.config is config
        fname = "07_hardware_noise.py"
        clip = 20.0
    else:
        if not args.source:
            p.error("--source or --hardware-tuned is required")
        source = args.source
        clip = 20.0 if args.clip == "20" else float("inf")
        target = ("R_" if args.clip == "20" else "U_") + source
        fname = SOURCES[source]
        if args.budget:
            if args.clip != "20":
                p.error("--budget varies the budget only; clipping stays at 20")
            me, pa = (int(x) for x in args.budget.split(":"))
            config.MAX_EPOCHS, config.PATIENCE = me, pa
            assert loop.config is config
            target = "B_" + source
    assert target != source and (target.startswith(("R_", "U_", "B_")) or target == HW_TARGET)
    if clip != config.GRAD_CLIP_NORM:
        config.GRAD_CLIP_NORM = clip
    assert loop.config is config and config.GRAD_CLIP_NORM == clip

    recorders.install(save_weights=True,
                      extra={"replay_of": source, "clip_norm": str(clip),
                             "max_epochs": config.MAX_EPOCHS, "patience": config.PATIENCE})
    mod = load_module(fname)

    if args.hardware_tuned and args.code_at:
        p.error("--hardware-tuned runs the current code; --code-at does not apply")
    rs = select(source, args.where, args.seeds, args.from_list,
                None if args.hardware_tuned else args.code_at)
    if args.quick:
        rs = quick_set(rs)
    print(f"replay | source={source} -> {target} | clip={clip} | runs={len(rs)} | "
          f"code={config.git_sha()[:8]} ({CODE_ROOT}) | device={config.DEVICE}", flush=True)

    cold_ok = (not args.hardware_tuned) and install_cold_cache(mod)
    table = pcr.load_table()
    kind = "H" if args.hardware_tuned else target[0]          # R, U or B

    def ok(v):
        x = v.get("verdict", "")
        if kind == "R":
            return x.startswith("IDENTICAL")
        if kind == "B":
            return not x.startswith("PREFIX DIFFERENT")
        return True

    def attempt(rec, cache):
        k = rec["keys"]
        path = shards.shard_path(target, **k)
        if os.path.exists(path):
            os.remove(path)                    # run_cell skips a run whose shard exists
        COLD["target"] = ((k["dataset"], str(k["regime"]), int(k["seed"]))
                          if cache == "cold" else None)
        COLD["info"] = None
        try:
            run_one(mod, source, target, rec)
        finally:
            unused = COLD["target"] is not None
            COLD["target"] = None
        new, v = check(source, target, rec, path)
        v["cache"] = cache
        if cache == "cold":
            v["cold_info"] = COLD["info"] if not unused else "the run never read its feature cache"
        return path, new, v

    def conditions(rec):
        """Cache states to try, in order (see FEATURE-CACHE STATE above)."""
        if kind == "H" or not cold_ok or source in CROSS_DEVICE:
            return ["warm"]
        if kind == "U":
            rpath = shards.shard_path("R_" + source, **rec["keys"])
            if not os.path.exists(rpath):
                raise RuntimeError(f"U_ needs the R_ re-execution first: {rpath}")
            with open(rpath, encoding="utf-8") as f:
                return [json.load(f).get("replay_check", {}).get("cache", "warm")]
        if args.cache != "auto":
            return [args.cache]
        first = pcr.predicted(table, source, rec) or "warm"
        return [first, "cold" if first == "warm" else "warm"]

    n_same = n_diff = 0
    t0 = time.time()
    for i, rec in enumerate(rs, 1):
        k = rec["keys"]
        path = shards.shard_path(target, **k)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done = json.load(f)
            rc = done.get("replay_check", {})
            v = rc.get("verdict", "?")
            # A result of the v57 code that differed was produced before the cache
            # state was handled (or, for 03, with the wrong rates): redo it.
            stale = "cache" not in rc and (not ok(rc) or v.startswith("not compared (different budget"))
            if not stale:
                print(f"[{i}/{len(rs)}] exists  {v}  {shards.shard_name(target, **k)}", flush=True)
                n_same += ok(rc)
                n_diff += not ok(rc)
                continue
        pred = pcr.predicted(table, source, rec)
        tried = []
        for cache in conditions(rec):
            path, new, verdict = attempt(rec, cache)
            tried.append({"cache": cache, "verdict": verdict.get("verdict"),
                          "first_difference": verdict.get("first_difference")})
            if ok(verdict):
                break
        verdict["predicted_cache"] = pred
        verdict["attempts"] = tried
        new["replay_check"] = verdict
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new, f, indent=2)
        os.replace(tmp, path)
        good = ok(verdict)
        n_same += good
        n_diff += not good
        rr = new.get("recorded", {})
        print(f"[{i}/{len(rs)}] {verdict['verdict'][:28]:28s} cache={verdict['cache']:4s} "
              f"(logs: {pred or '-'}) steps>20={rr.get('above_threshold')} "
              f"max={rr.get('max_step_norm') or float('nan'):.3f} "
              f"{'' if good else verdict.get('first_difference')} "
              f"| {k['dataset']} n={k['regime']} s={k['seed']} {k.get('arm', '')} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)

    label = {"R": "identical", "B": "prefix identical or not comparable", "U": "done",
             "H": "done"}[kind]
    print(f"\nreplay summary {source} -> {target}: {label} {n_same}, "
          f"{'different' if kind in 'RB' else 'other'} {n_diff}, of {len(rs)}")
    if args.quick or args.gate:
        if kind == "H":
            print(f"QUICK CHECK PASSED (smoke test: {len(rs)} runs completed and recorded; "
                  f"no identity expected)")
        else:
            bad = (n_diff > 0 and source not in CROSS_DEVICE) or (n_same + n_diff) < len(rs) \
                or len(rs) == 0
            print("QUICK CHECK " + ("FAILED: a re-execution at the original protocol differs from "
                                    "its record. Do not launch; send this output."
                                    if bad else "PASSED"))


if __name__ == "__main__":
    main()
