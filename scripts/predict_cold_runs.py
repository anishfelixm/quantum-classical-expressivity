"""
Prediction, from the original run logs, of which recorded runs built their own
feature cache ("cold" runs). Written 16 October 2026, before any run is
re-executed under this prediction.

    python scripts/predict_cold_runs.py            # writes docs/cold_cache_prediction.json

Standard library only. Reads the committed run logs (logs_*.txt); writes one
JSON file. Run from the repository root.

WHY THIS MATTERS
----------------
Every frozen-backbone run reads its 256-d features from a cache file
(01.get_cached_features: artifacts/feature_cache/<dataset>__r<n>__s<seed>.pt).
When the file does not exist yet, the same function builds it: it loads the
images and constructs the ResNet-18 (models/backbone.py,
torchvision.models.resnet18). Constructing a torchvision ResNet initialises
its layers randomly before the pretrained weights are loaded over them, and
that initialisation draws from the global PyTorch random stream - the same
stream the head's own initialisation draws from immediately afterwards
(registry.build_arm does not re-seed it). So the first run that touched a
given (dataset, n, seed) cache started its head from a different, equally
random, point of the stream than a run that found the cache already on disk.

Nothing about the run is wrong: its head is initialised from the seeded stream
like every other run, and nothing about the draw depends on the data or the
arm. But a re-execution only reproduces it if the cache is absent again at
that moment. Replay therefore needs to know which runs were cache builders.

HOW THE LOGS SHOW IT
--------------------
Building a cache calls get_loaders, which makes medmnist print
"Using downloaded and verified file" three times (train, val, test). A run on
the cached path that is preceded by those three lines built its cache; one
preceded by none did not. Runs that never use the cache (the noise sweep, which
evaluates on images; layer3-trainable runs) load images every time and print
three lines regardless: for those, six lines would mean a cache build.

OUTPUT
------
docs/cold_cache_prediction.json:
    {"written": ..., "rule": ..., "per_log": {log: {source, commit, runs, cold}},
     "runs": [[source, commit8, dataset, regime, dim, seed, arm, lr, fp, cold, log], ...]}
dim, lr and fp are null where the log line does not show them. Every run line
of every log of a re-executed source is listed, so a record with no log line is
distinguishable from a record predicted warm.
"""
import glob
import json
import os
import re
import time

SOURCES = {"01_frozen_tuned", "03_robustness_v2", "09_lr_selection", "10_capacity",
           "12_bottleneck", "14_readout", "15_dim8", "16_dim16", "17_fulldata", "18_depth",
           "19_angle", "20_tanh", "21_ansatz", "25_encoder", "26_bottleneck_tuned",
           "27_capacity_tuned", "29_compression_frozen"}
# Logs whose header does not name the namespace.
BY_FILE = {"logs_lr.txt": "09_lr_selection", "logs_noise_v2.txt": "03_robustness_v2",
           "logs_bottleneck2.txt": "12_bottleneck", "logs_capacity.txt": "10_capacity",
           "logs_capacity2.txt": "10_capacity"}
LOAD = "Using downloaded and verified file"
SHA = re.compile(r"sha=([0-9a-f]{8})")
NS = re.compile(r"ns=(\S+)")
R01 = re.compile(r"^\[\d+/\d+\]\s+(\w+)\s+r=(\S+)\s+d=(\d+)\s+(froz|adap)\s+s=(\d+)\s+(\S+)")
R09 = re.compile(r"^\[\d+/\d+\]\s+(\w+)\s+n=(\S+)\s+s=(\d+)\s+(\S+)\s+lr=(\S+)")
R12 = re.compile(r"^\[\d+/\d+\]\s+(\w+)\s+n=(\S+)\s+s=(\d+)\s+bn=(\S+)\s+(\S+)")
R10 = re.compile(r"^\[\d+/\d+\]\s+(\w+)\s+n=(\S+)\s+s=(\d+)\s+rank=(\S+)")
R03 = re.compile(r"^\[\d+/\d+\]\s+(\w+)\s+n=(\S+)\s+s=(\d+)\s+(\S+)\s+auc ")
RUN = re.compile(r"^\[\d+/\d+\]")
FP = {"froz": "all", "adap": "layer3_only"}


PREDICTION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "docs", "cold_cache_prediction.json")


def load_table(path=PREDICTION_FILE):
    """Index of the prediction: (source, dataset, regime, seed) -> list of log entries."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["runs"]
    table = {}
    for r in rows:
        ns, sha, ds, reg, dim, seed, arm, lr, fp, cold, log = r
        table.setdefault((ns, ds, str(reg), int(seed)), []).append(
            {"sha": sha, "dim": dim, "arm": arm, "lr": lr, "fp": fp, "cold": cold, "log": log})
    return table


def predicted(table, source, rec):
    """
    "cold" or "warm" when the original logs show the run; None when no log line
    matches it, or when matching lines disagree (the same keys run twice under
    different settings). Matching uses every field the log line shows.
    """
    k = rec["keys"]
    regime = k.get("regime", k.get("n"))
    try:
        cand = table.get((source, k["dataset"], str(regime), int(k["seed"])), [])
    except (KeyError, ValueError):
        return None
    if "arm" in k:
        cand = [c for c in cand if c["arm"] == k["arm"]]
    if "dim" in k:
        cand = [c for c in cand if c["dim"] is None or int(c["dim"]) == int(k["dim"])]
    if "lr" in k:
        cand = [c for c in cand if c["lr"] is None or c["lr"] == str(k["lr"])]
    if "fp" in k:
        cand = [c for c in cand if c["fp"] is None or c["fp"] == k["fp"]]
    sha = str(rec.get("git_sha", ""))[:8]
    same = [c for c in cand if c["sha"] == sha]
    cand = same or cand
    flags = {c["cold"] for c in cand}
    if len(flags) != 1:
        return None
    return "cold" if flags.pop() else "warm"


def main():
    rows, per_log, unclear = [], {}, []
    for path in sorted(glob.glob("logs_*.txt")):
        log = os.path.basename(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        ns = BY_FILE.get(log)
        sha = None
        loads = 0
        for line in lines:
            s = line.strip()
            if NS.search(s) and "cells" in s:
                ns = NS.search(s).group(1)
            if SHA.search(s) and ("cells" in s or s.startswith("device=")):
                sha = SHA.search(s).group(1)
            if s.startswith(LOAD):
                loads += 1
                continue
            if not RUN.match(s):
                continue
            n_loads, loads = loads, 0
            if ns not in SOURCES or log == "logs_noise.txt":
                continue
            dim = lr = fp = None
            m = None
            if ns == "09_lr_selection":
                m = R09.match(s)
                if m:
                    ds, reg, seed, arm, lr = m.groups()
                    dim = 4
            elif ns == "03_robustness_v2":
                m = R03.match(s)
                if m:
                    ds, reg, seed, arm = m.groups()
                    dim = 4
            elif ns == "12_bottleneck":
                m = R12.match(s)
                if m:
                    ds, reg, seed, _bn, arm = m.groups()
                    dim = 4
            elif ns == "10_capacity":
                m = R10.match(s)
                if m:
                    ds, reg, seed, _rank = m.groups()
                    arm, dim = "low_rank", 4
            else:
                m = R01.match(s)
                if m:
                    ds, reg, dim, fpw, seed, arm = m.groups()
                    dim, fp = int(dim), FP[fpw]
            if not m:
                unclear.append((log, s[:100]))
                continue
            # image-path runs load the images every time (3 lines); a cache build adds 3
            image_path = ns == "03_robustness_v2" or fp == "layer3_only"
            base = 3 if image_path else 0
            if n_loads not in (base, base + 3):
                unclear.append((log, f"{n_loads} load lines before: {s[:80]}"))
            cold = n_loads >= base + 3
            rows.append([ns, sha, ds, str(reg), dim, int(seed), arm, lr, fp, cold, log])
            d = per_log.setdefault(log, {"source": ns, "commit": sha, "runs": 0, "cold": 0})
            d["runs"] += 1
            d["cold"] += cold
    out = {"written": time.strftime("%Y-%m-%d %H:%M"),
           "rule": "cold = a cached-path run preceded by three data-load lines "
                   "(image-path runs: six)",
           "per_log": per_log, "unparsed": unclear, "runs": rows}
    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", "cold_cache_prediction.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    print(f"{'log':36s} {'source':24s} {'commit':9s} {'runs':>6s} {'cold':>6s}")
    for log, d in per_log.items():
        print(f"{log:36s} {d['source']:24s} {str(d['commit']):9s} {d['runs']:6d} {d['cold']:6d}")
    print(f"total runs {len(rows)}, predicted cold {sum(r[9] for r in rows)}, "
          f"unparsed lines {len(unclear)}")
    for u in unclear[:20]:
        print("  unparsed:", u)


if __name__ == "__main__":
    main()
