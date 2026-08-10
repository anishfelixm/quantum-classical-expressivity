"""
Result shard I/O.

WHY SHARDS AND NOT ONE BIG JSON
-------------------------------
The previous scripts held everything in memory and rewrote a single JSON at the
end of each dataset loop. A crash at hour 200 of a multi-day sweep lost
everything since the last dataset boundary, and there was no way to resume.

One shard per (experiment, dataset, regime, dim, seed, arm). The driver skips
shards that already exist, so an interrupted sweep resumes by re-running the
same command.

PROVENANCE
----------
Every shard records the git commit that produced it. When a reviewer asks which
code version generated a given table - and for a journal with a code
availability statement, someone will - the answer is exact rather than
approximate.
"""
import json
import os
import time

import config


def shard_name(experiment, dataset, regime, dim, seed, arm, **extra):
    parts = [experiment, dataset, f"r{regime}", f"d{dim}", f"s{seed}", arm]
    parts += [f"{k}{v}" for k, v in sorted(extra.items())]
    return "__".join(str(p).replace("/", "-").replace(" ", "") for p in parts) + ".json"


def shard_path(experiment, **keys):
    d = os.path.join(config.SHARD_DIR, experiment)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, shard_name(experiment, **keys))


def exists(experiment, **keys) -> bool:
    return os.path.exists(shard_path(experiment, **keys))


def write(experiment, payload: dict, **keys) -> str:
    path = shard_path(experiment, **keys)
    record = {
        "experiment": experiment,
        "keys": keys,
        "git_sha": config.git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_seed_policy": "config.set_determinism",
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
