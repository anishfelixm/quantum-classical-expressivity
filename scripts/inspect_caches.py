"""
Read-only inventory of the bootstrap caches the figures will be built from.

Why this exists: the plotting script must read these files exactly as they were
written on the cluster. Rather than assume their layout, print it. Nothing here
writes, recomputes or modifies anything.

Run from the repository root:
    python scripts/inspect_caches.py
"""
import glob
import json
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is not installed in this Python. Activate the environment you use for the repo.")

ROOT = "artifacts"
if not os.path.isdir(ROOT):
    sys.exit("No artifacts/ folder here. Run from the repository root after extracting.")

for sub in ("family_table_cache", "exploratory_cache"):
    files = sorted(glob.glob(os.path.join(ROOT, sub, "*")))
    print(f"\n=== {sub}: {len(files)} files ===")
    for path in files:
        name = os.path.basename(path)
        if not path.endswith(".npz"):
            print(f"  {name}  (not .npz, {os.path.getsize(path)} bytes)")
            continue
        z = np.load(path, allow_pickle=False)
        parts = [f"{k}{tuple(z[k].shape)}" for k in z.files if k != "meta"]
        print(f"  {name:36s} " + "  ".join(parts))
        if "meta" in z.files:
            try:
                meta = json.loads(str(z["meta"]))
                cells = meta.get("cells", [])
                first = cells[0] if cells else None
                other = {k: v for k, v in meta.items() if k != "cells"}
                print(f"      meta: {other}  | {len(cells)} cells, first = {first}")
            except ValueError:
                print(f"      meta (raw): {str(z['meta'])[:200]}")

for name in ("family_table.json", "exploratory_table.json", "lr_selection.json"):
    path = os.path.join(ROOT, name)
    print(f"\n=== {name} ===")
    if not os.path.exists(path):
        print("  missing")
        continue
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"  list of {len(data)} rows; first row:")
        print("  " + json.dumps(data[0])[:600] if data else "  (empty)")
    else:
        print(f"  dict with keys: {list(data)[:30]}")

try:
    import matplotlib
    print(f"\nnumpy {np.__version__} | matplotlib {matplotlib.__version__}")
except ImportError:
    print(f"\nnumpy {np.__version__} | matplotlib NOT INSTALLED")
