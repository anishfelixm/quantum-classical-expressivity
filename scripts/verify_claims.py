"""
Check, against the run records and the data on the pod, every statement in the
manuscript that cannot be checked from the code or the committed logs alone.
No training. Run from the repository root in qml_v2:

    python scripts/verify_claims.py                       # sections A-E (minutes)
    python scripts/verify_claims.py --sections F --workers 8
                                                          # F: the analysis code re-run on
                                                          # real data (about an hour)

Writes logs_verify_claims_<sections>.txt in the current directory. Each line
states the claim as the manuscript makes it and the value found. The ledger
(docs/CLAIMS_LEDGER.md) says which claim each line settles.

A  Environment: versions, GPU and partition, CPUs, free disk.
B  Data: official split sizes and per-class counts; test-set sizes as stored
   in the predictions; validation and training counts per n as recorded in the
   runs (min(2n, available) per class; exactly n per class); the full-data
   training sizes and the PathMNIST subsample per class.
C  Training records: the early-stopping floor per dataset and n; the mean
   pre-clip gradient norms at the default rate 1e-3 per arm and dataset (the
   source of Amendment 3's "0.48-0.76 / 0.97-1.39 / 1.27-2.55"), from the
   diagnostic runs (01_frozen) and from the tuning runs (09_lr_selection).
D  Searches for the records behind the clip threshold's stated basis (9.62,
   15.87, deep_funnel): every shard folder including archived ones, every text
   file and log under the home folder, and the listings of every archive.
E  Single numbers: the bootstrap standard error of one test AUC on
   BreastMNIST; whether the noise sweep's heads (03_robustness_v2) are the
   primary runs' heads (clean AUC against 01_frozen_tuned); the shot-noise
   retention per run as well as per cell (07).
F  The analysis code on real data: the primary, re-uploading, function-class,
   E7 and re-upload-vs-matched contrasts recomputed from scratch (no cache)
   by scripts/paired_conditions.py, which Amendments 20-21 use. The first
   three must reproduce artifacts/family_table.json exactly; E7 and the
   re-upload-vs-matched values must equal the published ones.
"""
import argparse
import collections
import glob
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LINES = []
DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        return f"(failed: {e})"


def section_a(cfg):
    import torch
    out("=== A. Environment ===")
    out(f"python {platform.python_version()}")
    out(f"torch {torch.__version__}, cuda available {torch.cuda.is_available()}")
    for mod in ("pennylane", "numpy", "medmnist", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            out(f"{mod} {m.__version__}")
        except Exception as e:                                # noqa: BLE001
            out(f"{mod}: {e}")
    out("GPU: " + sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))
    out("MIG: " + (sh("nvidia-smi -L") or "(none)"))
    out(f"CPUs: {os.cpu_count()}")
    t, u, f = shutil.disk_usage(cfg.ARTIFACT_ROOT)
    out(f"disk at {cfg.ARTIFACT_ROOT}: {f / 1e9:.1f} GB free of {t / 1e9:.1f} GB")


def section_b(cfg, shards):
    import medmnist
    out("\n=== B. Data ===")
    out("Claim: official split sizes Breast 546/78/156, Pneumonia 4,708/524/624, "
        "Blood 11,959/1,712/3,421, Path 89,996/10,004/7,180")
    avail = {}
    for ds in DATASETS:
        info = medmnist.INFO[ds]
        Cls = getattr(medmnist, info["python_class"])
        sizes, per = [], {}
        for split in ("train", "val", "test"):
            d = Cls(split=split, download=False, root=cfg.DATA_CACHE)
            lab = np.asarray(d.labels).ravel()
            sizes.append(len(lab))
            per[split] = np.bincount(lab, minlength=len(info["label"])).tolist()
        avail[ds] = per
        out(f"  {ds:15s} train/val/test = {sizes[0]}/{sizes[1]}/{sizes[2]}; "
            f"val per class {per['val']}")
    out("Claim: every class exactly n in training; validation min(2n, available) per class")
    bad_tr = bad_va = checked = 0
    for r in shards.load_all("01_frozen_tuned"):
        m = r.get("meta", {})
        k = r["keys"]
        n = int(k["regime"])
        tr, va = m.get("train_class_counts"), m.get("val_class_counts")
        if tr is None or va is None:
            continue
        checked += 1
        bad_tr += any(c != n for c in tr)
        bad_va += any(c != min(2 * n, a) for c, a in zip(va, avail[k["dataset"]]["val"]))
    out(f"  01_frozen_tuned runs checked {checked}: training counts not n in {bad_tr}; "
        f"validation counts not min(2n, available) in {bad_va}")
    out("Claim: test sets used unchanged (156 ... 7,180)")
    for ds in DATASETS:
        rs = [r for r in shards.load_all("01_frozen_tuned") if r["keys"]["dataset"] == ds][:1]
        if rs:
            p, y = shards.load_predictions("01_frozen_tuned", **rs[0]["keys"])
            out(f"  {ds:15s} stored test labels: {len(y)}")
    out("Claim: full data = complete train and val splits; PathMNIST train subsampled to 10,000 "
        "in proportion to class frequency")
    for r in shards.load_all("17_fulldata"):
        if r["keys"]["arm"] == "linear" and int(r["keys"]["seed"]) == 42:
            m = r.get("meta", {})
            out(f"  {r['keys']['dataset']:15s} n_train {m.get('n_train')}  n_val {m.get('n_val')}  "
                f"train per class {m.get('train_class_counts')}")


def section_c(cfg, shards):
    out("\n=== C. Training records ===")
    out("Early-stopping floor min(max(20, 200 // B), 50), B = ceil(n_train / 32):")
    for ds, C in (("breast/pneumonia", 2), ("bloodmnist", 8), ("pathmnist", 9)):
        row = []
        for n in (5, 10, 20, 50, 100):
            b = (n * C + 31) // 32
            row.append(f"n={n}: {min(max(20, 200 // b), 50)}")
        out(f"  {ds:17s} " + "  ".join(row))
    out("Amendment 3 (26 Aug): 'quantum_vqc 0.48-0.76; linear 0.97-1.39; fourier_rff 1.27-2.55; "
        "deep_funnel 2.88-4.79' (mean pre-clip norms at d=4, before tuning)")
    for ns, rate_ok in (("01_frozen", lambda r: r["keys"].get("lrh") is None),
                        ("09_lr_selection", lambda r: str(r["keys"].get("lr")) == "1e-03")):
        acc = collections.defaultdict(list)
        for r in shards.load_all(ns):
            k = r["keys"]
            if k.get("fp", "all") != "all" or not rate_ok(r):
                continue
            g = (r.get("history") or {}).get("pre_clip_grad_norm")
            if g:
                v = [x for x in g if x is not None]
                if v:
                    acc[(k["arm"], k["dataset"], str(k["regime"]))].append(float(np.mean(v)))
            elif r.get("mean_grad_norm") is not None:
                acc[(k["arm"], k["dataset"], str(k["regime"]))].append(r["mean_grad_norm"])
        out(f"  {ns} (rate 1e-3, frozen backbone):")
        for arm in sorted({a for a, _, _ in acc}):
            per_ds = {ds: statistics.mean([x for (a, d, _), v in acc.items() if a == arm and d == ds for x in v])
                      for ds in DATASETS if any(a == arm and d == ds for a, d, _ in acc)}
            per_n = collections.defaultdict(list)
            for (a, d, n), v in acc.items():
                if a == arm:
                    per_n[n] += v
            allv = [x for (a, _, _), v in acc.items() if a == arm for x in v]
            out(f"    {arm:24s} overall {statistics.mean(allv):.3f}; per dataset "
                + ", ".join(f"{d[:5]} {v:.2f}" for d, v in per_ds.items())
                + f" (range {min(per_ds.values()):.2f}-{max(per_ds.values()):.2f}); per n "
                + ", ".join(f"{n} {statistics.mean(v):.2f}" for n, v in sorted(per_n.items(), key=lambda t: int(t[0]) if t[0].isdigit() else 999)))


def section_d(cfg):
    out("\n=== D. Searching for the records behind 9.62 / 15.87 / deep_funnel ===")
    hits = 0
    for root in sorted({cfg.ARTIFACT_ROOT, os.path.expanduser("~")}):
        for dp, dns, fns in os.walk(root):
            if "/proc" in dp or "replay_trees" in dp or "/.cache" in dp or "/feature_cache" in dp:
                continue
            for fn in fns:
                path = os.path.join(dp, fn)
                if "deep_funnel" in fn:
                    out(f"  file name: {path}")
                    hits += 1
                if fn.endswith((".txt", ".log", ".out", ".md", ".json", ".py", ".ipynb", ".sh")):
                    try:
                        if os.path.getsize(path) > 50e6:
                            continue
                        with open(path, encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if re.search(r"9\.62|15\.87|deep_funnel", line) and "verify_claims" not in path:
                                    if hits < 200:
                                        out(f"  {path}:{i}: {line.strip()[:160]}")
                                    hits += 1
                    except OSError:
                        pass
                if fn.endswith((".tgz", ".tar.gz", ".tar")) and os.path.getsize(path) < 20e9:
                    try:
                        with tarfile.open(path) as t:
                            for m in t:
                                if "deep_funnel" in m.name:
                                    out(f"  in archive {path}: {m.name}")
                                    hits += 1
                    except Exception as e:                    # noqa: BLE001
                        out(f"  (could not list {path}: {e})")
    out(f"  matches: {hits}")


def section_e(cfg, shards):
    out("\n=== E. Single numbers ===")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    out("Claim: on BreastMNIST (156 test images) the standard error of one AUC is about 0.03-0.04")
    for n in ("5", "100"):
        ses, aucs = [], []
        for r in shards.load_all("01_frozen_tuned"):
            k = r["keys"]
            if k["dataset"] != "breastmnist" or str(k["regime"]) != n or k["arm"] != "quantum_vqc":
                continue
            p, y = shards.load_predictions("01_frozen_tuned", **k)
            s = p[:, 1]
            b = []
            for _ in range(500):
                i = rng.integers(0, len(y), len(y))
                if len(set(y[i])) == 2:
                    b.append(roc_auc_score(y[i], s[i]))
            ses.append(float(np.std(b, ddof=1)))
            aucs.append(r["metrics"]["auc"])
        out(f"  VQC n={n}: bootstrap SE over test images, median {statistics.median(ses):.3f} "
            f"(range {min(ses):.3f}-{max(ses):.3f}) over {len(ses)} runs; mean AUC {statistics.mean(aucs):.3f}")
    out("Claim (implicit): the noise sweep's heads are the primary runs' heads")
    pri = {(k["dataset"], int(k["regime"]), int(k["seed"]), k["arm"]): r["metrics"]["auc"]
           for r in shards.load_all("01_frozen_tuned") for k in [r["keys"]]}
    diffs = collections.defaultdict(list)
    for r in shards.load_all("03_robustness_v2"):
        k = r["keys"]
        key = (k["dataset"], int(k["regime"]), int(k["seed"]), k["arm"])
        if key in pri and "0.00" in r.get("curve", {}):
            diffs[k["arm"]].append(abs(r["curve"]["0.00"]["auc"] - pri[key]))
    for arm, v in sorted(diffs.items()):
        out(f"  {arm:24s} {len(v)} runs; |clean AUC (03) - AUC (01)| max {max(v):.2e}, "
            f"identical in {sum(1 for x in v if x == 0)}")
    out("Claim: 1,024 shots retained 0.989-1.005 of the exact AUC (cell means)")
    ratios = []
    for r in shards.load_all("07_hardware_noise"):
        sc = r.get("shot_curve", {})
        if "exact" in sc and "1024" in sc:
            ratios.append(sc["1024"]["auc"] / sc["exact"]["auc"])
    if ratios:
        out(f"  per run: {len(ratios)} runs, ratio {min(ratios):.3f}-{max(ratios):.3f}")


def section_f(workers, B=None):
    import paired_conditions as pc
    out("\n=== F. Analysis code on real data, recomputed from scratch ===")
    ctx = pc.context()
    path = os.path.join(ctx.config.ARTIFACT_ROOT, "family_table.json")
    with open(path, encoding="utf-8") as f:
        fam = {r["test"]: r for r in json.load(f)}
    published = {  # docs/RESULTS.md section 3, 14_exploratory_table.py output
        "x_reupload_vs_fourier_r2": [0.0150, 0.0148, 0.0029, -0.0002, -0.0038],
        "x_reupload_vs_control": [0.0181, 0.0092, 0.0074, 0.0022, 0.0029],
    }
    groups = [("primary", "quantum_vqc", "matched_param_fullrank"),
              ("hs1", "quantum_reupload", "quantum_vqc"),
              ("hs2", "quantum_vqc", "fourier_rff"),
              ("x_reupload_vs_fourier_r2", "quantum_reupload", "fourier_rff_r2"),
              ("x_reupload_vs_control", "quantum_reupload", "matched_param_fullrank")]
    ok = True
    for name, a, b in groups:
        cells = pc.load_two(ctx, ("01_frozen_tuned", a), ("01_frozen_tuned", b))
        rows, slope = pc.summarise_one(cells, name, B or pc.B_DEFAULT, workers)
        if name in ("primary", "hs1", "hs2"):
            test = {"primary": "H-P1", "hs1": "H-S1 n=", "hs2": "H-S2 n="}[name]
            for r in rows:
                t = "H-P1" if name == "primary" else f"{test}{r['n']}"
                if name == "primary" and r["n"] != 5:
                    continue
                pub = fam[t]
                same = (r["R"]["estimate"] == pub["estimate"] and r["R"]["ci_lo"] == pub["ci_lo"]
                        and r["R"]["ci_hi"] == pub["ci_hi"] and r["R"]["p"] == pub["p"])
                ok &= same
                out(f"  {t:10s} recomputed {pc.fmt(r['R'])} p={r['R']['p']:.4f} | published "
                    f"{pub['estimate']:+.4f} [{pub['ci_lo']:+.4f},{pub['ci_hi']:+.4f}] p={pub['p']:.4f} "
                    f"-> {'IDENTICAL' if same else 'DIFFERENT'}")
            if name == "primary":
                pub = fam["H-P2"]
                same = slope["R"]["estimate"] == pub["estimate"] and slope["R"]["ci_lo"] == pub["ci_lo"]
                ok &= same
                out(f"  H-P2       recomputed {pc.fmt(slope['R'], 5)} | published {pub['estimate']:+.5f} "
                    f"[{pub['ci_lo']:+.5f},{pub['ci_hi']:+.5f}] -> {'IDENTICAL' if same else 'DIFFERENT'}")
        else:
            est = [round(r["R"]["estimate"], 4) for r in rows]
            same = all(abs(x - y) < 5e-5 for x, y in zip(est, published[name]))
            ok &= same
            out(f"  {name:26s} recomputed {est} | published {published[name]} -> "
                f"{'MATCHES to 4 decimals' if same else 'DIFFERENT'}")
    out("F PASSED: the analysis code reproduces the published results on the real data."
        if ok else "F FAILED: do not use Amendment 20-21 analyses until this is resolved.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sections", nargs="+", default=list("ABCDE"))
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--B", type=int, default=None, help="testing only; F uses 2000")
    args = p.parse_args()
    import config
    import shards
    for s in args.sections:
        {"A": lambda: section_a(config), "B": lambda: section_b(config, shards),
         "C": lambda: section_c(config, shards), "D": lambda: section_d(config),
         "E": lambda: section_e(config, shards), "F": lambda: section_f(args.workers, args.B)}[s]()
    dst = os.path.join(os.getcwd(), f"logs_verify_claims_{''.join(args.sections)}.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst}")


if __name__ == "__main__":
    main()
