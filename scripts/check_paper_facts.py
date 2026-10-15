"""
Read every result shard and print the facts the manuscript states about HOW the
runs were trained. Plain Python 3 standard library only: no GPU, no project
imports. It reads shard files and changes nothing; its only output is the
report below, also written to logs_paper_facts.txt in the current directory.

    python scripts/check_paper_facts.py artifacts/shards     (from the repo root)

The argument is the folder that contains the namespace folders
(01_frozen_tuned, 14_readout, ...). If you are not sure where that is, pass the
backup root and the script will look for a folder called "shards" below it.

WHY
---
Several facts in the manuscript cannot be confirmed from the code or the logs,
only from the run records:

1. Learning rates. Some launches passed --use-tuned-lr (the log prints
   "using tuned LRs"), others passed explicit --lr-head/--lr-quantum, which the
   log does not print. Every shard records the rate it ACTUALLY trained with
   (metrics.lr_head, metrics.lr_quantum), so this settles it per namespace and arm.
2. Early stopping. The code stops on validation AUC with patience 30 and a
   minimum-epoch floor. For every run, the script recomputes from the recorded
   best epoch and training-set size the epoch at which that rule says training
   must have stopped, and checks it against the epochs the run actually ran. If
   every run matches, the rule in the paper is the rule that was used.
3. Gradient clipping at norm 20. The records hold the pre-clipping gradient
   norm per epoch, as the MEAN over that epoch's batches; the per-batch maximum
   is not stored. So "no epoch mean above 20" is necessary but not sufficient
   for "clipping never bound" - except where an epoch is a single batch
   (n <= 10 binary, n = 5 multi-class), where the epoch value IS the batch
   value.
4. Run time, and the raw trained-head Lipschitz constants that the noise sweep
   records (not normalised - see src/08_lipschitz.py).
"""
import collections
import json
import os
import statistics
import sys

GROUP_IGNORE = {"dataset", "regime", "seed", "lrh", "lrq", "lr"}
OUT_LINES = []


def out(s=""):
    print(s)
    OUT_LINES.append(s)


def find_shard_root(path):
    if any(os.path.isdir(os.path.join(path, d)) and d[:2].isdigit()
           for d in os.listdir(path)):
        return path
    for dp, dns, _ in os.walk(path):
        if os.path.basename(dp) == "shards" and any(d[:2].isdigit() for d in dns):
            return dp
    return None


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def fmt_counts(c):
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/check_paper_facts.py <shards folder>")
    root = find_shard_root(sys.argv[1])
    if root is None:
        sys.exit(f"No shard namespaces found under {sys.argv[1]}")
    out(f"shard root: {root}")

    for ns in sorted(os.listdir(root)):
        nsdir = os.path.join(root, ns)
        if not os.path.isdir(nsdir) or ns.startswith("_"):
            continue
        files = [f for f in os.listdir(nsdir) if f.endswith(".json")]
        if not files:
            continue
        groups = collections.defaultdict(list)
        shas = collections.Counter()
        bad = 0
        for fn in files:
            try:
                with open(os.path.join(nsdir, fn), encoding="utf-8") as f:
                    r = json.load(f)
            except Exception:
                bad += 1
                continue
            k = r.get("keys", {})
            ignore = GROUP_IGNORE - {"lr"} if ns.startswith("09") else GROUP_IGNORE
            gk = tuple(sorted((a, str(b)) for a, b in k.items() if a not in ignore))
            groups[gk].append(r)
            shas[str(r.get("git_sha", "?"))[:8]] += 1

        out("")
        out("=" * 100)
        out(f"NAMESPACE {ns}   shards={len(files)}   unreadable={bad}")
        out(f"  git commits: {fmt_counts(shas)}")
        # Training budget: 30_budget_check runs 01 with MAX_EPOCHS=1000, PATIENCE=200
        # (scripts/run_budget_check.py); every other namespace uses config defaults.
        max_ep, pat = (1000, 200) if ns.startswith("30_budget") else (100, 30)
        for gk, rs in sorted(groups.items()):
            label = " ".join(f"{a}={b}" for a, b in gk)
            lrh, lrq = collections.Counter(), collections.Counter()
            epochs, best, times, early = [], [], [], 0
            gmax, over20, n_epochs_total = 0.0, 0, 0
            lip_tr, n_train, n_eval = [], collections.Counter(), collections.Counter()
            val_auc, mean_gn = [], []
            rule_ok, rule_bad, rule_na, bad_examples = 0, 0, 0, []
            for r in rs:
                k = r.get("keys", {})
                m = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
                if not m and isinstance(r.get("test_metrics"), dict):   # 09_lr_selection
                    m = r["test_metrics"]
                h = m.get("lr_head", g(r, "config", "lr_head"))
                q = m.get("lr_quantum", g(r, "config", "lr_quantum"))
                if h is None and isinstance(r.get("lr"), dict):
                    h, q = r["lr"].get("head"), r["lr"].get("quantum")
                if h is None and "lrh" in k:
                    h = f"key:{k['lrh']}"
                if q is None and "lrq" in k:
                    q = f"key:{k['lrq']}"
                lrh[str(h)] += 1
                lrq[str(q)] += 1
                if m.get("epochs_run") is not None:
                    epochs.append(m["epochs_run"])
                    if m["epochs_run"] < max_ep:
                        early += 1
                    if m.get("mean_epoch_time") is not None:
                        times.append(m["epochs_run"] * m["mean_epoch_time"])
                if m.get("best_epoch") is not None:
                    best.append(m["best_epoch"] + 1)
                # Early-stopping rule in src/train/loop.py + config.min_epochs_for:
                # stop at 0-indexed epoch e = max(best + 30, min_epochs) if e <= 99,
                # where min_epochs = min(max(20, 200 // B), 50), B = ceil(n_train / 32).
                nt0 = g(r, "meta", "n_train")
                if (m.get("epochs_run") is not None and m.get("best_epoch") is not None
                        and nt0 and m["best_epoch"] >= 0):
                    B = (int(nt0) + 31) // 32
                    min_e = min(max(20, 200 // max(B, 1)), max_ep // 2)
                    e = max(m["best_epoch"] + pat, min_e)
                    expected = e + 1 if e <= max_ep - 1 else max_ep
                    if m["epochs_run"] == expected:
                        rule_ok += 1
                    else:
                        rule_bad += 1
                        if len(bad_examples) < 3:
                            bad_examples.append((k.get("dataset"), k.get("regime"), k.get("seed"),
                                                 m["epochs_run"], expected))
                elif m.get("epochs_run") is not None:
                    rule_na += 1
                hist = g(r, "history", "pre_clip_grad_norm") or []
                for v in hist:
                    if v is None:
                        continue
                    n_epochs_total += 1
                    gmax = max(gmax, v)
                    over20 += v > 20
                if r.get("val_auc") is not None:          # 09_lr_selection payload
                    val_auc.append(r["val_auc"])
                if r.get("mean_grad_norm") is not None:
                    mean_gn.append(r["mean_grad_norm"])
                lt = g(r, "lipschitz_trained", "lipschitz_max")
                if lt is not None:
                    lip_tr.append(lt)
                nt = g(r, "meta", "n_train")
                if nt is not None:
                    n_train[f"{k.get('dataset')}:{nt}"] += 1
                if r.get("n_eval") is not None:
                    n_eval[f"{k.get('dataset')}:{r['n_eval']}"] += 1

            out(f"  [{label}]  runs={len(rs)}")
            out(f"      lr_head: {fmt_counts(lrh)}")
            out(f"      lr_quantum: {fmt_counts(lrq)}")
            if epochs:
                out(f"      epochs run: min {min(epochs)}, median {statistics.median(epochs)}, "
                    f"max {max(epochs)}; stopped before the {max_ep}-epoch cap: {early} of {len(epochs)}")
            if rule_ok or rule_bad:
                out(f"      early-stopping rule (val AUC, patience {pat}, floor min(max(20,200//B),{max_ep // 2})): "
                    f"matches {rule_ok}, does NOT match {rule_bad}, not checkable {rule_na}"
                    + (f"; e.g. (dataset, n, seed, ran, expected) {bad_examples}" if bad_examples else ""))
            if best:
                out(f"      best (selected) epoch: min {min(best)}, median {statistics.median(best)}, "
                    f"max {max(best)}")
            if n_epochs_total:
                out(f"      pre-clip grad norm (epoch means): max {gmax:.3f}; "
                    f"epochs above 20: {over20} of {n_epochs_total}")
            if times:
                out(f"      summed training time: {sum(times)/3600:.2f} h")
            if val_auc:
                out(f"      mean validation AUC (selection criterion): {statistics.mean(val_auc):.4f}, n={len(val_auc)}")
            if mean_gn:
                out(f"      mean pre-clip gradient norm: {statistics.mean(mean_gn):.3f} "
                    f"(range of run means {min(mean_gn):.2f}-{max(mean_gn):.2f})")
            if lip_tr:
                out(f"      trained-head Lipschitz (max over points): mean {statistics.mean(lip_tr):.3f}, "
                    f"n={len(lip_tr)}")
            if n_train and ("full" in label or len(n_train) <= 8):
                out(f"      n_train by dataset: {fmt_counts(n_train)}")
            if n_eval:
                out(f"      n_eval by dataset: {fmt_counts(n_eval)}")

    dst = os.path.join(os.getcwd(), "logs_paper_facts.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT_LINES) + "\n")
    out("")
    out(f"written to {dst}")


if __name__ == "__main__":
    main()
