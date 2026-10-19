"""
Amendment 23: hardware feasibility of the circuits as trained in the study
(tuned rate 1e-2), and depolarising noise after every layer.

    python scripts/analyse_hardware_tuned.py run --code-at <commit of the 33_hardware_tuned records>
    python scripts/analyse_hardware_tuned.py summary          # no torch needed

`run` rebuilds every model of 33_hardware_tuned (written by
`replay.py --hardware-tuned`) from its kept weights with the code that trained
it, and computes part (b). `summary` reports parts (a) and (b) and the
predictions; it writes logs_stats_hardware_tuned.txt.

PART (a), from the 33_hardware_tuned records themselves (07's own evaluation)
----------------------------------------------------------------------------
Shot levels {exact, 8192, 1024, 256, 64} and end-of-circuit depolarising
p in {0, 0.001, 0.005, 0.01, 0.05}, up to 500 test images per run, per cell
(dataset, n): mean AUC. Beside it, the same cells of 07_hardware_noise (rate
1e-3), descriptively. Consistency check named in the amendment: each run's
weights against the R_01_frozen_tuned weights of the same dataset, n and seed
(quantum_vqc at 1e-2), which come from the same protocol; the largest absolute
difference is reported per run.

PART (b), computed here
-----------------------
Depolarising noise after EVERY StronglyEntanglingLayer, applied with the
independent NumPy density-matrix simulator of tests/test_independent_theory.py
(`run(z, W, d, L, p=p, where="layer")`), at p in {0, 0.001, 0.005, 0.01, 0.05},
on the same test images 07 used (numpy default_rng(seed).choice of up to 500).
The logits are the run's own trained classifier applied to the simulator's
<X_i>; AUC by the project's compute_metrics.
Required first, per run (Amendment 23): the simulator reproduces the PennyLane
head outputs of those weights within 1e-6 on every one of those images. A run
that fails is not reported in (b).
Sanity check also printed: the simulator's end-of-circuit AUC against 07's
recorded depolarising AUC at every p.

PREDICTIONS (Amendment 23)
--------------------------
 (i)   at 1,024 shots the cell-mean AUC is between 0.98 and 1.02 of exact;
 (ii)  end-of-circuit noise leaves binary AUC exactly unchanged;
 (iii) per-layer noise at p <= 0.01 changes the cell-mean binary AUC by less
       than 0.005 in every cell.
Exploratory feasibility; no classical counterpart.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeat                                          # noqa: E402

ART = os.environ.get("QML_ARTIFACT_ROOT", os.path.join(codeat.REPO, "artifacts"))
OUT = os.path.join(ART, "hardware_tuned", "per_layer.json")
NS, ORIG = "33_hardware_tuned", "07_hardware_noise"
P_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05]
BINARY = {"breastmnist", "pneumoniamnist"}
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def shards_dir(ns):
    return os.path.join(ART, "shards", ns)


def load_ns(ns):
    d = shards_dir(ns)
    rows = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    rows.append(json.load(f))
    return rows


# ------------------------------------------------------------------ run
def run(sha):
    codeat.setup(sha)
    import numpy as np
    import torch
    import config
    import shards
    from data.medmnist_loader import num_classes_of
    from models.registry import build_arm
    from train.metrics import compute_metrics
    m01 = codeat.load_by_path("01_frozen_backbone_ablation",
                              os.path.join(os.path.dirname(config.__file__), "01_frozen_backbone_ablation.py"))
    sim = codeat.load_by_path("independent_theory",
                              os.path.join(codeat.REPO, "tests", "test_independent_theory.py"))
    dev = config.DEVICE
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            res = json.load(f)
    # The runs may carry more than one commit (re-launched after a pull). They
    # are rebuilt with the code of --code-at, which is valid only for commits
    # whose src/ is identical to it; any other commit is skipped and reported.
    import subprocess
    rows, same_src = [], {}
    for r in shards.load_all(NS):
        c = str(r.get("git_sha", ""))
        if c not in same_src:
            same_src[c] = subprocess.run(["git", "-C", codeat.REPO, "diff", "--quiet", c, sha, "--", "src"]
                                         ).returncode == 0
        if same_src[c]:
            rows.append(r)
    for c, ok_ in same_src.items():
        print(f"  records at {c[:8]}: src/ {'identical to' if ok_ else 'DIFFERS from'} {sha[:8]}"
              f"{'' if ok_ else ' - skipped'}")
    # Partner: the primary namespace's quantum_vqc run at the tuned rates, frozen
    # backbone, learned projection, default depth, tanh and angle (no other key).
    r01 = {}
    for q in shards.load_all("R_01_frozen_tuned"):
        kq = q["keys"]
        if (kq.get("arm") == "quantum_vqc" and kq.get("fp") == "all" and int(kq.get("aug", 0)) == 0
                and kq.get("lrh") == "1e-02" and kq.get("lrq") == "1e-02"
                and set(kq) <= {"dataset", "regime", "dim", "seed", "arm", "fp", "aug", "lrh", "lrq", "L"}
                and int(kq.get("L", config.VQC_LAYERS)) == config.VQC_LAYERS):
            r01[(kq["dataset"], str(kq["regime"]), int(kq["dim"]), int(kq["seed"]))] = q
    print(f"Amendment 23(b) | {len(rows)} runs @ {sha[:8]} | {dev}", flush=True)
    for r in rows:
        k = r["keys"]
        name = shards.shard_name(NS, **k)
        if name in res:
            continue
        e = {"keys": k}
        rec = r.get("recorded", {})
        wfile = os.path.join(config.ARTIFACT_ROOT, rec.get("weights_file", ""))
        if not rec.get("weights_file") or not os.path.exists(wfile):
            e["status"] = "no weights file"
            res[name] = e
            continue
        seed, C, d = int(k["seed"]), num_classes_of(k["dataset"]), int(k["dim"])
        config.set_determinism(seed)
        model = build_arm(k["arm"], d=d, num_classes=C, n_layers=config.VQC_LAYERS, seed=seed,
                          build_backbone=False).to(dev).eval()
        sd = torch.load(wfile, map_location=dev, weights_only=True)
        model.load_state_dict(sd, strict=False)

        # 23(a) consistency with the primary namespace's weights (same protocol)
        q = r01.get((k["dataset"], str(k["regime"]), d, seed))
        if q and q.get("recorded", {}).get("weights_file"):
            w01 = torch.load(os.path.join(config.ARTIFACT_ROOT, q["recorded"]["weights_file"]),
                             map_location="cpu", weights_only=True)
            common = [n for n in sd if n in w01]
            e["vs_R01_max_abs_weight_diff"] = max(float((sd[n].cpu() - w01[n]).abs().max()) for n in common) \
                if common else None
        else:
            e["vs_R01_max_abs_weight_diff"] = "no R_01_frozen_tuned partner with weights"

        loaders, _ = m01.get_cached_features(k["dataset"], str(k["regime"]), seed)
        te = loaders["test"]
        feats, labels = te.feats, te.labels.cpu().numpy()
        n_eval = int(r.get("n_eval", len(labels)))
        idx = np.random.default_rng(seed).choice(len(labels), min(n_eval, len(labels)), replace=False)
        feats, labels = feats[idx], labels[idx]
        with torch.no_grad():
            z = model.latent(feats)
            v_pl = model.head(z).double().cpu().numpy()
            P = torch.softmax(model.classifier(model.head(z)), dim=1).cpu().numpy()
        e["exact_auc_rebuilt"] = float(compute_metrics(labels, P.argmax(1), P, C)["auc"])
        e["exact_auc_recorded"] = r["shot_curve"]["exact"]["auc"]

        W = model.head.q_layer.weights.detach().double().cpu().numpy()
        L = W.shape[0]
        zz = z.double().cpu().numpy()
        v_np = np.array([sim.run(zi, W, d, L) for zi in zz])
        e["sim_vs_pennylane_max_abs"] = float(np.abs(v_np - v_pl).max())
        if e["sim_vs_pennylane_max_abs"] > 1e-6:
            e["status"] = "simulator does not reproduce the head within 1e-6"
            res[name] = e
            continue

        def auc_of(V):
            with torch.no_grad():
                lg = model.classifier(torch.as_tensor(V, dtype=torch.float32, device=dev))
                Pr = torch.softmax(lg, dim=1).cpu().numpy()
            return float(compute_metrics(labels, Pr.argmax(1), Pr, C)["auc"])

        e["per_layer_auc"], e["end_auc_sim"] = {}, {}
        for p in P_LEVELS:
            Vl = v_np if p == 0 else np.array([sim.run(zi, W, d, L, p=p, where="layer") for zi in zz])
            Ve = v_np if p == 0 else np.array([sim.run(zi, W, d, L, p=p, where="end") for zi in zz])
            e["per_layer_auc"][f"{p:.3f}"] = auc_of(Vl)
            e["end_auc_sim"][f"{p:.3f}"] = auc_of(Ve)
        e["status"] = "ok"
        res[name] = e
        with open(OUT + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(OUT + ".tmp", OUT)
        print(f"  {k['dataset']} n={k['regime']} s={seed}: sim-PL {e['sim_vs_pennylane_max_abs']:.1e}  "
              f"exact {e['exact_auc_rebuilt']:.4f} (rec {e['exact_auc_recorded']:.4f})  "
              f"per-layer p=.01 {e['per_layer_auc']['0.010']:.4f}", flush=True)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(OUT + ".tmp", OUT)


# ------------------------------------------------------------------ summary
def summary():
    mean = lambda v: sum(v) / len(v) if v else float("nan")      # noqa: E731
    rows, orig = load_ns(NS), load_ns(ORIG)
    out(f"Amendment 23 - hardware feasibility at the tuned rate (exploratory; no classical counterpart)")
    out(f"{NS}: {len(rows)} runs; {ORIG} (rate 1e-3): {len(orig)} runs")

    def cells(rs):
        c = collections.defaultdict(list)
        for r in rs:
            c[(r["keys"]["dataset"], str(r["keys"]["regime"]))].append(r)
        return c
    c33, c07 = cells(rows), cells(orig)
    shots = ["exact", "8192", "1024", "256", "64"]
    out("\n(a) shot noise, cell-mean AUC (tuned rate)   [07 at 1e-3: exact]")
    out(f"  {'dataset':16s} {'n':>4s} {'runs':>4s} " + " ".join(f"{s:>8s}" for s in shots)
        + f" {'1024/exact':>11s} {'07 exact':>9s}")
    p_i, p_ii = [], []
    for cell in sorted(c33):
        rs = c33[cell]
        m = {s: mean([r["shot_curve"][s]["auc"] for r in rs if s in r["shot_curve"]]) for s in shots}
        ratio = m["1024"] / m["exact"]
        p_i.append((cell, ratio))
        o = mean([r["shot_curve"]["exact"]["auc"] for r in c07.get(cell, [])])
        out(f"  {cell[0]:16s} {cell[1]:>4s} {len(rs):4d} " + " ".join(f"{m[s]:8.4f}" for s in shots)
            + f" {ratio:11.4f} {o:9.4f}")
    out("\n(a) end-of-circuit depolarising, cell-mean AUC (07's own evaluation)")
    ps = [f"{p:.3f}" for p in P_LEVELS]
    out(f"  {'dataset':16s} {'n':>4s} " + " ".join(f"{p:>8s}" for p in ps))
    for cell in sorted(c33):
        rs = c33[cell]
        out(f"  {cell[0]:16s} {cell[1]:>4s} " + " ".join(
            f"{mean([r['depol_curve'][p]['auc'] for r in rs]):8.4f}" for p in ps))
        if cell[0] in BINARY:
            for r in rs:
                base = r["depol_curve"]["0.000"]["auc"]
                p_ii += [r["depol_curve"][p]["auc"] == base for p in ps]

    res = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            res = json.load(f)
    ok = [e for e in res.values() if e.get("status") == "ok"]
    notok = collections.Counter(e.get("status") for e in res.values() if e.get("status") != "ok")
    out(f"\n(b) per-layer depolarising (independent NumPy simulator): {len(ok)} runs used; "
        f"not used {dict(notok) or 0}")
    if ok:
        out(f"  simulator vs PennyLane head outputs: max |diff| "
            f"{max(e['sim_vs_pennylane_max_abs'] for e in ok):.1e} (required < 1e-6)")
        out(f"  rebuilt exact AUC vs recorded exact AUC: max |diff| "
            f"{max(abs(e['exact_auc_rebuilt'] - e['exact_auc_recorded']) for e in ok):.1e}")
        w = [e["vs_R01_max_abs_weight_diff"] for e in res.values()
             if isinstance(e.get("vs_R01_max_abs_weight_diff"), float)]
        out(f"  weights vs R_01_frozen_tuned (same dataset, n, seed): {len(w)} pairs, "
            f"max |diff| {max(w) if w else float('nan'):.2e}, identical in {sum(x == 0.0 for x in w)}")
        byname = {json.dumps(r["keys"], sort_keys=True): r for r in rows}
        dev_end = []
        for e in ok:
            r = byname.get(json.dumps(e["keys"], sort_keys=True))
            if r:
                dev_end += [abs(e["end_auc_sim"][p] - r["depol_curve"][p]["auc"]) for p in ps]
        out(f"  sanity: simulator end-of-circuit AUC vs 07's recorded AUC, max |diff| "
            f"{max(dev_end) if dev_end else float('nan'):.1e}")
        cb = collections.defaultdict(list)
        for e in ok:
            cb[(e["keys"]["dataset"], str(e["keys"]["regime"]))].append(e)
        out(f"\n  {'dataset':16s} {'n':>4s} {'runs':>4s} " + " ".join(f"{p:>8s}" for p in ps)
            + "   change at 0.010")
        p_iii = []
        for cell in sorted(cb):
            es = cb[cell]
            m = {p: mean([e["per_layer_auc"][p] for e in es]) for p in ps}
            ch = m["0.010"] - m["0.000"]
            if cell[0] in BINARY:
                p_iii += [abs(m[p] - m["0.000"]) < 0.005 for p in ("0.001", "0.005", "0.010")]
            out(f"  {cell[0]:16s} {cell[1]:>4s} {len(es):4d} " + " ".join(f"{m[p]:8.4f}" for p in ps)
                + f"   {ch:+.4f}")
    else:
        p_iii = []
    out("\nPredictions (Amendment 23)")
    if p_i:
        out(f"  (i)   1,024 shots within 0.98-1.02 of exact in every cell: "
            f"{'HOLDS' if all(0.98 <= x <= 1.02 for _, x in p_i) else 'FAILS'}"
            f"  (range {min(x for _, x in p_i):.4f}-{max(x for _, x in p_i):.4f})")
    else:
        out("  (i)   not computable")
    out(f"  (ii)  end-of-circuit noise leaves binary AUC exactly unchanged: "
        f"{'HOLDS' if p_ii and all(p_ii) else 'FAILS' if p_ii else 'not computable'}")
    out(f"  (iii) per-layer p <= 0.01 changes cell-mean binary AUC by < 0.005 in every cell: "
        f"{'HOLDS' if p_iii and all(p_iii) else 'FAILS' if p_iii else 'not computable'}")
    dst = os.path.join(os.getcwd(), "logs_stats_hardware_tuned.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "summary"])
    p.add_argument("--code-at", default=None)
    a = p.parse_args()
    if a.mode == "run":
        if not a.code_at:
            p.error("run needs --code-at (the commit recorded in 33_hardware_tuned)")
        run(a.code_at)
    else:
        summary()


if __name__ == "__main__":
    main()
