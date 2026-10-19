"""
Amendment 22: sensitivity of the TRAINED heads, from the weights that
Amendment 21's re-execution kept.

    python scripts/analyse_trained_heads.py run --namespace R_03_robustness_v2 --code-at <commit>
    python scripts/analyse_trained_heads.py run --namespace R_15_dim8  --code-at <commit>
    python scripts/analyse_trained_heads.py run --namespace R_16_dim16 --code-at <commit>
    python scripts/analyse_trained_heads.py summary          # no torch needed

`run` rebuilds each trained model with the code of the commit that trained it
(--code-at: the commit recorded in the R_ shards; scripts/codeat.py), loads its
weights, checks that the rebuilt model reproduces the run's stored test
probabilities, then measures the head's sensitivity. One JSON file per
namespace, artifacts/trained_heads/<namespace>.json, resumable. `summary`
writes logs_stats_trained_heads.txt.

WHAT IS MEASURED (fixed in Amendment 22, 15 October, before any weights existed)
-------------------------------------------------------------------------------
The estimator of src/08_lipschitz.py, applied to the trained head alone
(projection and classifier excluded): 512 inputs, 32 random unit directions per
input, eps = 1e-3, the largest finite-difference ratio, divided by the largest
absolute head output over the same inputs. Two input sets:
  (a) uniform on [-angle_scale, angle_scale]^d, as at initialisation. This calls
      08's own lipschitz_estimate, loaded from the main repository's src/;
  (b) the run's own test latents (tanh-squashed projection of its cached test
      features, by its trained projection), the first 512 in test-set order.
      The same estimator, written out below with the inputs given instead of
      drawn (estimate_at): same directions, same eps, same maxima.
The random directions (and, for set (a), the inputs) come from a generator
seeded with the run's seed, so the heads compared within a pair see the same
draws.

INTEGRITY CHECK (fixed 17 October, before this script was first run)
--------------------------------------------------------------------
A rebuilt model is used only if its test probabilities, computed from the cached
test features, agree with the probabilities stored by its re-execution:
    01-path namespaces (R_15, R_16): stored from the same cached features; every
        probability within 1e-5 (float32 storage);
    R_03_robustness_v2: stored at sigma = 0 from the images through the full
        model, so the features differ from the cached ones by floating-point
        order only; every probability within 1e-4.
A run that fails is listed and excluded, never repaired.

SUMMARY (pooling fixed 17 October, before any result)
-----------------------------------------------------
Normalised sensitivity S = L_max / |out|max, per run. Reported: mean S per head,
input set and n; and the paired ratio S(VQC) / S(partner) per run, pooled as the
geometric mean over runs (the natural mean of ratios), with the share of runs
above 1. Partner: matched_param_fullrank at d = 4 (R_03); low_rank at d = 8 and
16 (R_15, R_16), the parity control defined at every d.
Prediction (Amendment 22): the pooled ratio exceeds 1 on both input sets.
Exploratory and descriptive.
"""
import argparse
import collections
import inspect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeat                                          # noqa: E402

OUT_DIR = os.path.join(os.environ.get("QML_ARTIFACT_ROOT", os.path.join(codeat.REPO, "artifacts")),
                       "trained_heads")
PARTNER = {"R_03_robustness_v2": "matched_param_fullrank",
           "R_15_dim8": "low_rank", "R_16_dim16": "low_rank"}
ARMS = {"R_03_robustness_v2": None,                       # all seven heads
        "R_15_dim8": {"quantum_vqc", "low_rank"},
        "R_16_dim16": {"quantum_vqc", "low_rank"}}
TOL = {"R_03_robustness_v2": 1e-4, "R_15_dim8": 1e-5, "R_16_dim16": 1e-5}
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ run
def run(namespace, sha):
    codeat.setup(sha)
    import numpy as np
    import torch
    import config
    import shards
    from data.medmnist_loader import num_classes_of
    from models.registry import build_arm
    lip = codeat.load_by_path("lipschitz08", os.path.join(codeat.REPO, "src", "08_lipschitz.py"))
    m01 = codeat.load_by_path("01_frozen_backbone_ablation",
                              os.path.join(os.path.dirname(config.__file__), "01_frozen_backbone_ablation.py"))

    @torch.no_grad()
    def estimate_at(head, z, n_directions=32, eps=1e-3, generator=None):
        """08's estimator with the inputs given (input set b)."""
        f0 = head(z)
        worst = torch.zeros(z.shape[0], device=z.device)
        for _ in range(n_directions):
            u = torch.randn(z.shape[0], z.shape[1], generator=generator, device=z.device)
            u = u / u.norm(dim=1, keepdim=True)
            worst = torch.maximum(worst, (head(z + eps * u) - f0).norm(dim=1) / eps)
        return {"lipschitz_max": float(worst.max()), "output_absmax": float(f0.abs().max()),
                "n_points": int(z.shape[0])}

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, namespace + ".json")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = json.load(f)
    rows = [r for r in shards.load_all(namespace) if str(r.get("git_sha", "")).startswith(sha[:8])]
    want = ARMS[namespace]
    rows = [r for r in rows if want is None or r["keys"]["arm"] in want]
    accepted = inspect.signature(build_arm).parameters
    dev = config.DEVICE
    print(f"Amendment 22 | {namespace} @ {sha[:8]} | {len(rows)} runs | {len(done)} already done | {dev}",
          flush=True)
    for i, r in enumerate(rows, 1):
        k = r["keys"]
        name = shards.shard_name(namespace, **k)
        if name in done:
            continue
        rec = r.get("recorded", {})
        wfile = os.path.join(config.ARTIFACT_ROOT, rec.get("weights_file", ""))
        entry = {"keys": k, "arm": k["arm"], "dataset": k["dataset"], "regime": str(k["regime"]),
                 "seed": int(k["seed"]), "dim": int(k["dim"])}
        if not rec.get("weights_file") or not os.path.exists(wfile):
            entry["status"] = "no weights file"
            done[name] = entry
            continue
        C = num_classes_of(k["dataset"])
        seed = int(k["seed"])
        config.set_determinism(seed)
        if namespace == "R_03_robustness_v2":
            args = dict(n_layers=config.VQC_LAYERS, seed=seed, build_backbone=False)
        else:
            c = r.get("config", {})
            args = dict(n_layers=c.get("n_layers"), seed=seed, build_backbone=False,
                        use_tanh=c.get("use_tanh"), angle_scale=c.get("angle_scale"),
                        bottleneck_policy=c.get("bottleneck_policy"), head_rank=c.get("head_rank"))
        args = {a: v for a, v in args.items() if a in accepted and v is not None}
        model = build_arm(k["arm"], d=int(k["dim"]), num_classes=C, **args).to(dev).eval()
        sd = torch.load(wfile, map_location=dev, weights_only=True)
        msd = model.state_dict()
        unexpected = [n for n in sd if n not in msd]
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        missing = [n for n in trainable if n not in sd]
        if unexpected or missing:
            entry.update(status="weights do not match the model", unexpected=unexpected, missing=missing)
            done[name] = entry
            continue
        model.load_state_dict(sd, strict=False)

        loaders, _ = m01.get_cached_features(k["dataset"], str(k["regime"]), seed)
        te = loaders["test"]
        with torch.no_grad():
            P = torch.softmax(model.forward_from_features(te.feats), dim=1).double().cpu().numpy()
        stored, labels = shards.load_predictions(namespace, condition=("0.00" if namespace.startswith("R_03")
                                                                       else None), **k)
        if stored is None or stored.shape != P.shape:
            entry.update(status="no comparable stored predictions")
            done[name] = entry
            continue
        diff = float(np.abs(stored - P).max())
        entry["integrity_max_abs_prob_diff"] = diff
        if diff > TOL[namespace]:
            entry["status"] = f"integrity check failed (max |diff| {diff:.2e} > {TOL[namespace]:.0e})"
            done[name] = entry
            continue

        head, d = model.head, int(k["dim"])
        g = torch.Generator(device=dev).manual_seed(seed)
        a = lip.lipschitz_estimate(head, d, model.angle_scale, device=dev, generator=g)
        with torch.no_grad():
            z = model.latent(te.feats[:512])
        g = torch.Generator(device=dev).manual_seed(seed)
        b = estimate_at(head, z, generator=g)
        entry.update(status="ok",
                     a={"L_max": a["lipschitz_max"], "absmax": a["output_absmax"],
                        "S": a["lipschitz_max"] / a["output_absmax"] if a["output_absmax"] > 0 else None},
                     b={"L_max": b["lipschitz_max"], "absmax": b["output_absmax"], "n": b["n_points"],
                        "S": b["lipschitz_max"] / b["output_absmax"] if b["output_absmax"] > 0 else None})
        done[name] = entry
        if i % 25 == 0 or i == len(rows):
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(done, f)
            os.replace(path + ".tmp", path)
            print(f"  [{i}/{len(rows)}] last: {k['arm']} {k['dataset']} n={k['regime']} s={seed} "
                  f"S_a={entry['a']['S']:.3f} S_b={entry['b']['S']:.3f} integrity {diff:.1e}", flush=True)
        del model
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(done, f)
    os.replace(path + ".tmp", path)
    bad = collections.Counter(e["status"] for e in done.values() if e.get("status") != "ok")
    print(f"done: {sum(1 for e in done.values() if e.get('status') == 'ok')} ok; not used: {dict(bad)}")


# ------------------------------------------------------------------ summary
def geo(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def summary():
    out("Amendment 22 - sensitivity of the trained heads (exploratory, descriptive)")
    out("S = L_max / |out|max of the head alone; (a) uniform inputs, (b) the run's test latents")
    verdicts = []
    for ns, partner in PARTNER.items():
        path = os.path.join(OUT_DIR, ns + ".json")
        if not os.path.exists(path):
            out(f"\n{ns}: not run")
            verdicts.append((ns, "not run"))
            continue
        with open(path, encoding="utf-8") as f:
            done = json.load(f)
        ok = [e for e in done.values() if e.get("status") == "ok"]
        bad = collections.Counter(e["status"] for e in done.values() if e.get("status") != "ok")
        diffs = [e["integrity_max_abs_prob_diff"] for e in ok]
        out(f"\n=== {ns}: {len(ok)} runs used; not used {dict(bad) or 0}; "
            f"integrity max |prob diff| {max(diffs) if diffs else float('nan'):.1e} (tolerance {TOL[ns]:.0e})")
        regs = sorted({e["regime"] for e in ok}, key=lambda x: int(x) if x.isdigit() else 10 ** 9)
        arms = sorted({e["arm"] for e in ok})
        for s in ("a", "b"):
            out(f"\n  input set ({s}): mean S per head and n")
            out(f"  {'head':26s} " + " ".join(f"{'n=' + r:>9s}" for r in regs) + f" {'all':>9s}")
            for arm in arms:
                vals = []
                for r in regs + ["*"]:
                    v = [e[s]["S"] for e in ok if e["arm"] == arm and (r == "*" or e["regime"] == r)
                         and e[s]["S"] is not None]
                    vals.append(f"{sum(v) / len(v):9.3f}" if v else "        -")
                out(f"  {arm:26s} " + " ".join(vals))
        idx = {(e["dataset"], e["regime"], e["seed"], e["arm"]): e for e in ok}
        out(f"\n  paired ratio S(quantum_vqc) / S({partner}), geometric mean over runs "
            f"[share of runs above 1]")
        res = {}
        for s in ("a", "b"):
            line, allr = [], []
            for r in regs + ["*"]:
                rat = []
                for (ds, rg, sd, arm), e in idx.items():
                    if arm != "quantum_vqc" or (r != "*" and rg != r):
                        continue
                    q = idx.get((ds, rg, sd, partner))
                    if q and e[s]["S"] and q[s]["S"]:
                        rat.append(e[s]["S"] / q[s]["S"])
                if r == "*":
                    allr = rat
                line.append(f"{geo(rat):6.3f} [{sum(x > 1 for x in rat)}/{len(rat)}]" if rat else "-")
            out(f"  ({s}) " + "  ".join(f"{'n=' + r if r != '*' else 'all'}: {v}"
                                        for r, v in zip(regs + ["*"], line)))
            res[s] = geo(allr)
        holds = all(res[s] > 1 for s in ("a", "b"))
        verdicts.append((ns, f"pooled ratio (a) {res['a']:.3f}, (b) {res['b']:.3f} -> "
                             f"prediction {'HOLDS' if holds else 'FAILS'}"))
    out("\nPrediction (Amendment 22): the VQC's normalised sensitivity exceeds its partner's on both "
        "input sets (pooled ratio > 1)")
    for ns, v in verdicts:
        out(f"  {ns:22s} {v}")
    dst = os.path.join(os.getcwd(), "logs_stats_trained_heads.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten to {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "summary"])
    p.add_argument("--namespace", choices=sorted(PARTNER))
    p.add_argument("--code-at", default=None)
    a = p.parse_args()
    if a.mode == "run":
        if not a.namespace or not a.code_at:
            p.error("run needs --namespace and --code-at (the commit recorded in that namespace)")
        run(a.namespace, a.code_at)
    else:
        summary()


if __name__ == "__main__":
    main()
