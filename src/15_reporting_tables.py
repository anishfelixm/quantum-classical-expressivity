"""
THE REPORTING TABLES: everything the analysis plan promised that is not a
confirmatory test, computed from the saved run records and predictions.

    python src/15_reporting_tables.py                 # all tables
    python src/15_reporting_tables.py --nested        # also nested intervals for d=8/16/full (slow; needs scipy)
    python src/15_reporting_tables.py --quick         # smoke test: primary namespace only

No training, no GPU, no torch. Reads artifacts/shards/<namespace>/*.json and
artifacts/predictions/<namespace>/*.npz exactly as the cluster wrote them.

WHY THIS FILE EXISTS
--------------------
An audit of docs/analysis_plan.md on 12 October 2026 found items the plan
promised and the project had not yet produced. None needs a new run; all are
read off records that already exist:

  1. Absolute AUC per arm, dataset and n. Every comparison in the paper is a
     difference; a reader must also see the level. §6 limitation 4 asserts the
     primary arms sit at "AUC 0.60-0.68 at n=5"; this table checks it.
  2. The F1-selected sensitivity analysis promised by Amendment 2. Every run
     stored the test metrics of its F1-selected checkpoint
     (metrics["test_f1_selected"]); they were never tabled.
  3. Cohen's d with every p-value (§5).
  4. ECE and probability spread under sensor noise, and the noise statistic G
     for the Fourier and linear heads (Amendment 13: "reported alongside as
     exploratory ... reported descriptively").
  5. Practical significance: effects below 0.01 AUC labelled negligible (§5).
  6. Tables for d=8, d=16 and full data, which have no figure.
  7. Provenance: the earliest and latest run timestamp and the git commits of
     every namespace, so "the plan was committed before the confirmatory run"
     can be checked against the plan's own commit date.
  8. Trainable capacity per component, read from the runs themselves, for the
     overview figure.

DEFINITIONS, FIXED HERE BEFORE ANY OF THESE NUMBERS WERE SEEN (13 Oct 2026)
---------------------------------------------------------------------------
Cohen's d. Every confirmatory test is paired on seed, so the effect size is the
paired d_z = mean / SD over seeds of the per-seed contrast, where the per-seed
contrast is pooled across cells with equal weight exactly as the test pools
them: per regime for per-n tests, across all cells for H-S3, H-S4 and H-S7, and
the per-seed slope of pooled delta on log2 n for H-P2 and H-S5b. It describes
how consistent the effect is across seeds; it is not a second significance
test, and its denominator ignores test-set sampling variance, which the
confirmatory interval includes. For H-S4 the per-seed analogue is
mean over cells of |delta_frozen,s| - |delta_adaptive,s|, which differs from
the registered statistic (absolute value taken after averaging over seeds);
this is stated in the table.

F1-selected sensitivity. The same contrasts recomputed on the F1-selected
checkpoint's test AUC, pooled across datasets, with a seed-level 95% interval
(t distribution over per-seed pooled contrasts). Descriptive only: it ignores
test-set variance and is not corrected.

Negligible. |estimate| < 0.01 AUC, applied to tests measured in AUC units.

Absolute AUC and noise metrics are means and SDs over seeds; descriptive.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

REGIMES = ("5", "10", "20", "50", "100")
SIGMAS = ("0.05", "0.10", "0.15", "0.20")
DATASETS = ("bloodmnist", "breastmnist", "pathmnist", "pneumoniamnist")
NEGLIGIBLE = 0.01
ARM_LABEL = {
    "quantum_vqc": "VQC", "matched_param_fullrank": "matched classical head",
    "low_rank": "low-rank classical head", "fourier_rff": "Fourier fit {-1,0,1}^d",
    "fourier_rff_r2": "Fourier fit {-2..2}^d", "quantum_reupload": "re-uploading VQC",
    "quantum_rich": "VQC, 2-local readout", "quantum_rich_padded": "VQC, padded readout",
    "quantum_basic": "VQC, basic entangler", "linear": "identity head (tanh + linear)",
}
# Keys that vary BETWEEN the arms of one comparison and so must not define the cell.
ARM_LEVEL_KEYS = {"seed", "arm", "lrh", "lrq", "rank"}

# name -> (namespace, arm A, arm B, extra filter on cell keys, cross-key contrast)
# A selector is (arm, rank or None). Cell filter: required key values ("" = key absent).
CONTRASTS = {
    "primary":         ("01_frozen_tuned", ("quantum_vqc", None), ("matched_param_fullrank", None), {}),
    "hs1":             ("01_frozen_tuned", ("quantum_reupload", None), ("quantum_vqc", None), {}),
    "hs2":             ("01_frozen_tuned", ("quantum_vqc", None), ("fourier_rff", None), {}),
    "e7":              ("01_frozen_tuned", ("quantum_reupload", None), ("fourier_rff_r2", None), {}),
    "reupload_vs_control": ("01_frozen_tuned", ("quantum_reupload", None), ("matched_param_fullrank", None), {}),
    "hs5":             ("10_capacity", ("low_rank", "0"), ("low_rank", "8"), {}),
    "hs5_followup":    ("27_capacity_tuned", ("low_rank", "0"), ("low_rank", "8"), {}),
    "hs6_pca":         ("12_bottleneck", ("quantum_vqc", None), ("matched_param_fullrank", None), {"bn": "pca"}),
    "hs6_random":      ("12_bottleneck", ("quantum_vqc", None), ("matched_param_fullrank", None), {"bn": "random"}),
    "hs6_learned_ref": ("12_bottleneck", ("quantum_vqc", None), ("matched_param_fullrank", None), {"bn": ""}),
    "hs6f_pca":        ("26_bottleneck_tuned", ("quantum_vqc", None), ("matched_param_fullrank", None), {"bn": "pca"}),
    "hs6f_random":     ("26_bottleneck_tuned", ("quantum_vqc", None), ("matched_param_fullrank", None), {"bn": "random"}),
    "hs7a":            ("14_readout", ("quantum_rich", None), ("quantum_rich_padded", None), {}),
    "hs7b":            ("14_readout", ("quantum_rich", None), ("quantum_vqc", None), {}),
    "ansatz":          ("21_ansatz", ("quantum_basic", None), ("quantum_vqc", None), {}),
    "d8":              ("15_dim8", ("quantum_vqc", None), ("low_rank", None), {}),
    "d16":             ("16_dim16", ("quantum_vqc", None), ("low_rank", None), {}),
    "full":            ("17_fulldata", ("quantum_vqc", None), ("matched_param_fullrank", None), {}),
}

# The 23 confirmatory tests: (test id, plain-language description, contrast, how, unit)
FAMILY = (
    [("H-P1", "VQC vs matched head, n=5", "primary", ("regime", "5"), "auc"),
     ("H-P2", "VQC vs matched head, slope over n", "primary", ("slope",), "slope")]
    + [(f"H-S1 n={n}", f"re-uploading vs single-encoding VQC, n={n}", "hs1", ("regime", n), "auc") for n in REGIMES]
    + [(f"H-S2 n={n}", f"VQC vs Fourier fit of its own class, n={n}", "hs2", ("regime", n), "auc") for n in REGIMES]
    + [(f"H-S3 s={s}", f"VQC F1 loss minus AUC loss, sigma={s}", "hs3", ("sigma", s), "gap") for s in SIGMAS]
    + [("H-S4", "encoder adaptation shrinks the head difference", "hs4", ("all",), "auc"),
       ("H-S5a", "smaller vs larger classical head, n=5", "hs5", ("regime", "5"), "auc"),
       ("H-S5b", "smaller vs larger classical head, slope", "hs5", ("slope",), "slope"),
       ("H-S6 pca", "VQC vs matched head, frozen PCA, n=5", "hs6_pca", ("regime", "5"), "auc"),
       ("H-S6 random", "VQC vs matched head, frozen random, n=5", "hs6_random", ("regime", "5"), "auc"),
       ("H-S7a", "2-local vs padded readout", "hs7a", ("all",), "auc"),
       ("H-S7b", "2-local vs single readout", "hs7b", ("all",), "auc")]
)
assert len(FAMILY) == 23


# ============================================================ metrics (torch-free)
def _avg_ranks(a):
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    upper = np.cumsum(cnt)
    return (upper - (cnt - 1) / 2.0)[inv]


def auc(y, P):
    """Binary AUC of class 1, or macro one-vs-rest; NaN if a class is absent."""
    C = P.shape[1]
    cols = [1] if C == 2 else range(C)
    out = []
    for c in cols:
        pos = y == c
        npos = int(pos.sum())
        nneg = len(y) - npos
        if npos == 0 or nneg == 0:
            return float("nan")
        r = _avg_ranks(P[:, c])
        out.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(out))


def macro_f1(y, P):
    """scikit-learn f1_score(average='macro', zero_division=0) on the argmax."""
    C = P.shape[1]
    pred = P.argmax(1)
    f = []
    for c in range(C):
        if not ((y == c).any() or (pred == c).any()):
            continue
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        den = 2 * tp + fp + fn
        f.append(2 * tp / den if den else 0.0)
    return float(np.mean(f)) if f else 0.0


def ece(y, P, n_bins=15):
    """Identical to train.metrics.expected_calibration_error."""
    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def selfcheck():
    """The torch-free metrics agree with scikit-learn, when it is installed."""
    try:
        from sklearn.metrics import f1_score, roc_auc_score
    except ImportError:
        print("  (scikit-learn not installed: metric self-check skipped; the "
              "recomputed clean AUC is still checked against every run record)")
        return
    rng = np.random.default_rng(0)
    for C in (2, 8):
        y = rng.integers(0, C, 400)
        P = rng.dirichlet(np.ones(C), 400)
        P[:60] = P[:1]                                   # ties
        ref = roc_auc_score(y, P[:, 1]) if C == 2 else roc_auc_score(
            y, P, multi_class="ovr", average="macro")
        assert abs(auc(y, P) - ref) < 1e-12, "AUC disagrees with scikit-learn"
        assert abs(macro_f1(y, P) - f1_score(y, P.argmax(1), average="macro",
                                             zero_division=0)) < 1e-12
    print("  metric self-check vs scikit-learn: OK")


# ============================================================ records
def _s(v):
    return str(v)


def load_namespace(art, ns):
    recs = []
    for fn in sorted(glob.glob(os.path.join(art, "shards", ns, "*.json"))):
        with open(fn) as f:
            r = json.load(f)
        r["_keys"] = {k: _s(v) for k, v in r.get("keys", {}).items()}
        recs.append(r)
    return recs


def cell_of(keys):
    return tuple(sorted((k, v) for k, v in keys.items() if k not in ARM_LEVEL_KEYS))


def _match(keys, sel, flt):
    arm, rank = sel
    if keys.get("arm") != arm:
        return False
    if rank is not None and keys.get("rank") != rank:
        return False
    for k, v in flt.items():
        if keys.get(k, "") != v:
            return False
    return True


def paired(recs, sel_a, sel_b, flt, value):
    """{cell: {seed: (value_a, value_b)}} for runs present in both arms."""
    A, B = defaultdict(dict), defaultdict(dict)
    ranks = {sel_a: set(), sel_b: set()}
    for r in recs:
        k = r["_keys"]
        for sel, store in ((sel_a, A), (sel_b, B)):
            if _match(k, sel, flt):
                ranks[sel].add(k.get("rank"))
                v = value(r)
                if v is not None and np.isfinite(v):
                    store[cell_of(k)][k["seed"]] = v
    for sel, seen in ranks.items():
        if sel[1] is None and len(seen) > 1:
            # Two ranks of the same arm in one namespace would silently overwrite
            # each other per seed. Refuse rather than mix them.
            raise RuntimeError(f"{sel[0]} appears at ranks {sorted(map(str, seen))}; "
                               f"name the rank in CONTRASTS")
    out = {}
    for c in A:
        seeds = sorted(set(A[c]) & set(B.get(c, {})))
        if seeds:
            out[c] = {s: (A[c][s], B[c][s]) for s in seeds}
    return out


def m_auc(r):
    return (r.get("metrics") or {}).get("auc")


def m_auc_f1sel(r):
    f = (r.get("metrics") or {}).get("test_f1_selected") or {}
    return f.get("auc")


def cget(cell, key):
    return dict(cell).get(key)


# ============================================================ effect sizes
def dz(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def seed_ci(x):
    """Mean and t-based 95% interval over seeds. Descriptive."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan")
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
         9: 2.306, 10: 2.262}.get(n, 1.96 + 2.4 / n)    # t(0.975, n-1); 40 seeds -> 2.02
    h = t * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def per_seed_pooled(pairs, regime=None):
    """Per-seed contrast averaged over the cells of one regime (or all cells),
    equal weight per cell - the same pooling as the confirmatory statistic.
    Only seeds present in every selected cell are used."""
    cells = [c for c in pairs if regime is None or cget(c, "regime") == regime]
    if not cells:
        return {}
    seeds = set.intersection(*(set(pairs[c]) for c in cells))
    return {s: float(np.mean([pairs[c][s][0] - pairs[c][s][1] for c in cells]))
            for s in sorted(seeds)}


def per_seed_slope(pairs):
    x = np.log2(np.array([float(n) for n in REGIMES]))
    by_n = [per_seed_pooled(pairs, n) for n in REGIMES]
    if any(not b for b in by_n):
        return {}
    seeds = set.intersection(*(set(b) for b in by_n))
    out = {}
    for s in sorted(seeds):
        y = np.array([b[s] for b in by_n])
        out[s] = float(np.polyfit(x, y, 1)[0])
    return out


# ============================================================ noise (03_robustness_v2)
def noise_table(art, recs):
    """Per arm and sigma: AUC, macro-F1, ECE, probability spread and G, recomputed
    from the stored per-sigma predictions. Also returns per-seed pooled G for the
    VQC (H-S3 effect sizes)."""
    rows = defaultdict(lambda: defaultdict(list))   # (arm, sigma) -> metric -> [values over runs]
    g_seed = defaultdict(lambda: defaultdict(list))  # sigma -> seed -> [G over cells] (VQC)
    worst = 0.0
    for r in recs:
        k = r["_keys"]
        pf = r.get("predictions_file")
        if not pf:
            continue
        path = os.path.join(art, "predictions", "03_robustness_v2", pf)
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            y = z["labels"].astype(int)
            P = {c: z[c].astype(np.float64) for c in z.files if c != "labels"}
        if "0.00" not in P:
            continue
        base = {"auc": auc(y, P["0.00"]), "f1": macro_f1(y, P["0.00"])}
        rec_auc = ((r.get("curve") or {}).get("0.00") or {}).get("auc") \
            if isinstance((r.get("curve") or {}).get("0.00"), dict) else None
        if rec_auc is not None and np.isfinite(rec_auc):
            worst = max(worst, abs(rec_auc - base["auc"]))
        for s in ("0.00",) + SIGMAS:
            if s not in P:
                continue
            a, f = auc(y, P[s]), macro_f1(y, P[s])
            rows[(k["arm"], s)]["auc"].append(a)
            rows[(k["arm"], s)]["f1"].append(f)
            rows[(k["arm"], s)]["ece"].append(ece(y, P[s]))
            rows[(k["arm"], s)]["spread"].append(float(P[s].max(1).std()))
            if s != "0.00" and base["f1"] > 0:
                g = (base["f1"] - f) / base["f1"] - (base["auc"] - a) / base["auc"]
                rows[(k["arm"], s)]["G"].append(g)
                if k["arm"] == "quantum_vqc":
                    g_seed[s][k["seed"]].append(g)
    out = []
    for (arm, s), m in sorted(rows.items()):
        row = {"arm": arm, "sigma": s, "runs": len(m["auc"])}
        for key in ("auc", "f1", "ece", "spread", "G"):
            if m.get(key):
                row[key] = float(np.mean(m[key]))
                row[key + "_sd"] = float(np.std(m[key], ddof=1)) if len(m[key]) > 1 else float("nan")
        out.append(row)
    return out, g_seed, worst


# ============================================================ main analyses
def absolute_auc(recs_by_ns):
    """Mean and SD over seeds of test AUC, per namespace, arm, cell."""
    out = []
    for ns, recs in recs_by_ns.items():
        by = defaultdict(list)
        for r in recs:
            a = m_auc(r)
            if a is None or not np.isfinite(a):
                continue
            k = r["_keys"]
            arm = k["arm"] + (f" (rank {k['rank']})" if "rank" in k else "")
            by[(arm, cell_of(k))].append(a)
        for (arm, cell), v in sorted(by.items()):
            c = dict(cell)
            out.append({"namespace": ns, "arm": arm, "dataset": c.get("dataset"),
                        "regime": c.get("regime"), "dim": c.get("dim"),
                        "extra": {k: v2 for k, v2 in c.items()
                                  if k not in ("dataset", "regime", "dim", "fp", "aug")},
                        "seeds": len(v), "auc_mean": float(np.mean(v)),
                        "auc_sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")})
    return out


def contrast_tables(recs_by_ns):
    """Per contrast: per-cell and per-regime means, seed SDs, d_z; F1-selected rerun."""
    out = {}
    for name, (ns, a, b, flt) in CONTRASTS.items():
        recs = recs_by_ns.get(ns)
        if not recs:
            out[name] = {"status": f"namespace {ns} not found"}
            continue
        P = paired(recs, a, b, flt, m_auc)
        if not P:
            out[name] = {"status": "no paired runs"}
            continue
        F = paired(recs, a, b, flt, m_auc_f1sel)
        cells = []
        for c, by_seed in sorted(P.items()):
            d = [x - y for x, y in by_seed.values()]
            cells.append({"dataset": cget(c, "dataset"), "regime": cget(c, "regime"),
                          "seeds": len(d), "auc_a": float(np.mean([x for x, _ in by_seed.values()])),
                          "auc_b": float(np.mean([y for _, y in by_seed.values()])),
                          "delta": float(np.mean(d)),
                          "sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                          "dz": dz(d)})
        regimes = sorted({cget(c, "regime") for c in P},
                         key=lambda r: float(r) if r.replace(".", "").isdigit() else 1e9)
        pooled = []
        for n in regimes:
            s = per_seed_pooled(P, n)
            sf = per_seed_pooled(F, n) if F else {}
            m, lo, hi = seed_ci(list(s.values()))
            mf, lof, hif = seed_ci(list(sf.values())) if sf else (float("nan"),) * 3
            pooled.append({"regime": n, "seeds": len(s), "delta": m, "seed_ci": [lo, hi],
                           "dz": dz(list(s.values())), "f1_selected_delta": mf,
                           "f1_selected_seed_ci": [lof, hif], "f1_selected_seeds": len(sf)})
        allc = per_seed_pooled(P)
        out[name] = {"status": "ok", "namespace": ns, "arm_a": a, "arm_b": b,
                     "cells": cells, "pooled_by_regime": pooled,
                     "pooled_all": {"seeds": len(allc), "delta": seed_ci(list(allc.values()))[0],
                                    "dz": dz(list(allc.values()))},
                     "slope": ({"dz": dz(list(per_seed_slope(P).values())),
                                "seeds": len(per_seed_slope(P))}
                               if all(r in regimes for r in REGIMES) else None),
                     "_pairs": P}
    return out


def encoder_dz(recs):
    """H-S4 per-seed analogue: mean over cells of |delta_frozen,s| - |delta_adaptive,s|."""
    by = defaultdict(lambda: defaultdict(dict))     # (dataset, regime) -> fp -> seed -> (vqc, mpfr)
    vals = defaultdict(dict)
    for r in recs:
        k = r["_keys"]
        a = m_auc(r)
        if a is None or k["arm"] not in ("quantum_vqc", "matched_param_fullrank"):
            continue
        vals[(k["dataset"], k["regime"], k.get("fp"), k["seed"])][k["arm"]] = a
    for (ds, n, fp, s), v in vals.items():
        if len(v) == 2:
            by[(ds, n)][fp][s] = v["quantum_vqc"] - v["matched_param_fullrank"]
    per_seed = defaultdict(list)
    for cell, fps in by.items():
        if "all" not in fps or "layer3_only" not in fps:
            continue
        for s in set(fps["all"]) & set(fps["layer3_only"]):
            per_seed[s].append(abs(fps["all"][s]) - abs(fps["layer3_only"][s]))
    n_cells = len(by)
    full = [np.mean(v) for v in per_seed.values() if len(v) == n_cells]
    return dz(full), len(full)


def family_effect_sizes(art, contrasts, hs3_seed, hs4):
    path = os.path.join(art, "family_table.json")
    table = {}
    if os.path.exists(path):
        with open(path) as f:
            table = {r["test"]: r for r in json.load(f)}
    rows = []
    for tid, desc, cname, how, unit in FAMILY:
        d, seeds = float("nan"), 0
        if cname == "hs3":
            per = hs3_seed.get(how[1], {})
            vals = [np.mean(v) for v in per.values()]
            d, seeds = dz(vals), len(vals)
        elif cname == "hs4":
            d, seeds = hs4
        else:
            C = contrasts.get(cname, {})
            if C.get("status") == "ok":
                P = C["_pairs"]
                if how[0] == "regime":
                    v = per_seed_pooled(P, how[1])
                elif how[0] == "slope":
                    v = per_seed_slope(P)
                else:
                    v = per_seed_pooled(P)
                d, seeds = dz(list(v.values())), len(v)
        t = table.get(tid, {})
        est, padj = t.get("estimate"), t.get("p_adj")
        negl = None
        if unit == "auc" and est is not None:
            negl = abs(est) < NEGLIGIBLE
        rows.append({"test": tid, "description": desc, "estimate": est,
                     "ci_lo": t.get("ci_lo"), "ci_hi": t.get("ci_hi"), "p_adj": padj,
                     "verdict": t.get("verdict"), "cohens_dz": d, "dz_seeds": seeds,
                     "negligible_lt_0.01_auc": negl,
                     "note": ("per-seed analogue; the registered statistic takes |.| after "
                              "averaging over seeds") if cname == "hs4" else ""})
    return rows


def provenance(recs_by_ns):
    out = []
    for ns, recs in sorted(recs_by_ns.items()):
        ts = sorted(r.get("timestamp", "") for r in recs if r.get("timestamp"))
        shas = defaultdict(int)
        for r in recs:
            shas[(r.get("git_sha") or "?")[:8]] += 1
        out.append({"namespace": ns, "runs": len(recs), "first": ts[0] if ts else None,
                    "last": ts[-1] if ts else None, "git_shas": dict(shas)})
    return out


def capacity(recs_by_ns):
    out = defaultdict(set)
    for ns in ("01_frozen_tuned", "26_bottleneck_tuned", "27_capacity_tuned"):
        for r in recs_by_ns.get(ns, []):
            c = r.get("capacity")
            if not c:
                continue
            k = r["_keys"]
            arm = k["arm"] + (f"@rank{k['rank']}" if "rank" in k else "")
            key = (ns, arm, k["dataset"], k.get("bn", "learned"), k.get("dim"))
            row = {p: c.get(p) for p in ("backbone", "bottleneck", "head", "classifier", "total")}
            row["angle_scale"] = (r.get("config") or {}).get("angle_scale")
            out[key].add(json.dumps(row, sort_keys=True))
    rows = []
    for (ns, arm, ds, bn, dim), vals in sorted(out.items()):
        # Capacity is fixed by the architecture, so every run of one arm on one
        # dataset must report the same counts. Anything else is flagged, not hidden.
        for v in sorted(vals):
            rows.append({"namespace": ns, "arm": arm, "dataset": ds, "bottleneck_policy": bn,
                         "dim": dim, "consistent": len(vals) == 1, **json.loads(v)})
        if len(vals) != 1:
            print(f"  WARNING: capacity differs across runs of {ns}/{arm}/{ds}/{bn}")
    return rows


# ============================================================ nested (optional)
def nested_tables(art, recs_by_ns, B):
    """Nested-bootstrap pooled delta(n) for d=8, d=16, full data, using the
    family table's own statistic and pooling (13_family_table, imported)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import importlib
    ft = importlib.import_module("13_family_table")
    res = {}
    for name in ("d8", "d16", "full"):
        ns, a, b, flt = CONTRASTS[name]
        recs = recs_by_ns.get(ns, [])
        by = defaultdict(lambda: defaultdict(dict))
        for r in recs:
            k = r["_keys"]
            for tag, sel in (("a", a), ("b", b)):
                if _match(k, sel, flt) and r.get("predictions_file"):
                    by[cell_of(k)][k["seed"]][tag] = r["predictions_file"]
        cells = []
        for cell, seeds in sorted(by.items()):
            ss = sorted(s for s, v in seeds.items() if "a" in v and "b" in v)
            if len(ss) < 2:
                continue
            arms, y0 = {"a": [], "b": []}, None
            for s in ss:
                for tag in ("a", "b"):
                    with np.load(os.path.join(art, "predictions", ns, seeds[s][tag])) as z:
                        y = z["labels"].astype(int)
                        arr = z["probs"] if "probs" in z.files else z[[f for f in z.files if f != "labels"][0]]
                        arms[tag].append(arr.astype(np.float32))
                    if y0 is None:
                        y0 = y
                    elif not np.array_equal(y0, y):
                        raise RuntimeError(f"{ns}: labels differ across runs of one cell")
            c = dict(cell)
            cells.append({"dataset": c.get("dataset"), "regime": c.get("regime"), "seeds": ss,
                          "n_seeds": len(ss), "y": y0,
                          "arms": {"a": np.stack(arms["a"]), "b": np.stack(arms["b"])}})
        if not cells:
            res[name] = {"status": "no data"}
            continue
        t0 = time.time()
        obs, reps = [], []
        for c in cells:
            o, r = ft._cell_job((c, "diff", B, ft.cell_seed("rep_" + name, c)))
            obs.append(o)
            reps.append(r)
        regimes = sorted({c["regime"] for c in cells}, key=lambda r: float(r) if r != "full" else 1e9)
        rows = []
        for n in regimes:
            ix = [i for i, c in enumerate(cells) if c["regime"] == n]
            o = ft._nanmean(np.stack([obs[i] for i in ix]), 0)
            R = ft._nanmean(np.stack([reps[i] for i in ix]), 0)
            s = ft.ci_p(o[0], R[:, 0], B)
            rows.append({"regime": n, "cells": len(ix), **(s or {"status": "too few replicates"})})
        res[name] = {"status": "ok", "rows": rows, "seconds": time.time() - t0}
        print(f"  nested {name}: {len(cells)} cells in {time.time() - t0:.0f}s")
    return res


# ============================================================ output
def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def fmt(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out-tables", default=os.path.join("paper", "tables"))
    ap.add_argument("--nested", action="store_true",
                    help="nested-bootstrap intervals for d=8, d=16, full data (needs scipy)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="primary namespace only")
    args = ap.parse_args()
    art = args.artifacts
    rep = os.path.join(art, "reporting")
    os.makedirs(rep, exist_ok=True)
    os.makedirs(args.out_tables, exist_ok=True)

    print("REPORTING TABLES - descriptive companions to the confirmatory family")
    selfcheck()
    names = {"01_frozen_tuned"} if args.quick else (
        {v[0] for v in CONTRASTS.values()} | {"03_robustness_v2", "25_encoder"})
    recs_by_ns = {}
    for ns in sorted(names):
        recs = load_namespace(art, ns)
        if recs:
            recs_by_ns[ns] = recs
        print(f"  {ns:22s} {len(recs):5d} run records" + ("" if recs else "   (not found)"))

    # integrity: every primary run's recorded AUC against its stored predictions
    worst, checked = 0.0, 0
    for r in recs_by_ns.get("01_frozen_tuned", [])[: (40 if args.quick else None)]:
        pf, a = r.get("predictions_file"), m_auc(r)
        if not pf or a is None:
            continue
        path = os.path.join(art, "predictions", "01_frozen_tuned", pf)
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            P = z["probs"].astype(np.float64)
            y = z["labels"].astype(int)
        row = P.sum(1, keepdims=True)
        if np.abs(row - 1).max() > 1e-6:
            P = P / row
        worst = max(worst, abs(auc(y, P) - a))
        checked += 1
    print(f"  integrity: {checked} primary-namespace runs, max |recorded - recomputed AUC| = {worst:.2e}")
    if checked and worst > 1e-6:
        sys.exit("INTEGRITY FAILURE: recorded and recomputed AUC disagree; nothing written.")

    absolute = absolute_auc(recs_by_ns)
    contrasts = contrast_tables(recs_by_ns)
    noise, hs3_seed, noise_worst = ([], {}, 0.0)
    if "03_robustness_v2" in recs_by_ns:
        noise, hs3_seed, noise_worst = noise_table(art, recs_by_ns["03_robustness_v2"])
    hs4 = encoder_dz(recs_by_ns.get("25_encoder", []))
    effects = family_effect_sizes(art, contrasts, hs3_seed, hs4)
    prov = provenance(recs_by_ns)
    cap = capacity(recs_by_ns)
    nested = nested_tables(art, recs_by_ns, args.B) if args.nested else {}

    public = {k: {kk: vv for kk, vv in v.items() if kk != "_pairs"} for k, v in contrasts.items()}
    blob = {"absolute_auc": absolute, "contrasts": public, "noise": noise,
            "effect_sizes": effects, "provenance": prov, "capacity": cap,
            "nested_exploratory": nested,
            "definitions": "see the module docstring of src/15_reporting_tables.py"}
    with open(os.path.join(rep, "reporting_tables.json"), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1, default=float)
    with open(os.path.join(rep, "capacity.json"), "w") as f:
        json.dump(cap, f, indent=1)
    write_csv(os.path.join(args.out_tables, "absolute_auc.csv"), absolute)
    write_csv(os.path.join(args.out_tables, "effect_sizes.csv"), effects)
    write_csv(os.path.join(args.out_tables, "noise_metrics.csv"), noise)
    write_csv(os.path.join(args.out_tables, "provenance.csv"), prov)
    write_csv(os.path.join(args.out_tables, "capacity.csv"), cap)

    # ---------------- console summary
    print("\n=== 1. Absolute AUC, primary arms, n=5 (mean over seeds) ===")
    for r in absolute:
        if r["namespace"] == "01_frozen_tuned" and r["regime"] == "5" and \
                r["arm"] in ("quantum_vqc", "matched_param_fullrank"):
            print(f"  {r['dataset']:15s} {ARM_LABEL.get(r['arm'], r['arm']):24s} "
                  f"{r['auc_mean']:.4f} (SD {r['auc_sd']:.4f}, {r['seeds']} seeds)")

    print("\n=== 2. Sensitivity: checkpoint chosen by F1 instead of AUC (descriptive, seed-level CI) ===")
    for name in ("primary", "hs1", "hs2"):
        C = contrasts.get(name, {})
        if C.get("status") != "ok":
            continue
        print(f"  {name}:")
        for p in C["pooled_by_regime"]:
            print(f"    n={p['regime']:>4s}  AUC-selected {fmt(p['delta'])}   "
                  f"F1-selected {fmt(p['f1_selected_delta'])} "
                  f"[{fmt(p['f1_selected_seed_ci'][0])}, {fmt(p['f1_selected_seed_ci'][1])}]")

    print("\n=== 3 & 5. Effect sizes for the 23 confirmatory tests ===")
    for e in effects:
        neg = {True: "negligible (<0.01)", False: "", None: ""}[e["negligible_lt_0.01_auc"]]
        print(f"  {e['test']:12s} est {fmt(e['estimate'])}  d_z {fmt(e['cohens_dz'], 2)} "
              f"({e['dz_seeds']} seeds)  {e['verdict'] or '':26s} {neg}")

    print("\n=== 4. Noise, descriptive: G = rel. F1 loss - rel. AUC loss, ECE ===")
    if noise_worst:
        print(f"  clean AUC recomputed vs recorded: max diff {noise_worst:.2e}")
    arms = sorted({r["arm"] for r in noise})
    for s in SIGMAS:
        parts = []
        for a in arms:
            r = next((x for x in noise if x["arm"] == a and x["sigma"] == s), None)
            if r and "G" in r:
                parts.append(f"{a.split('_')[0][:6]} G={r['G']:.3f} ECE={r['ece']:.3f}")
        print(f"  sigma={s}: " + " | ".join(parts))

    print("\n=== 6. Exploratory scope (descriptive means; seed-level CI) ===")
    for name in ("d8", "d16", "full"):
        C = contrasts.get(name, {})
        if C.get("status") != "ok":
            print(f"  {name}: {C.get('status')}")
            continue
        for p in C["pooled_by_regime"]:
            print(f"  {name:5s} n={p['regime']:>4s} delta {fmt(p['delta'])} "
                  f"[{fmt(p['seed_ci'][0])}, {fmt(p['seed_ci'][1])}] ({p['seeds']} seeds)")
    for name, N in nested.items():
        for r in N.get("rows", []):
            print(f"  nested {name:5s} n={r['regime']:>4s} {fmt(r.get('estimate'))} "
                  f"[{fmt(r.get('ci_lo'))}, {fmt(r.get('ci_hi'))}]")

    print("\n=== 7. Provenance ===")
    for p in prov:
        print(f"  {p['namespace']:22s} {p['runs']:5d} runs  {p['first']} -> {p['last']}  "
              f"commits {p['git_shas']}")
    print("  Compare with the analysis plan's commit date:")
    print("    git log --reverse --date=iso --format=\"%h %ad %s\" -- docs/analysis_plan.md")

    print("\n=== 8. Trainable capacity (from the run records) ===")
    for c in cap:
        if c["dataset"] in ("breastmnist", "bloodmnist") and c["arm"] in ("quantum_vqc", "matched_param_fullrank"):
            print(f"  {c['namespace']:20s} {c['arm']:24s} {c['dataset']:12s} {c['bottleneck_policy']:8s} "
                  f"projection {c['bottleneck']}, head {c['head']}, classifier {c['classifier']}, total {c['total']}")

    print(f"\nWritten: {rep}/reporting_tables.json, capacity.json; "
          f"{args.out_tables}/*.csv")


if __name__ == "__main__":
    main()
