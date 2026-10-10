"""
THE PAPER'S FIGURES, drawn from the bootstrap caches and nothing else.

    python src/eval/generate_paper_plots.py                # verify, then draw
    python src/eval/generate_paper_plots.py --check-only   # verify only
    python src/eval/generate_paper_plots.py --out paper/figures

WHY THIS FILE IS BUILT THE WAY IT IS
------------------------------------
Every number in a figure must be the number in the tables. So this script does
not re-run any analysis and does not read predictions or shards. It reads the
per-cell bootstrap replicates that 13_family_table.py and 14_exploratory_table.py
saved on the cluster (artifacts/family_table_cache, artifacts/exploratory_cache)
and pools them exactly as those scripts do: equal weight per cell, the mean over
cells taken within each replicate, 95% percentile intervals.

Before drawing anything it re-derives every confirmatory row from the caches and
compares estimate, interval, p and BH-adjusted p with artifacts/family_table.json,
and every matching row of artifacts/exploratory_table.json. Any disagreement
beyond 1e-9 aborts: a figure that disagrees with its table is worse than no
figure. The pooling functions below are written out again rather than imported
from 13_family_table.py, so the comparison is an independent reproduction, and so
this script needs only numpy and matplotlib (no torch, no scipy, no GPU).

WHAT IS AND IS NOT HERE
-----------------------
Figure 1 (pipeline diagram) is drawn in the manuscript itself (TikZ); it carries
no data. The d=8, d=16 and full-data results came from 04_statistical_analysis.py
and have no replicate cache, so they are reported as a table, not plotted here.

Every figure caption must carry the evidential tier of what it shows -
confirmatory, pre-specified follow-up, or exploratory. figure_data.json, written
next to the figures, holds every plotted value with its interval, for the text
and for reviewers.
"""
import argparse
import json
import os
import sys

import numpy as np

REGIMES = (5, 10, 20, 50, 100)
SIGMAS = ("0.05", "0.10", "0.15", "0.20")
FAMILY_SIZE = 23
ALPHA = 0.05
TOL = 1e-9
B_DEFAULT = 2000

DATASET_LABEL = {"bloodmnist": "BloodMNIST", "breastmnist": "BreastMNIST",
                 "pathmnist": "PathMNIST", "pneumoniamnist": "PneumoniaMNIST"}

# The confirmatory family, exactly as 13_family_table.TESTS declares it:
# test id -> (cache, how the statistic is read, predicted sign).
FAMILY = (
    [("H-P1", "primary", ("regime", 5), +1), ("H-P2", "primary", ("slope",), -1)]
    + [(f"H-S1 n={n}", "hs1", ("regime", n), {5: -1, 10: -1, 20: 0, 50: +1, 100: +1}[n])
       for n in REGIMES]
    + [(f"H-S2 n={n}", "hs2", ("regime", n), -1) for n in REGIMES]
    + [(f"H-S3 s={s}", "hs3", ("index", i), +1) for i, s in enumerate(SIGMAS)]
    + [("H-S4", "hs4", ("index", 0), +1),
       ("H-S5a", "hs5", ("regime", 5), +1), ("H-S5b", "hs5", ("slope",), -1),
       ("H-S6 pca", "hs6_pca", ("regime", 5), +1),
       ("H-S6 random", "hs6_random", ("regime", 5), +1),
       ("H-S7a", "hs7a", ("index", 0), +1), ("H-S7b", "hs7b", ("index", 0), +1)]
)
assert len(FAMILY) == FAMILY_SIZE
BY_REGIME = {"primary": True, "hs1": True, "hs2": True, "hs3": False, "hs4": False,
             "hs5": True, "hs6_pca": True, "hs6_random": True, "hs7a": False,
             "hs7b": False}
EXPLORATORY_BY_REGIME = {"gap_control": False, "gap_vqc_minus_control": False}


# ============================================================ statistics
def nanmean(x, axis=None):
    """Mean ignoring NaN; all-NaN slices give NaN without a warning."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    n = ok.sum(axis=axis)
    s = np.where(ok, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def ci_p(estimate, reps, B=B_DEFAULT):
    """Percentile 95% interval and +1-corrected two-sided p, as in 13_family_table."""
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    if r.size < B // 2:
        return None
    lo, hi = np.percentile(r, [2.5, 97.5])
    k_le, k_ge = int((r <= 0).sum()), int((r >= 0).sum())
    p = min(1.0, 2 * min(k_le + 1, k_ge + 1) / (r.size + 1))
    return {"estimate": float(estimate), "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(p), "b_valid": int(r.size)}


def slope(x, Y):
    """OLS slope of Y on x along the last axis."""
    xc = x - x.mean()
    return (Y - Y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def benjamini_hochberg(p, m):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    n = int(ok.sum())
    out = np.full_like(p, np.nan)
    if n == 0:
        return out
    m = max(m, n)
    order = np.argsort(np.where(ok, p, np.inf))[:n]
    adj = p[order] * m / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


class Group:
    """One cache file: per-cell observed statistic and its B replicates."""

    def __init__(self, path, by_regime):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.obs = np.asarray(z["obs"], float)                 # [cells, k]
        self.reps = np.asarray(z["reps"], float)               # [cells, B, k]
        self.cells = [(c[0], int(c[1]), len(c[2])) for c in self.meta["cells"]]
        self.by_regime = by_regime
        self.B = int(self.meta["B"])
        if len(self.cells) != 20 or self.obs.shape[0] != 20:
            raise RuntimeError(f"{path}: expected 20 cells, found {len(self.cells)}")
        if len({c[2] for c in self.cells}) != 1:
            raise RuntimeError(f"{path}: unequal seed counts across cells")
        self.n_seeds = self.cells[0][2]

    def _idx(self, regime=None, exclude=None):
        return [i for i, (ds, n, _) in enumerate(self.cells)
                if (regime is None or n == regime) and ds != exclude]

    def pooled(self, regime=None, exclude=None):
        """Equal-weight mean over the selected cells, within each replicate."""
        ix = self._idx(regime, exclude)
        return nanmean(self.obs[ix], 0), nanmean(self.reps[ix], 0)

    def at(self, n, k=0, exclude=None):
        o, r = self.pooled(n, exclude)
        return ci_p(o[k], r[:, k], self.B)

    def overall(self, k):
        o, r = self.pooled(None)
        return ci_p(o[k], r[:, k], self.B)

    def regime_slope(self):
        x = np.log2(np.array(REGIMES, float))
        pooled = [self.pooled(n) for n in REGIMES]
        o = np.array([p[0][0] for p in pooled])
        R = np.stack([p[1][:, 0] for p in pooled], axis=1)
        R = R[np.isfinite(R).all(1)]
        return ci_p(float(slope(x, o)), slope(x, R), self.B)

    def cell(self, dataset, n, k=0):
        i = next(i for i, (ds, nn, _) in enumerate(self.cells) if ds == dataset and nn == n)
        return ci_p(self.obs[i, k], self.reps[i, :, k], self.B)

    def datasets(self):
        return sorted({c[0] for c in self.cells})

    def series(self, k=0):
        return [self.at(n, k) for n in REGIMES]


# ============================================================ verification
def _close(a, b):
    return abs(float(a) - float(b)) <= TOL


def verify(art, fam, exp):
    """Re-derive every tabled number from the caches; abort on any disagreement."""
    problems, checked = [], 0
    path = os.path.join(art, "family_table.json")
    with open(path) as f:
        table = {r["test"]: r for r in json.load(f)}
    rows = []
    for tid, g, how, sign in FAMILY:
        G = fam[g]
        s = (G.at(how[1]) if how[0] == "regime" else
             G.regime_slope() if how[0] == "slope" else G.overall(how[1]))
        rows.append((tid, sign, s))
    adj = benjamini_hochberg([s["p"] for _, _, s in rows], FAMILY_SIZE)
    for (tid, sign, s), a in zip(rows, adj):
        t = table.get(tid)
        if t is None or t.get("status") != "ok":
            problems.append(f"{tid}: missing or not computed in family_table.json")
            continue
        for key in ("estimate", "ci_lo", "ci_hi", "p"):
            if not _close(s[key], t[key]):
                problems.append(f"{tid} {key}: cache {s[key]:+.10f} vs table {t[key]:+.10f}")
        if not _close(a, t["p_adj"]):
            problems.append(f"{tid} p_adj: recomputed {a:.10f} vs table {t['p_adj']:.10f}")
        sig = a <= ALPHA
        verdict = ("difference (no prediction)" if sign == 0 and sig else
                   "no difference" if sign == 0 else
                   "SUPPORTED" if sig and np.sign(s["estimate"]) == sign else
                   "OPPOSITE to prediction" if sig else "not supported")
        if verdict != t["verdict"]:
            problems.append(f"{tid} verdict: recomputed {verdict!r} vs table {t['verdict']!r}")
        checked += 1
    hp1 = fam["primary"].at(5)["estimate"]
    if round(hp1, 4) != 0.0142:
        problems.append(f"anchor: H-P1 delta(5) = {hp1:+.4f}, expected +0.0142")

    n_exp, skipped = 0, []
    epath = os.path.join(art, "exploratory_table.json")
    if os.path.exists(epath):
        with open(epath) as f:
            erows = json.load(f)
        for r in erows:
            g, tag = r.get("analysis"), r.get("row", "")
            if g not in exp:
                skipped.append(f"{g}/{tag}")
                continue
            G = exp[g]
            if tag.startswith("n="):
                s = G.at(int(tag.split("=")[1].split()[0]))
            elif tag.startswith("sigma="):
                s = G.overall(SIGMAS.index(tag.split("=")[1]))
            elif tag == "slope" or "slope" in tag:
                s = G.regime_slope()
            elif "delta_0(5)" in tag:
                s = G.at(5)
            else:
                skipped.append(f"{g}/{tag}")
                continue
            for key in ("estimate", "ci_lo", "ci_hi", "p"):
                if not _close(s[key], r[key]):
                    problems.append(f"exploratory {g}/{tag} {key}: cache {s[key]:+.10f} "
                                    f"vs table {r[key]:+.10f}")
            n_exp += 1
    print(f"Verified: {checked}/{FAMILY_SIZE} confirmatory rows (estimate, CI, p, p_adj, "
          f"verdict) and {n_exp} exploratory rows reproduced from the caches.")
    if skipped:
        print(f"  not checkable here ({len(skipped)}): {', '.join(skipped)}")
    if problems:
        print("\nVERIFICATION FAILED - no figure has been drawn:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"  anchor H-P1 delta(5) = {hp1:+.4f}  OK")


# ============================================================ drawing
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Categorical slots 1-4 of the reference palette, in fixed order; validated with
# scripts/validate_palette.js (all checks pass; slots 3-4 are below 3:1 contrast,
# so every series also has its own marker shape and a legend entry).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
NEUTRAL = "#898781"
MaxNLocator = None               # bound in setup_matplotlib, so --check-only needs no matplotlib
COL1, COL2 = 3.5, 7.16           # IEEE single- and double-column widths, inches


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global MaxNLocator
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embedded TrueType for IEEE PDF eXpress
    })
    return plt


def legend_above(ax, ncol=1, **kw):
    """Legend outside the plotting area, so it can never cover a data point."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, borderaxespad=0,
              handletextpad=0.4, columnspacing=1.0, **kw)


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color=AXIS, lw=0.8, zorder=0)


def n_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(REGIMES)
    ax.set_xticklabels([str(n) for n in REGIMES])
    ax.minorticks_off()
    ax.set_xlabel("labelled images per class, n")
    ax.set_xlim(4.2, 120)


def plot_series(ax, stats, i, label, dodge=0.0, color=None, marker=None, ls="-"):
    """One Δ(n) series with 95% intervals; dodge is a multiplicative x offset."""
    x = np.array(REGIMES, float) * 10 ** dodge
    y = np.array([s["estimate"] for s in stats])
    lo = y - np.array([s["ci_lo"] for s in stats])
    hi = np.array([s["ci_hi"] for s in stats]) - y
    ax.errorbar(x, y, yerr=[lo, hi], color=color or SERIES[i], marker=marker or MARKERS[i],
                ls=ls, capsize=1.8, elinewidth=0.8, capthick=0.8, label=label,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)


def save(fig, out, name, data, record):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    record[name] = data
    print(f"  {name}.pdf / .png")


def rows_by_n(stats):
    return {f"n={n}": s for n, s in zip(REGIMES, stats)}


def fig_forest(plt, fam, out, record):
    """Fig. 2 - the confirmatory family. Three panels, because the statistics
    have three different units; one axis per unit, never two scales on one axis."""
    with open(os.path.join(ART, "family_table.json")) as f:
        table = json.load(f)
    groups = [("Δ AUC", [r for r in table if r["test"] not in ("H-P2", "H-S5b")
                          and not r["test"].startswith("H-S3")]),
              ("slope of Δ AUC per doubling of n", [r for r in table
                                                    if r["test"] in ("H-P2", "H-S5b")]),
              ("relative F1 loss − relative AUC loss", [r for r in table
                                                       if r["test"].startswith("H-S3")])]
    style = {"SUPPORTED": (SERIES[0], "o", "supported"),
             "OPPOSITE to prediction": (SERIES[1], "s", "opposite to prediction"),
             "difference (no prediction)": (SERIES[2], "^", "difference, no prediction"),
             "not supported": (NEUTRAL, "o", "not supported")}
    fig, axes = plt.subplots(3, 1, figsize=(COL1, 5.2), sharex=False,
                             gridspec_kw={"height_ratios": [len(g[1]) + 1.2 for g in groups]})
    seen = set()
    for ax, (xlabel, rows) in zip(axes, groups):
        for j, r in enumerate(rows):
            c, m, lab = style[r["verdict"]]
            filled = r["verdict"] != "not supported"
            ax.errorbar(r["estimate"], j, xerr=[[r["estimate"] - r["ci_lo"]],
                                                [r["ci_hi"] - r["estimate"]]],
                        color=c, marker=m, ms=4, capsize=1.5, elinewidth=0.8,
                        mfc=c if filled else "white", mec=c, ls="none",
                        label=None if lab in seen else lab, zorder=3)
            seen.add(lab)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["test"].replace(" s=", " σ=") for r in rows])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))   # no colliding tick labels
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.5)
        if not rows[0]["test"].startswith("H-S3"):
            zero_line(ax, vertical=True)
    handles, labels = axes[0].get_legend_handles_labels()
    for a in axes[1:]:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.0)
    save(fig, out, "fig2_confirmatory_forest", table, record)
    plt.close(fig)


def fig_where(plt, fam, exp, out, record):
    """Fig. 3 (lead) - Δ(n) with the projection learned, frozen to PCA, frozen random."""
    s = {"learned (= H-P1/H-P2)": fam["primary"].series(),
         "frozen PCA": exp["hs6_followup_pca"].series(),
         "frozen random": exp["hs6_followup_random"].series()}
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    zero_line(ax)
    for i, (lab, st) in enumerate(s.items()):
        plot_series(ax, st, i, lab, dodge=(i - 1) * 0.022)
    n_axis(ax)
    ax.set_ylabel("Δ AUC, VQC − matched head")
    legend_above(ax, ncol=3, title="projection 256 → 4", title_fontsize=7)
    save(fig, out, "fig3_where_the_advantage_lives",
         {k: rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)


def fig_restriction(plt, fam, exp, out, record):
    """Fig. 4 - the restriction test, both runs, with the VQC effect for scale."""
    s = {"rank 0 − rank 8, original H-S5 (1e-3, 10 seeds)": fam["hs5"].series(),
         "rank 0 − rank 8, follow-up (1e-2, 40 seeds)": exp["hs5_followup"].series()}
    ref = fam["primary"].series()
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    zero_line(ax)
    for i, (lab, st) in enumerate(s.items()):
        plot_series(ax, st, i, lab, dodge=(i - 0.5) * 0.03)
    y = [r["estimate"] for r in ref]
    ax.plot(REGIMES, y, color=NEUTRAL, ls="--", lw=1.0, marker="x", ms=4,
            label="VQC − matched head (H-P1/H-P2), for scale", zorder=2)
    n_axis(ax)
    ax.set_ylabel("Δ AUC")
    legend_above(ax, fontsize=6.5)
    save(fig, out, "fig4_restriction",
         {**{k: rows_by_n(v) for k, v in s.items()}, "reference_primary": rows_by_n(ref)},
         record)
    plt.close(fig)


def fig_scope(plt, fam, out, record):
    """Fig. 5 - per-dataset Δ(n) and leave-one-dataset-out Δ(5). Exploratory."""
    G = fam["primary"]
    dss = G.datasets()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    zero_line(a)
    per = {}
    for i, ds in enumerate(dss):
        st = [G.cell(ds, n) for n in REGIMES]
        per[ds] = rows_by_n(st)
        plot_series(a, st, i, DATASET_LABEL.get(ds, ds), dodge=(i - 1.5) * 0.018)
    n_axis(a)
    a.set_ylabel("Δ AUC, VQC − matched head")
    a.set_title("(a) each dataset (40 seeds per cell)", loc="left", pad=16)
    legend_above(a, ncol=4)

    loo = [("all four", G.at(5))] + [(f"excl. {DATASET_LABEL.get(d, d).replace('MNIST', '')}",
                                      G.at(5, exclude=d)) for d in dss]
    zero_line(b, vertical=True)
    for j, (lab, s) in enumerate(loo):
        c = SERIES[0] if j == 0 else INK2
        b.errorbar(s["estimate"], j, xerr=[[s["estimate"] - s["ci_lo"]],
                                           [s["ci_hi"] - s["estimate"]]],
                   color=c, marker="o", ms=4, capsize=1.5, elinewidth=0.8, ls="none")
    b.set_yticks(range(len(loo)))
    b.set_yticklabels([l for l, _ in loo])
    b.set_ylim(len(loo) - 0.4, -0.6)
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color=GRID, lw=0.5)
    b.set_xlabel("pooled Δ AUC at n = 5")
    b.xaxis.set_major_locator(MaxNLocator(nbins=5))
    b.set_title("(b) leave one dataset out", loc="left", pad=16)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig5_scope", {"per_dataset": per,
                                  "leave_one_out_n5": {l: s for l, s in loo}}, record)
    plt.close(fig)


def fig_function_class(plt, fam, exp, out, record):
    """Fig. 6 - each VQC against a direct fit over its own classical function class."""
    s = {"single encoding − Fourier {−1,0,1}⁴ (H-S2)": fam["hs2"].series(),
         "re-uploading − Fourier {−2,…,2}⁴ (E7)": exp["reupload_vs_fourier_r2"].series()}
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    zero_line(ax)
    for i, (lab, st) in enumerate(s.items()):
        plot_series(ax, st, i, lab, dodge=(i - 0.5) * 0.03)
    n_axis(ax)
    ax.set_ylabel("Δ AUC, VQC − classical fit")
    legend_above(ax, fontsize=6.5)
    save(fig, out, "fig6_function_class", {k: rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)


def fig_noise(plt, fam, exp, out, record):
    """Fig. 7 - F1 collapses while AUC holds, for both heads; the VQC's excess."""
    sig = np.array([float(s) for s in SIGMAS])
    vqc = [fam["hs3"].overall(i) for i in range(4)]
    ctl = [exp["gap_control"].overall(i) for i in range(4)]
    dif = [exp["gap_vqc_minus_control"].overall(i) for i in range(4)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.3))

    def draw(ax, stats, i, lab, dx):
        y = np.array([s["estimate"] for s in stats])
        ax.errorbar(sig + dx, y, yerr=[y - [s["ci_lo"] for s in stats],
                                       [s["ci_hi"] for s in stats] - y],
                    color=SERIES[i], marker=MARKERS[i], capsize=1.8, elinewidth=0.8,
                    label=lab, markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    draw(a, vqc, 0, "VQC (H-S3)", -0.003)
    draw(a, ctl, 1, "matched classical head", 0.003)
    a.set_ylim(0, None)
    a.set_title("(a) relative F1 loss − relative AUC loss", loc="left", pad=16)
    legend_above(a, ncol=2)
    zero_line(b)
    draw(b, dif, 0, "VQC − matched head, paired", 0)
    b.set_title("(b) the VQC's excess, paired", loc="left", pad=16)
    for ax in (a, b):
        ax.set_xticks(sig)
        ax.set_xlabel("sensor-noise σ (native 28×28 pixels)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig7_noise", {"vqc_gap": dict(zip(SIGMAS, vqc)),
                                  "control_gap": dict(zip(SIGMAS, ctl)),
                                  "paired_difference": dict(zip(SIGMAS, dif))}, record)
    plt.close(fig)


# ============================================================ main
ART = "artifacts"


def main():
    global ART
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=os.path.join("paper", "figures"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    ART = args.artifacts

    fdir, edir = os.path.join(ART, "family_table_cache"), os.path.join(ART, "exploratory_cache")
    fam = {g: Group(os.path.join(fdir, f"{g}.npz"), BY_REGIME[g]) for g in BY_REGIME}
    exp = {os.path.splitext(f)[0]: Group(os.path.join(edir, f),
                                         EXPLORATORY_BY_REGIME.get(os.path.splitext(f)[0], True))
           for f in sorted(os.listdir(edir)) if f.endswith(".npz")}
    print(f"Loaded {len(fam)} confirmatory and {len(exp)} exploratory caches from {ART}/")
    for name, G in list(fam.items()) + list(exp.items()):
        if G.B != B_DEFAULT:
            sys.exit(f"{name}: cache has B={G.B}, not {B_DEFAULT} - testing cache, not plottable")

    verify(ART, fam, exp)
    if args.check_only:
        return

    os.makedirs(args.out, exist_ok=True)
    plt = setup_matplotlib()
    record = {}
    print(f"\nDrawing into {args.out}/")
    fig_forest(plt, fam, args.out, record)
    fig_where(plt, fam, exp, args.out, record)
    fig_restriction(plt, fam, exp, args.out, record)
    fig_scope(plt, fam, args.out, record)
    fig_function_class(plt, fam, exp, args.out, record)
    fig_noise(plt, fam, exp, args.out, record)
    with open(os.path.join(args.out, "figure_data.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"  figure_data.json - every plotted value with its interval")

    loo = record["fig5_scope"]["leave_one_out_n5"]
    print("\nLeave-one-dataset-out, Δ(5), nested (exploratory):")
    for k, s in loo.items():
        print(f"  {k:28s} {s['estimate']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
