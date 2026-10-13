"""
THE PAPER'S FIGURES - every mark drawn from saved results, nothing typed in.

    python src/eval/generate_paper_plots.py                # verify, then draw
    python src/eval/generate_paper_plots.py --check-only   # verify only
    python src/eval/generate_paper_plots.py --out paper/figures

WHERE EVERY NUMBER COMES FROM
-----------------------------
  bootstrap caches        artifacts/family_table_cache, artifacts/exploratory_cache
                          (per-cell replicates written by 13_family_table.py and
                          14_exploratory_table.py on the cluster)
  run records             artifacts/reporting/capacity.json, written by
                          15_reporting_tables.py from the shards' own
                          "capacity" field (parameter counts per component)
  trained model           artifacts/reproduced/function_class.npz, written by
                          16_reproduce_for_figure.py (PennyLane's own outputs
                          from a retrained confirmatory run)

No estimate, interval, parameter count, circuit depth or dimension is written
into this file. The bottleneck dimension d is read from the run records, the
circuit depth L from the head's parameter count (3 L d), the feature width from
the projection's (F d + d). A figure whose input is missing is skipped with a
message; there is no fallback to made-up values.

VERIFICATION BEFORE DRAWING
---------------------------
Every confirmatory row (estimate, interval, p, BH-adjusted p, verdict) and every
row of exploratory_table.json is re-derived from the caches and compared with
the tables to 1e-9. Any disagreement aborts before a single figure is drawn.
The pooling functions are written out again here rather than imported, so the
comparison is an independent reproduction, and so this script needs only numpy
and matplotlib.

LABELS
------
Figures carry plain-language labels only. Hypothesis identifiers belong in the
manuscript's test table, which maps each figure element back to the analysis
plan. Every caption must state the evidential tier of what it shows.

STYLE
-----
Colour has one meaning throughout: blue is the primary contrast (VQC head minus
the parameter-matched classical head, projection learned), orange and green are
the other conditions of the same figure, grey is a reference. Every series also
has its own marker, so nothing depends on colour alone. Palette validated with
the dataviz skill's validator (CVD separation and normal-vision floor pass).
Intervals are shaded 95% nested-bootstrap intervals. 3D is used once, where the
data genuinely are a surface.
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



# ============================================================ style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
TINT_BLUE, TINT_WARM = "#eef4fc", "#faf9f7"
COL1, COL2 = 3.5, 7.16           # IEEE single- and double-column widths, inches
VERDICT_STYLE = {
    "SUPPORTED": (BLUE, "o", True, "supported"),
    "OPPOSITE to prediction": (ORANGE, "s", True, "opposite to prediction"),
    "difference (no prediction)": (GREEN, "^", True, "difference, no direction predicted"),
    "not supported": (GREY, "o", False, "not supported"),
}
DIV = None


def setup_matplotlib():
    global DIV
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # Diverging pair of the reference palette: blue <-> red, neutral grey midpoint.
    DIV = LinearSegmentedColormap.from_list(
        "div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2b0ae", "#e34948", "#9e2423"])
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embedded TrueType for IEEE PDF eXpress
        "axes.titlelocation": "left",
    })
    return plt


def points(v):
    """AUC difference -> 'AUC points' (x100), the unit every difference figure uses."""
    return v * 100.0


def points_axis(ax, which="y"):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-12 else f"{v * 100:+.4g}".replace("-", "\u2212")))


def n_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(REGIMES)
    ax.set_xticklabels([str(n) for n in REGIMES])
    ax.minorticks_off()
    ax.set_xlim(REGIMES[0] * 0.86, REGIMES[-1] * 1.15)
    ax.set_xlabel("labelled images per class")


def band(ax, stats, color, marker, label, ls="-", alpha=0.14, z=3, x=None):
    x = np.array(REGIMES if x is None else x, float)
    y = np.array([s["estimate"] for s in stats])
    ax.fill_between(x, [s["ci_lo"] for s in stats], [s["ci_hi"] for s in stats],
                    color=color, alpha=alpha, lw=0, zorder=z - 1)
    ax.plot(x, y, color=color, ls=ls, marker=marker, mec="white", mew=0.6,
            label=label, zorder=z)


def regions(ax, top_label, bottom_label):
    """Shade 'ahead' regions so the sign of every point reads without the axis."""
    lo, hi = ax.get_ylim()
    ax.axhspan(0, max(hi, 0), color=TINT_BLUE, zorder=0, lw=0)
    ax.axhspan(min(lo, 0), 0, color=TINT_WARM, zorder=0, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_ylim(lo, hi)
    x1 = ax.get_xlim()[1]
    ax.text(x1 / 1.03, hi - 0.06 * (hi - lo), top_label, ha="right", va="top",
            fontsize=7, color=INK2)
    ax.text(x1 / 1.03, lo + 0.05 * (hi - lo), bottom_label, ha="right", va="bottom",
            fontsize=7, color=INK2)


def legend_above(ax, ncol=1, **kw):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, borderaxespad=0,
              handlelength=1.8, handletextpad=0.5, columnspacing=1.2, **kw)


def save(fig, out, name, data, record):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    record[name] = data
    print(f"  {name}.pdf / .png")


def rows_by_n(stats):
    return {f"n={n}": s for n, s in zip(REGIMES, stats)}


# ============================================================ data from run records
class Architecture:
    """Every architectural number the figures print, derived from run records."""

    def __init__(self, cap_rows):
        def pick(ns, arm, policy, binary=True):
            rows = [r for r in cap_rows if r["namespace"] == ns and r["arm"] == arm
                    and r["bottleneck_policy"] == policy and r.get("consistent", True)]
            rows = [r for r in rows if (r["dataset"] in ("breastmnist", "pneumoniamnist")) == binary]
            return rows[0] if rows else None

        self.vqc = pick("01_frozen_tuned", "quantum_vqc", "learned")
        self.ctl = pick("01_frozen_tuned", "matched_param_fullrank", "learned")
        self.vqc_multi = pick("01_frozen_tuned", "quantum_vqc", "learned", binary=False)
        self.frozen = pick("26_bottleneck_tuned", "quantum_vqc", "pca")
        self.r0 = pick("27_capacity_tuned", "low_rank@rank0", "learned")
        self.r8 = pick("27_capacity_tuned", "low_rank@rank8", "learned")
        if self.vqc is None or self.ctl is None:
            raise RuntimeError("capacity.json lacks the primary arms")
        self.d = int(self.vqc["dim"])
        self.F = self.vqc["bottleneck"] // self.d - 1            # projection = F d + d
        self.L = self.vqc["head"] // (3 * self.d)                # StronglyEntanglingLayers: 3 L d
        self.C = self.vqc["classifier"] // (self.d + 1)          # classifier = d C + C
        self.angle = self.vqc.get("angle_scale")
        assert self.F * self.d + self.d == self.vqc["bottleneck"], "projection count not F*d + d"
        assert 3 * self.L * self.d == self.vqc["head"], "head count not 3*L*d"
        assert self.ctl["head"] == self.vqc["head"], "primary arms are not parameter-matched"


def angle_label(a):
    if a is None:
        return "tanh · s"
    for k, txt in ((1, "π"), (2, "π/2"), (4, "π/4")):
        if abs(a - np.pi / k) < 1e-6:
            return f"tanh(·) × {txt}"
    return f"tanh(·) × {a:.3g}"


# ============================================================ figure 1: overview
def fig_overview(plt, arch, out, record):
    from matplotlib.patches import FancyBboxPatch
    d, F, L, C = arch.d, arch.F, arch.L, arch.C
    v, fz = arch.vqc, arch.frozen
    share = v["bottleneck"] / v["total"] if v.get("total") else float("nan")
    fig = plt.figure(figsize=(COL2, 3.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.3, 1],
                          hspace=0.28, wspace=0.12)

    # ---- (a) the pipeline
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    ax.set_title("(a) the hybrid pipeline; only the head differs between arms", fontsize=7.5)

    def box(x, w, title, sub, y=8, h=17, fill="#ffffff", edge=AXIS, bold=False, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fill, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.15)
        if sub:
            ax.text(x + w / 2, y + h * 0.2, sub, ha="center", va="center", fontsize=6.2,
                    color=INK2, linespacing=1.1)

    def arrow(x0, x1, y=16.5):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))

    box(0.5, 11, "image", "28×28\n→ 224×224")
    arrow(12.2, 14.2)
    box(14.5, 13, "ResNet-18\nto layer 3", f"frozen\n{v['backbone']} trainable")
    arrow(28.2, 30.2)
    box(30.5, 15.5, f"projection\n{F} → {d}", f"learned\n{v['bottleneck']:,} parameters",
        fill=TINT_BLUE, edge=BLUE, bold=True)
    ax.text(38.25, 4.2, f"{100 * share:.0f}% of trainable parameters", ha="center",
            fontsize=6.6, color=BLUE, fontweight="bold")
    if fz is not None:
        ax.text(38.25, 28.5, f"control: frozen PCA or random,\n{fz['bottleneck']} trainable",
                ha="center", fontsize=6.2, color=INK2, linespacing=1.1)
    arrow(46.4, 48.4)
    box(48.7, 11.5, "squash\nto angles", angle_label(arch.angle) + "\nall arms")
    arrow(60.6, 62.6)
    box(62.9, 17, f"VQC head\n{v['head']} parameters", "", y=17, h=8, edge=BLUE, fs=6.6)
    box(62.9, 17, f"matched classical\n{arch.ctl['head']} parameters", "", y=8, h=8, edge=GREY, fs=6.6)
    ax.text(71.4, 4.2, "the only part that differs", ha="center", fontsize=6.2, color=INK2)
    arrow(80.3, 82.3)
    box(82.6, 16.9, f"classifier\n{d} → C", f"{v['classifier']} parameters\n(C = {C})")

    # ---- (b) the circuit, drawn from d and L
    axc = fig.add_subplot(gs[1, 0])
    axc.axis("off")
    axc.set_title(f"(b) VQC head: {d} qubits, {L} layers, {v['head']} trainable angles", fontsize=7.5)
    ranges = [(l % (d - 1)) + 1 if d > 1 else 0 for l in range(L)]   # PennyLane's default ranges
    x_meter = 0.2 + 1.1 + L * (0.9 + 0.55 * d + 0.45) + 0.1     # same steps as the loop below
    end = x_meter + 0.45
    axc.set_xlim(-1.5, end + 0.3); axc.set_ylim(-1.0, d - 0.3)
    for w in range(d):
        yy = d - 1 - w
        axc.plot([-0.5, end], [yy, yy], color=INK2, lw=0.7, zorder=0)
        axc.text(-0.7, yy, f"z{w + 1}", ha="right", va="center", fontsize=7)

    def gate(xc, yy, txt, ec, wdt=0.8):
        axc.add_patch(FancyBboxPatch((xc - wdt / 2, yy - 0.28), wdt, 0.56,
                                     boxstyle="round,pad=0.02,rounding_size=0.08",
                                     fc="white", ec=ec, lw=0.8, zorder=2))
        axc.text(xc, yy, txt, ha="center", va="center", fontsize=6, zorder=3)

    x = 0.2
    for w in range(d):
        gate(x, d - 1 - w, "RY", ORANGE)
    x += 1.1
    for r in ranges:
        for w in range(d):
            gate(x, d - 1 - w, "Rot", BLUE)
        x += 0.9
        for w in range(d):
            t = (w + r) % d
            yc, yt = d - 1 - w, d - 1 - t
            axc.plot([x, x], [yc, yt], color=INK2, lw=0.7, zorder=1)
            axc.plot(x, yc, "o", color=INK2, ms=2.8, zorder=3)
            axc.plot(x, yt, "o", mfc="white", mec=INK2, ms=6, zorder=3)
            axc.plot([x - 0.12, x + 0.12], [yt, yt], color=INK2, lw=0.6, zorder=4)
            axc.plot([x, x], [yt - 0.12, yt + 0.12], color=INK2, lw=0.6, zorder=4)
            x += 0.55
        x += 0.45
    for w in range(d):
        gate(x + 0.1, d - 1 - w, "⟨X⟩", INK2)
    axc.text(0.2, -0.85, "encode", ha="center", fontsize=6.2, color=ORANGE)
    axc.text(1.25, -0.85, f"{L} × (rotate every qubit, then a ring of CNOTs)",
             ha="left", fontsize=6.2, color=BLUE)

    # ---- (c) where the trainable parameters are
    axb = fig.add_subplot(gs[1, 1])
    axb.set_title("(c) where the trainable parameters are (binary task)", fontsize=7.5)
    rows = [("projection learned", v)] + ([("projection frozen", fz)] if fz else [])
    parts = [("projection", "bottleneck", BLUE), ("head", "head", ORANGE),
             ("classifier", "classifier", GREY)]
    for i, (lab, r) in enumerate(rows):
        y0 = i * 2.1
        tot = float(r["total"])
        axb.text(0, y0 - 0.6, f"{lab}: {int(tot):,} in total", fontsize=6.8, color=INK, va="bottom")
        left, small = 0.0, []
        for name, key, col in parts:
            frac = r[key] / tot if tot else 0
            if frac <= 0:
                continue
            axb.barh(y0, frac, left=left, color=col, height=0.9, edgecolor="white", lw=1.2)
            if frac > 0.25:
                axb.text(left + frac / 2, y0, f"{name}\n{r[key]:,} ({100 * frac:.0f}%)",
                         ha="center", va="center", fontsize=6.0, color="white", linespacing=1.0)
            else:
                small.append(f"{name} {r[key]:,} ({100 * frac:.0f}%)")
            left += frac
        if small:
            axb.text(1.0, y0 + 0.55, " · ".join(small), ha="right", va="top", fontsize=6.0, color=INK2)
    axb.set_xlim(0, 1); axb.set_ylim(len(rows) * 2.1 - 1.1, -1.0)
    axb.axis("off")
    save(fig, out, "fig1_overview",
         {"d": d, "F": F, "L": L, "C_binary": C, "angle_scale": arch.angle,
          "learned": v, "frozen": fz, "matched_head": arch.ctl["head"], "projection_share": share},
         record)
    plt.close(fig)


# ============================================================ figure 2: the function class
def fourier_basis(Z, K=1):
    """[1, cos(w.z), sin(w.z)] over the canonical half of {-K..K}^d \\ {0}: (2K+1)^d columns."""
    import itertools
    d = Z.shape[1]
    om = [w for w in itertools.product(range(-K, K + 1), repeat=d)
          if any(w) and next(x for x in w if x) > 0]
    om = np.array(om, float)
    P = Z @ om.T
    return np.concatenate([np.ones((len(Z), 1)), np.cos(P), np.sin(P)], 1)


def fig_function_class_3d(plt, path, out, record):
    if not os.path.exists(path):
        print(f"  fig2_function_class_3d SKIPPED - {path} not found "
              f"(run src/16_reproduce_for_figure.py)")
        return
    z = np.load(path, allow_pickle=False)
    tz = z["test_z"]
    G1, G2 = z["grid_z1"], z["grid_z2"]
    if "pair_grids" in z.files:
        # The plane along which the real test latents spread most: largest
        # product of interquartile ranges. Fixed rule, stated in the caption.
        iqr = np.subtract(*np.percentile(tz, [75, 25], axis=0))
        k = int(np.argmax([iqr[i] * iqr[j] for i, j in z["pairs"]]))
        (ci, cj), Q = z["pairs"][k], z["pair_grids"][k][..., 0]
        fixed = [(m, float(z["median_z"][m])) for m in range(tz.shape[1]) if m not in (ci, cj)]
    else:
        ci, cj, Q = 0, 1, z["grid_out"][..., 0]
        fixed = [(2, float(z["fixed_z34"][0])), (3, float(z["fixed_z34"][1]))]
    ci, cj = int(ci), int(cj)
    d = z["fit_z"].shape[1]
    Phi = fourier_basis(z["fit_z"])
    coef, *_ = np.linalg.lstsq(Phi, z["fit_out"][:, 0], rcond=None)
    fit_resid = float(np.abs(Phi @ coef - z["fit_out"][:, 0]).max())
    Zg = np.zeros((G1.size, d))
    for m, val in fixed:
        Zg[:, m] = val
    Zg[:, ci], Zg[:, cj] = G1.ravel(), G2.ravel()
    Fsurf = (fourier_basis(Zg) @ coef).reshape(G1.shape)
    diff = np.abs(Q - Fsurf)
    lim = float(np.abs(Q).max())
    a = float(z["angle_scale"])
    n_terms = Phi.shape[1]
    sub = lambda i: "₁₂₃₄₅₆₇₈"[i]
    H, xe, ye = np.histogram2d(tz[:, ci], tz[:, cj], bins=20, range=[[-a, a], [-a, a]])
    Hp = np.pad(H, 1, mode="edge")                           # 3x3 box smoothing, display only
    H = sum(Hp[1 + di:Hp.shape[0] - 1 + di, 1 + dj:Hp.shape[1] - 1 + dj]
            for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    Xc, Yc = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, indexing="ij")

    fig = plt.figure(figsize=(COL2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.32)
    for k2, (S, title) in enumerate([(Q, "(a) trained VQC head, ⟨X₁⟩ (PennyLane)"),
                                     (Fsurf, f"(b) sum of {n_terms} sines and cosines")]):
        ax = fig.add_subplot(gs[0, k2], projection="3d")
        ax.plot_surface(G1, G2, S, cmap=DIV, vmin=-lim, vmax=lim, rstride=1, cstride=1,
                        linewidth=0.12, edgecolor=(1, 1, 1, 0.3), antialiased=True)
        if k2 == 0 and H.max() > 0:
            ax.contourf(Xc, Yc, H / H.max(), zdir="z", offset=-lim, levels=[0.02, 0.15, 0.4, 1.01],
                        colors=["#e1e0d9", "#c3c2b7", "#898781"], alpha=0.9)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel(f"z{sub(ci)}", labelpad=-9); ax.set_ylabel(f"z{sub(cj)}", labelpad=-9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.zaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(3))
        ax.tick_params(axis="z", pad=1, labelsize=6)
        if k2 == 1:
            ax.set_zticklabels([])                           # same scale as (a)
        ax.view_init(28, -55)
        for ax_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_.pane.set_facecolor((1, 1, 1, 0)); ax_.pane.set_edgecolor(GRID)
        ax.set_title(title, fontsize=7.3, pad=0)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(diff, origin="lower", extent=[-a, a, -a, a],
                   cmap="Greys", aspect="equal")
    ax.set_title("(c) |(a) − (b)|", fontsize=7.3)
    ax.set_xlabel(f"z{sub(ci)}"); ax.set_ylabel(f"z{sub(cj)}", labelpad=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.formatter.set_powerlimits((0, 0)); cb.update_ticks(); cb.ax.tick_params(labelsize=6)
    meta = {"cell": json.loads(str(z["cell"])), "auc_recorded": float(z["auc_recorded"]),
            "auc_reproduced": float(z["auc_reproduced"]), "max_prob_diff": float(z["max_prob_diff"]),
            "device": str(z["device"]), "n_basis": n_terms, "d": d,
            "plotted_coordinates": [ci + 1, cj + 1],
            "fixed_coordinates_at_test_median": {f"z{m + 1}": v for m, v in fixed},
            "test_latent_iqr": [float(x) for x in np.subtract(*np.percentile(tz, [75, 25], axis=0))],
            "share_of_test_latents_within_5pct_of_boundary":
                [float(np.mean(np.abs(tz[:, m]) > 0.95 * a)) for m in range(tz.shape[1])],
            "max_residual_fit_points": fit_resid, "max_residual_grid": float(diff.max())}
    print(f"     plane z{ci + 1}-z{cj + 1}; residual {fit_resid:.1e} (fit points), "
          f"{diff.max():.1e} (grid); AUC recorded {meta['auc_recorded']:.4f}, "
          f"reproduced {meta['auc_reproduced']:.4f}")
    print(f"     test latents near the tanh boundary, per coordinate: "
          f"{[round(x, 2) for x in meta['share_of_test_latents_within_5pct_of_boundary']]}")
    save(fig, out, "fig2_function_class_3d", meta, record)
    plt.close(fig)


# ============================================================ figure 3: where the advantage lives
def fig_where(plt, fam, exp, out, record):
    s = {"projection learned with the head": fam["primary"].series(),
         "projection frozen: PCA": exp["hs6_followup_pca"].series(),
         "projection frozen: random": exp["hs6_followup_random"].series()}
    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    for (lab, st), col, m in zip(s.items(), (BLUE, ORANGE, GREEN), ("o", "s", "^")):
        band(ax, st, col, m, lab)
    n_axis(ax)
    lo = min(r["ci_lo"] for st in s.values() for r in st)
    hi = max(r["ci_hi"] for st in s.values() for r in st)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    regions(ax, "VQC head ahead", "matched classical head ahead")
    points_axis(ax)
    ax.set_ylabel("Δ AUC (points), VQC − matched head")
    p5 = s["projection learned with the head"][0]
    ax.annotate(f"{points(p5['estimate']):+.1f} points at n = {REGIMES[0]}",
                xy=(REGIMES[0], p5["estimate"]), xytext=(REGIMES[0] * 1.9, p5["ci_hi"]),
                fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    legend_above(ax)
    save(fig, out, "fig3_where_the_advantage_lives", {k: rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)


# ============================================================ figure 4: restriction
def fig_restriction(plt, fam, exp, arch, out, record):
    def size(r):
        return f" ({r['head']} parameters)" if r else ""
    lab = (f"smaller{size(arch.r0)} − larger{size(arch.r8)} classical head"
           if arch and arch.r0 and arch.r8 else "smaller − larger classical head")
    follow, first = exp["hs5_followup"].series(), fam["hs5"].series()
    ref = fam["primary"].series()
    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    band(ax, first, GREY, "o", f"{lab}: first run", alpha=0.12)
    band(ax, follow, ORANGE, "s", f"{lab}: powered rerun")
    ax.plot(REGIMES, [r["estimate"] for r in ref], color=BLUE, ls="--", lw=1.0,
            marker="x", ms=4, label="VQC − matched head, for scale", zorder=2)
    n_axis(ax)
    lo = min(r["ci_lo"] for r in first + follow); hi = max(r["ci_hi"] for r in first + follow)
    pad = 0.1 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    regions(ax, "smaller head ahead", "larger head ahead")
    points_axis(ax)
    ax.set_ylabel("Δ AUC (points)")
    legend_above(ax, fontsize=6.5)
    save(fig, out, "fig4_restriction", {"first_run": rows_by_n(first), "powered_rerun": rows_by_n(follow),
                                        "reference_primary": rows_by_n(ref)}, record)
    plt.close(fig)


# ============================================================ figure 5: scope heatmap
def fig_scope(plt, fam, out, record):
    G = fam["primary"]
    names = G.datasets()
    rows = [(DATASET_LABEL.get(ds, ds), [G.cell(ds, n) for n in REGIMES]) for ds in names]
    rows.append(("all four, pooled", G.series()))
    M = np.array([[s["estimate"] for s in r] for _, r in rows])
    sig = np.array([[s["ci_lo"] > 0 or s["ci_hi"] < 0 for s in r] for _, r in rows])
    v = float(np.abs(M).max())
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    im = ax.imshow(M, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{points(M[i, j]):+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=7,
                    fontweight="bold" if sig[i, j] else "normal",
                    color="white" if abs(M[i, j]) > 0.66 * v else INK)
    ax.axhline(len(rows) - 1.5, color="white", lw=3)
    ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(REGIMES)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("labelled images per class")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Δ AUC (points), VQC − matched head", fontsize=7)
    points_axis(cb.ax)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, out, "fig5_scope_by_dataset",
         {"rows": {r[0]: rows_by_n(r[1]) for r in rows},
          "bold_means_interval_excludes_zero": True}, record)
    plt.close(fig)


# ============================================================ figure 6: function-class controls
def fig_function_class(plt, fam, exp, arch, out, record):
    d = arch.d if arch else None
    k1 = f" ({3 ** d} basis functions)" if d else ""
    k2 = f" ({5 ** d} basis functions)" if d else ""
    s = {f"single-encoding VQC − fit over its own class{k1}": fam["hs2"].series(),
         f"re-uploading VQC − fit over its own class{k2}": exp["reupload_vs_fourier_r2"].series()}
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for (lab, st), col, m in zip(s.items(), (ORANGE, GREEN), ("s", "^")):
        band(ax, st, col, m, lab)
    n_axis(ax)
    lo = min(r["ci_lo"] for st in s.values() for r in st)
    hi = max(r["ci_hi"] for st in s.values() for r in st)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    regions(ax, "VQC ahead", "classical fit ahead")
    points_axis(ax)
    ax.set_ylabel("Δ AUC (points)")
    legend_above(ax, fontsize=6.5)
    save(fig, out, "fig6_function_class", {k: rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)


# ============================================================ figure 7: noise
def fig_noise(plt, fam, exp, out, record):
    sig = np.array([float(s) for s in SIGMAS])
    vqc = [fam["hs3"].overall(i) for i in range(len(SIGMAS))]
    ctl = [exp["gap_control"].overall(i) for i in range(len(SIGMAS))]
    dif = [exp["gap_vqc_minus_control"].overall(i) for i in range(len(SIGMAS))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.35))

    def draw(ax, stats, col, m, lab, dx=0.0):
        y = np.array([s["estimate"] for s in stats])
        ax.errorbar(sig + dx, y, yerr=[y - [s["ci_lo"] for s in stats], [s["ci_hi"] for s in stats] - y],
                    color=col, marker=m, capsize=2, elinewidth=0.9, mec="white", mew=0.6,
                    label=lab, zorder=3)

    draw(a, vqc, BLUE, "o", "VQC head", -0.0025)
    draw(a, ctl, GREY, "s", "matched classical head", 0.0025)
    a.set_ylim(0, None)
    a.set_title("(a) F1 lost minus AUC lost, relative to clean images", pad=16)
    a.set_ylabel("relative F1 loss − relative AUC loss")
    legend_above(a, ncol=2)
    yb = np.array([s["estimate"] for s in dif])
    b.fill_between(sig, [s["ci_lo"] for s in dif], [s["ci_hi"] for s in dif], color=BLUE, alpha=0.14, lw=0)
    b.plot(sig, yb, color=BLUE, marker="o", mec="white", mew=0.6)
    b.axhline(0, color=AXIS, lw=0.8)
    b.set_ylim(min(0, min(s["ci_lo"] for s in dif)) - 0.005, None)
    b.set_title("(b) the VQC's excess over the matched head, paired", pad=16)
    b.set_ylabel("difference in the quantity of (a)")
    for ax in (a, b):
        ax.set_xticks(sig)
        ax.set_xlabel("sensor-noise σ, at native 28×28 resolution")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig7_noise", {"vqc": dict(zip(SIGMAS, vqc)), "matched_head": dict(zip(SIGMAS, ctl)),
                                  "paired_difference": dict(zip(SIGMAS, dif))}, record)
    plt.close(fig)


# ============================================================ supplementary: all 23 tests
PLAIN = {   # test id -> (group heading, row label). Group headings carry the question.
    "H-P1": ("VQC vs matched classical head", "n = 5"),
    **{f"H-S1 n={n}": ("Re-uploading vs single-encoding VQC", f"n = {n}") for n in REGIMES},
    **{f"H-S2 n={n}": ("VQC vs a direct fit of its own function class", f"n = {n}") for n in REGIMES},
    "H-S4": ("Adaptive encoder shrinks the head difference", "all cells"),
    "H-S5a": ("Smaller vs larger classical head", "n = 5"),
    "H-S6 pca": ("VQC vs matched head, projection frozen", "PCA, n = 5"),
    "H-S6 random": ("VQC vs matched head, projection frozen", "random, n = 5"),
    "H-S7a": ("Two-qubit readout (10 observables)", "vs padded readout"),
    "H-S7b": ("Two-qubit readout (10 observables)", "vs single-qubit readout"),
    "H-P2": ("Change per doubling of n", "VQC vs matched head"),
    "H-S5b": ("Change per doubling of n", "smaller vs larger classical head"),
    **{f"H-S3 s={s}": ("VQC under sensor noise", f"σ = {s}") for s in SIGMAS},
}


def fig_all_tests(plt, art, out, record):
    from matplotlib.ticker import MaxNLocator
    with open(os.path.join(art, "family_table.json")) as f:
        table = {r["test"]: r for r in json.load(f)}
    panels = [("Δ AUC (points)", [t for t in table if t not in ("H-P2", "H-S5b") and not t.startswith("H-S3")], True),
              ("slope per doubling of n (points)", ["H-P2", "H-S5b"], True),
              ("F1 loss − AUC loss, relative", [t for t in table if t.startswith("H-S3")], False)]
    heights = []
    layout = []
    for xlabel, ids, is_points in panels:
        rows, last = [], None
        for t in ids:
            g, lab = PLAIN[t]
            if g != last:
                rows.append(("h", g, None))
                last = g
            rows.append(("r", lab, table[t]))
        layout.append((xlabel, rows, is_points))
        heights.append(len(rows) + 0.8)
    H = 0.17 * sum(heights) + 1.6
    fig, axes = plt.subplots(3, 1, figsize=(COL1, H), gridspec_kw={"height_ratios": heights})
    seen = set()
    for ax, (xlabel, rows, is_points) in zip(axes, layout):
        ticks, labels = [], []
        for j, (kind, lab, r) in enumerate(rows):
            if kind == "h":
                ax.text(0.0, j, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=7, fontweight="bold", color=INK)
                continue
            col, m, filled, vlab = VERDICT_STYLE[r["verdict"]]
            ax.errorbar(r["estimate"], j, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                        color=col, marker=m, ms=4, capsize=1.5, elinewidth=0.8, ls="none",
                        mfc=col if filled else "white", mec=col,
                        label=None if vlab in seen else vlab, zorder=3)
            seen.add(vlab)
            ticks.append(j)
            labels.append(lab)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if is_points:
            ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
            points_axis(ax, "x")
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
    handles, labs = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labs:
                handles.append(h); labs.append(l)
    # Explicit margins: the group headings sit in the left margin, which
    # tight_layout cannot account for.
    fig.subplots_adjust(left=0.47, right=0.94, top=1 - 0.55 / H, bottom=0.45 / H, hspace=0.45)
    fig.legend(handles, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    save(fig, out, "figS1_all_confirmatory_tests",
         {t: dict(table[t], plain=PLAIN[t]) for t in table}, record)
    plt.close(fig)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=os.path.join("paper", "figures"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    art = args.artifacts

    fdir, edir = os.path.join(art, "family_table_cache"), os.path.join(art, "exploratory_cache")
    fam = {g: Group(os.path.join(fdir, f"{g}.npz"), BY_REGIME[g]) for g in BY_REGIME}
    exp = {os.path.splitext(f)[0]: Group(os.path.join(edir, f),
                                         EXPLORATORY_BY_REGIME.get(os.path.splitext(f)[0], True))
           for f in sorted(os.listdir(edir)) if f.endswith(".npz")}
    print(f"Loaded {len(fam)} confirmatory and {len(exp)} exploratory caches from {art}/")
    for name, G in list(fam.items()) + list(exp.items()):
        if G.B != B_DEFAULT:
            sys.exit(f"{name}: cache has B={G.B}, not {B_DEFAULT} - testing cache, not plottable")
    verify(art, fam, exp)
    if args.check_only:
        return

    cap_path = os.path.join(art, "reporting", "capacity.json")
    arch = None
    if os.path.exists(cap_path):
        with open(cap_path) as f:
            arch = Architecture(json.load(f))
        print(f"Architecture from run records: d={arch.d}, F={arch.F}, L={arch.L}, "
              f"projection {arch.vqc['bottleneck']}, head {arch.vqc['head']}, "
              f"classifier {arch.vqc['classifier']}")
    else:
        print(f"  {cap_path} not found - run src/15_reporting_tables.py first; "
              f"figure 1 is skipped and figures 4 and 6 omit parameter counts")

    os.makedirs(args.out, exist_ok=True)
    plt = setup_matplotlib()
    record = {}
    print(f"\nDrawing into {args.out}/")
    if arch:
        fig_overview(plt, arch, args.out, record)
    fig_function_class_3d(plt, os.path.join(art, "reproduced", "function_class.npz"), args.out, record)
    fig_where(plt, fam, exp, args.out, record)
    fig_restriction(plt, fam, exp, arch, args.out, record)
    fig_scope(plt, fam, args.out, record)
    fig_function_class(plt, fam, exp, arch, args.out, record)
    fig_noise(plt, fam, exp, args.out, record)
    fig_all_tests(plt, art, args.out, record)
    with open(os.path.join(args.out, "figure_data.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=float)
    print("  figure_data.json - every plotted value with its interval")


if __name__ == "__main__":
    main()
