"""
Figures for the checks of Amendments 28, 31 and 32 (all exploratory), and the
function-class figure redrawn to include them.

    python scripts/fig_amendments.py --art ART_FOLDER --out paper/figures

Reads only results already saved. Nothing is trained and no bootstrap is
recomputed:

    ART_FOLDER/squash_check.json          Amendment 28 (scripts/analyse_squash.py)
    ART_FOLDER/amendment31.json           Amendment 31 (scripts/analyse_amendment31.py contrasts)
    ART_FOLDER/amendment31_unclipped.json Amendment 31 without clipping (contrasts --unclipped)
    ART_FOLDER/amendment32.json           Amendment 32 (scripts/analyse_amendment32.py)
    paper/figures/figure_data.json        the values plotted in the published
                                          figures, written by
                                          src/eval/generate_paper_plots.py from the
                                          confirmatory bootstrap caches

Draws, with the helpers of src/eval/generate_paper_plots.py (imported, so fonts,
colours, units and axes match every other figure):

    fig8_input_bound
        VQC - matched head under three input designs: both bounded (the
        declared protocol), bound removed from the matched head only, both
        unbounded at their own tuned rates.

    fig6_function_class_all_inputs        (replaces fig6_function_class)
        VQC - sine-cosine model: the confirmatory H-S2 series (GPU, bounded,
        from figure_data.json), its CPU twin (bounded) and both unbounded.
        Why it replaces fig. 6: the old fig. 6 also drew the re-uploading
        contrast, which now has its own panel with all of its versions
        (fig. 4b). Keeping it in both would show one result twice.

    fig4_small_and_large_learners          (replaces fig4_restriction)
        (a) smaller - larger low-rank head: both at 1e-2, and the larger head
            at its selected rate.
        (b) re-uploading VQC - fit over its own 5^d class: at the inherited
            rate as recorded and without clipping, and the fit at its
            selected rate.

Why CPU series for figs 8 and 4: every contrast in them pairs arms trained on
the same device type (CPU, AMD EPYC 7B12), and the two inherited-rate contrasts
reproduce their GPU originals to within 0.0004 (restriction) and 0.0001
(re-upload), so each line differs from its partner only in the thing the
figure is about (the input, or the learning rate).

Before drawing, the script checks values against the outputs printed on
9-10 November and against the family table. A wrong, partial or missing file
stops it before anything is drawn.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
GP_PATH = os.path.join(REPO, "src", "eval", "generate_paper_plots.py")
FIGDATA = os.path.join(REPO, "paper", "figures", "figure_data.json")
FIG6_KEY = "single-encoding VQC − fit over its own class (81 basis functions)"

# (file, analysis, row, value printed in the Colab output, 4 decimals)
ANCHORS = [
    ("squash_check.json", "primary_cpu", "n=5", +0.0155),
    ("squash_check.json", "primary_nosquash", "n=5", +0.0002),
    ("amendment31.json", "nosquash_fair", "n=5", +0.0198),
    ("amendment31.json", "restriction_cpu_inherited", "n=5", +0.0097),
    ("amendment31.json", "restriction_tuned", "n=50", -0.0052),
    ("amendment31.json", "reupload_fc_tuned", "n=5", +0.0159),
    ("amendment31_unclipped.json", "reupload_fc_cpu_inherited", "n=5", +0.0278),
    ("amendment32.json", "fc_nosquash", "n=5", +0.0063),
    ("amendment32.json", "fc_squashed_cpu", "n=10", -0.0188),
]
# family table, H-S2 (confirmatory), n = 5 ... 100
HS2 = (-0.0095, -0.0187, -0.0212, -0.0203, -0.0161)


def load_gp():
    spec = importlib.util.spec_from_file_location("gp", GP_PATH)
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    return gp


def load_rows(path):
    with open(path) as f:
        rows = json.load(f)["rows"]
    return {(r["analysis"], r["row"]): r for r in rows}


def series(rows, analysis, regimes):
    out = []
    for n in regimes:
        r = rows.get((analysis, f"n={n}"))
        if r is None:
            sys.exit(f"{analysis} n={n} missing: incomplete analysis file - nothing drawn")
        out.append({"estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    return out


def check(files):
    bad = []
    for fname, a, row, want in ANCHORS:
        r = files[fname].get((a, row))
        if r is None or abs(round(r["estimate"], 4) - want) > 1e-9:
            got = None if r is None else round(r["estimate"], 4)
            bad.append(f"{fname} {a} {row}: expected {want:+.4f}, found {got}")
    return bad


def title(ax, text, n_legend_rows):
    """Panel title above the legend, as in fig. 7 of generate_paper_plots.py
    (title, then legend, then axes); the pad clears the legend rows."""
    ax.set_title(text, pad=6 + 10.5 * n_legend_rows)


def limits(ax, all_series, pad=0.12):
    lo = min(s["ci_lo"] for st in all_series for s in st)
    hi = max(s["ci_hi"] for st in all_series for s in st)
    p = pad * (hi - lo)
    ax.set_ylim(min(lo - p, -p), max(hi + p, p))


def draw(gp, ax, s, styles, top, bottom, ylabel):
    for (lab, st), (col, m, ls) in zip(s.items(), styles):
        gp.band(ax, st, col, m, lab, ls=ls, alpha=0.11)
    gp.n_axis(ax)
    limits(ax, s.values())
    gp.regions(ax, top, bottom)
    gp.points_axis(ax)
    ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True, help="folder holding the analysis JSON files")
    ap.add_argument("--out", default=os.path.join("paper", "figures"))
    args = ap.parse_args()

    gp = load_gp()
    R = gp.REGIMES

    # ---------------- load and check everything before drawing anything
    files, missing = {}, []
    for name in sorted({a[0] for a in ANCHORS}):
        p = os.path.join(args.art, name)
        if os.path.exists(p):
            files[name] = load_rows(p)
        else:
            missing.append(p)
    if not os.path.exists(FIGDATA):
        missing.append(FIGDATA)
    if missing:
        print("MISSING FILES - nothing drawn:")
        for p in missing:
            print("  " + p)
        sys.exit(1)
    bad = check(files)
    with open(FIGDATA, encoding="utf-8") as f:
        fd = json.load(f)
    try:
        hs2 = [fd["fig6_function_class"][FIG6_KEY][f"n={n}"] for n in R]
    except KeyError:
        hs2 = None
        bad.append(f"{FIGDATA}: no H-S2 series under fig6_function_class")
    if hs2 is not None:
        got = tuple(round(s["estimate"], 4) for s in hs2)
        if any(abs(g - w) > 1e-9 for g, w in zip(got, HS2)):
            bad.append(f"{FIGDATA}: H-S2 {got} does not match the family table {HS2}")
    if bad:
        print("ANCHOR CHECK FAILED - nothing drawn:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"All {len(ANCHORS) + 1} checks match the recorded outputs and the family table.")

    sq, a31 = files["squash_check.json"], files["amendment31.json"]
    a31u, a32 = files["amendment31_unclipped.json"], files["amendment32.json"]
    os.makedirs(args.out, exist_ok=True)
    plt = gp.setup_matplotlib()
    record = {}

    # ---------------- fig. 8: the input bound (single column)
    fig, ax = plt.subplots(figsize=(gp.COL1, 2.9))
    s = {"both bounded (declared protocol)": series(sq, "primary_cpu", R),
         "bound removed from the matched head only": series(sq, "primary_nosquash", R),
         "both unbounded, each at its tuned rate": series(a31, "nosquash_fair", R)}
    draw(gp, ax, s, [(gp.GREY, "o", "-"), (gp.ORANGE, "s", "-"), (gp.BLUE, "^", "-")],
         "VQC head ahead", "matched head ahead", "Δ AUC (points), VQC − matched head")
    gp.legend_above(ax, fontsize=6.5)
    gp.save(fig, args.out, "fig8_input_bound", {k: gp.rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)

    # ---------------- fig. 6: the circuit against its own function class (single column)
    fig, ax = plt.subplots(figsize=(gp.COL1, 2.9))
    s = {"bounded, GPU (confirmatory)": [dict(estimate=r["estimate"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"])
                                         for r in hs2],
         "bounded, CPU": series(a32, "fc_squashed_cpu", R),
         "both unbounded, each at its tuned rate, CPU": series(a32, "fc_nosquash", R)}
    draw(gp, ax, s, [(gp.ORANGE, "s", "-"), (gp.GREY, "o", "--"), (gp.BLUE, "^", "-")],
         "VQC head ahead", "sine–cosine model ahead", "Δ AUC (points), VQC − sine–cosine model")
    gp.legend_above(ax, fontsize=6.5)
    gp.save(fig, args.out, "fig6_function_class_all_inputs",
            {k: gp.rows_by_n(v) for k, v in s.items()}, record)
    plt.close(fig)

    # ---------------- fig. 4: small and large learners (double column)
    fig, (a, b) = plt.subplots(1, 2, figsize=(gp.COL2, 2.75))
    sa = {"both heads at 10⁻² (inherited)": series(a31, "restriction_cpu_inherited", R),
          "larger head at its selected rate, 3×10⁻³": series(a31, "restriction_tuned", R)}
    draw(gp, a, sa, [(gp.GREY, "o", "-"), (gp.ORANGE, "s", "-")],
         "smaller head ahead", "larger head ahead", "Δ AUC (points), 8 − 72 parameters")
    title(a, "(a) smaller − larger low-rank classical head", 3)
    gp.legend_above(a, fontsize=6.5)
    sb = {"fit at 10⁻² (inherited)": series(a31, "reupload_fc_cpu_inherited", R),
          "fit at 10⁻², without clipping": series(a31u, "reupload_fc_cpu_inherited", R),
          "fit at its selected rate, 3×10⁻³": series(a31, "reupload_fc_tuned", R)}
    draw(gp, b, sb, [(gp.GREY, "o", "-"), (gp.GREEN, "D", "--"), (gp.ORANGE, "s", "-")],
         "circuit ahead", "classical fit ahead", "Δ AUC (points), 24 − 2,500 parameters")
    title(b, "(b) re-uploading VQC − fit over its own function class", 3)
    gp.legend_above(b, fontsize=6.5)
    fig.tight_layout(w_pad=2.0)
    gp.save(fig, args.out, "fig4_small_and_large_learners",
            {"a": {k: gp.rows_by_n(v) for k, v in sa.items()},
             "b": {k: gp.rows_by_n(v) for k, v in sb.items()}}, record)
    plt.close(fig)

    path = os.path.join(args.out, "figure_data_amendments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=float)
    print(f"  {os.path.basename(path)} - every plotted value with its interval")
    print("DONE")


if __name__ == "__main__":
    main()
