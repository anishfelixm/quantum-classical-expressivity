"""
Redraw only Figures 1 and 2, without the bootstrap caches the full plot script needs.

WHY THIS EXISTS
---------------
src/eval/generate_paper_plots.py draws all eight figures, and its main() first
loads and checks the family and exploratory bootstrap caches. Figures 1 and 2 do
not use those caches: Figure 1 reads only the parameter counts in
artifacts/reporting/capacity.json, and Figure 2 reads only the retrained head in
artifacts/reproduced/function_class.npz. This script calls the same two drawing
functions, unchanged, so a label fix in either figure can be rebuilt on any
machine that has those two files.

The font is pinned to DejaVu Sans, the font the published figures were drawn
with, so the redrawn figures match the others in the paper on any machine.

    python scripts/redraw_fig1_fig2.py --artifacts artifacts --out paper/figures
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "..", "src", "eval", "generate_paper_plots.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default=os.path.join("paper", "figures"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("generate_paper_plots", PLOTS)
    gpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpp)

    plt = gpp.setup_matplotlib()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    os.makedirs(args.out, exist_ok=True)
    record = {}

    cap = os.path.join(args.artifacts, "reporting", "capacity.json")
    with open(cap) as f:
        arch = gpp.Architecture(json.load(f))
    gpp.fig_overview(plt, arch, args.out, record)

    gpp.fig_function_class_3d(plt, os.path.join(args.artifacts, "reproduced", "function_class.npz"),
                              args.out, record)
    print("done; figure_data.json is not rewritten (no plotted value changes)")


if __name__ == "__main__":
    main()
