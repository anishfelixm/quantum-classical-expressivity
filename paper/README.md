# Manuscript

*Where Does a Variational Quantum Head's Advantage Come From? A Controlled Study
of Medical Image Classification Under Data Scarcity.*
Anish Felix Mathias, Susham P S, Rajashree Shettar and Badarinath Kamasetty.
Submitted to IEEE Access on 21 November 2026; under review.

This folder holds the LaTeX source **as submitted**. The repository state at
submission is the tag `submitted-2026-11-21`.

## Building the PDF

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

`ieeeaccess.cls`, `IEEEtran.cls` and the logo images are the IEEE Access
template files. The references are inside `main.tex` (no `.bib` file).

The four author photographs used in the biographies are not published here.
To compile, either place any images named `photo_anish.jpg`, `photo_susham.jpg`,
`photo_shettar.jpg` and `photo_kamasetty.jpg` in this folder, or delete the
`[{\includegraphics...}]` option from the four `IEEEbiography` blocks.

The Data and Code Availability section reads `[Zenodo DOI]`, as submitted. The
data deposit was published on 24 November 2026 as
[10.5281/zenodo.24465986](https://doi.org/10.5281/zenodo.24465986). The DOI
goes into the revised manuscript; the source here stays as submitted.

## Figures

The file names come from the scripts that draw them, so they do not follow the
paper's numbering. Every figure is drawn from saved results; none is drawn by
hand.

| Paper | File in `figures/` | Drawn by |
|---|---|---|
| Fig. 1 | `fig1_overview` | `src/eval/generate_paper_plots.py` (or `scripts/redraw_fig1_fig2.py`) |
| Fig. 2 | `fig2_function_class_3d` | `src/eval/generate_paper_plots.py` (or `scripts/redraw_fig1_fig2.py`) |
| Fig. 3 | `fig5_scope_by_dataset` | `src/eval/generate_paper_plots.py` |
| Fig. 4 | `fig6_function_class_all_inputs` | `scripts/fig_amendments.py` |
| Fig. 5 | `fig3_where_the_advantage_lives` | `src/eval/generate_paper_plots.py` |
| Fig. 6 | `fig8_input_bound` | `scripts/fig_amendments.py` |
| Fig. 7 | `fig4_small_and_large_learners` | `scripts/fig_amendments.py` |
| Fig. 8 | `fig7_noise` | `src/eval/generate_paper_plots.py` |
| Fig. 9 | `figS1_all_confirmatory_tests` | `src/eval/generate_paper_plots.py` |

`figure_data.json` and `figure_data_amendments.json` hold every plotted value
with its interval.

To redraw:

```bash
python src/eval/generate_paper_plots.py --out paper/figures          # needs the bootstrap caches in artifacts/
python scripts/fig_amendments.py --art results/analysis --out paper/figures
```

`generate_paper_plots.py` also writes `fig4_restriction` and
`fig6_function_class`. The paper does not use them: `fig_amendments.py`
replaced them with Figs. 7 and 4, which add the later checks. The script was
left unchanged so that it stays the code that drew the published figures.

## Tables

`tables/*.csv` are written by `src/15_reporting_tables.py` (13 October 2026).
`tables/provenance.csv` covers the twelve namespaces of the confirmatory and
follow-up runs only. The paper's provenance table (Table 16) also lists the
learning-rate tuning, the exploratory GPU runs and the CPU checks, read from
the run records on 11 November 2026 (`docs/analysis_plan.md`, end of the file).
