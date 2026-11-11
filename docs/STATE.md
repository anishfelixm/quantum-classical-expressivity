# Project State

**Updated:** 11 November 2026. Keep this current. It is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first. Then read:
- `docs/analysis_plan.md`: binding; the amendments, with records at the end;
- `docs/RESULTS.md`: every number, by tier;
- `PAPER_PLAN_2026-11-11.md`, kept with the manuscript outside the repository:
  what the manuscript must say now.

This file says what is running and what comes next.

---

## Running now

**Nothing.** All experiments are finished, and they are frozen. Any further run
needs a new amendment, written and committed before it runs.

The last runs were Amendments 30, 31 and 32 (9–10 November, Colab CPU, AMD EPYC
7B12). Their outcomes are recorded in `analysis_plan.md`, "Outcomes of
Amendments 30, 31 (addition) and 32 — recorded 11 November 2026".

## Final counts

| What | Count |
|---|---|
| Reported runs reproduced bit for bit on the device type that recorded them | 35,740 (GPU 20,220; CPU 15,520) |
| Unclipped re-runs in Amendments 30–32, all step-level identical up to the first clipped step | 2,480 runs, 805,769 steps |
| Confirmatory tests | 23: 14 supported, 7 not, 1 opposite, 1 difference without prediction |
| Exploratory checks with predictions written before they ran | 19: 8 held, 8 failed, 3 held in part (`PAPER_PLAN`, `tab:checks`) |
| Amendments | 32 numbered |

## What the last three amendments changed

1. **The squash does not explain the advantage.** It cost the circuit +0.0130 to
   +0.0197 at every n. With both heads on raw z, each at its own tuned rate, the
   VQC's lead at n = 5 is +0.0198 [+0.0104, +0.0307] (exploratory). Amendment
   28's reading holds only when the squash is removed from the matched head
   alone.
2. **The function-class result holds on raw z.** The VQC never significantly
   beats `fourier_rff`. The squash did not detectably cost `fourier_rff`.
3. **Restriction at selected rates (exploratory).** The result is k = 0 − k = 8:
   +0.0131 at n = 5, −0.0052 at n = 50, slope −0.0039 [−0.0057, −0.0020]. That
   is the scarcity-dependent pattern, and it depends on the k = 8 rate. The
   confirmatory and follow-up verdicts are unchanged.
4. **Re-upload circuit against its 2,500-parameter fit.** In all six versions
   the circuit leads at n = 5 and 10.

## Next, in order

1. Commit these docs (no runs needed).
2. Manuscript: follow `PAPER_PLAN_2026-11-11.md`, Part 0, with
   `WRITING_RULES_2026-11-11.md`. Both are kept with the manuscript.
3. Fill in the first-run dates of the five new provenance rows from the run
   records:
   - `31_entangling_range`;
   - `32_squash`;
   - `34_lr_inherited`;
   - `35`–`37`;
   - `38`–`39`.
4. Zenodo: upload the run records, predictions, step norms and weights,
   including `31_entangling_range` to `39_fourier_cpu` and their `C_` and `CU_`
   folders, together with every `*.json` and `stats_*.txt` in `cpu_reruns/`.
   Put the DOI into the manuscript.
5. Back up `cpu_reruns/` with SHA-256, as for the 10 October backup.
6. Literature re-check in the week of submission. The 11 November check is
   recorded in `PAPER_PLAN` Part 1: Duong et al. 2026 is the closest work, and
   arXiv:2611.10544 is not yet read.

## Where everything is

| What | Where |
|---|---|
| Code and docs | GitHub, `feature/journal-expansion` |
| GPU run records (main study) | backups of 10 and 21 October (pod, local, Google Drive) |
| CPU run records, re-executions, unclipped re-runs, analysis outputs | Google Drive, `QML-backup-run/cpu_reruns/` |
| Intel session (not analysed) | `quarantine/CU_34_lr_inherited_intel_20261109` |
| Notebooks for the last runs | `scripts/a30_colab.ipynb`, `a31_colab.ipynb`, `a32_colab.ipynb` |
| Paper figures and tables | `paper/figures/`, `paper/tables/*.csv` |
| Manuscript, its plan and writing rules | outside the repository, by design |

## Operational rules that still apply

- If anything is ever re-run, use AMD EPYC 7B12 only. The notebooks stop at
  their device cell otherwise.
- Never `git pull` on a machine that is running a sweep.
- No analysis is changed after its result is seen.
