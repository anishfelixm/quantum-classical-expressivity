# Project State

**Updated:** 21 November 2026. This is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first. Then read:
- `docs/analysis_plan.md`: binding; the amendments, with records at the end;
- `docs/RESULTS.md`: every number, by tier;
- `docs/CLAIMS_LEDGER.md`: the status of every manuscript claim;
- `paper/`: the manuscript as submitted.

---

## Status

**Submitted to IEEE Access on 21 November 2026. Under review.** The repository
state at submission is the tag `submitted-2026-11-21` (commit `ca5e12d`).

**Nothing is running.** All experiments are finished and frozen. Any further
run needs a new amendment, written and committed before it runs.

## Final counts

| What | Count |
|---|---|
| Reported runs reproduced bit for bit on the device type that recorded them | 35,740 (GPU 20,220; CPU 15,520) |
| GPU runs in which clipping acted, re-run without clipping (Amendment 21) | 1,998 |
| CPU runs in which clipping acted, re-run without clipping | 6 (full data, Amendment 27) + 2,480 (Amendments 30–32, all step-level identical up to the first clipped step; 805,769 steps) |
| Confirmatory tests | 23: 14 supported, 7 not, 1 opposite, 1 difference without prediction |
| Exploratory checks with predictions written before they ran | 19: 8 held, 8 failed, 3 held in part (paper, Table 17) |
| Amendments | 33 (1–32, plus 3a) |

## Next

1. **Zenodo deposit.** Run records, per-sample predictions, step norms,
   weights, bootstrap caches and the October backups. Then put the DOI into
   `paper/main.tex` (Data and Code Availability) and `README.md`. The SHA-256
   of every archive goes in the deposit.
2. **Reviews.** Answer them from the records. Any new run needs a new
   amendment first.
3. **Literature check** before the revision. arXiv:2611.10544 is not yet read.
4. **Processor of the original full-data runs** (GPU server's CPU, September).
   The paper says AMD EPYC 7B12. This is confirmed only for their Colab
   re-execution. Search the server logs and the October backups for the model
   name; if it differs, correct the sentence in the revision.

## Where everything is

| What | Where |
|---|---|
| Code, docs, logs, small analysis outputs | GitHub, `feature/journal-expansion` (the default branch) |
| Manuscript as submitted | `paper/` |
| Run logs | `logs/` (see `logs/README.md`) |
| Analysis outputs of Amendments 26–32 | `results/analysis/` |
| GPU run records (main study) | backups of 10 and 21 October (local and Google Drive); Zenodo (pending) |
| CPU run records, re-executions, runs without clipping | Google Drive, `QML-backup-run/cpu_reruns/` and `exp31/`; Zenodo (pending) |
| Intel session (not analysed) | Google Drive, `quarantine/CU_34_lr_inherited_intel_20261109` |
| Notebooks for the Colab runs | `scripts/entangling_colab.ipynb`, `week_colab.ipynb`, `a30_colab.ipynb`, `a31_colab.ipynb`, `a32_colab.ipynb`, `cpu17_colab.ipynb`; the copies with saved outputs go to Zenodo |

## Repository changes after the experiments

- **21 November 2026, clean-up** (no code path changed):
  - the earlier, unsubmitted draft's files were removed; they remain in git
    history (for example at the tag `submitted-2026-11-21`);
  - the old manuscript source was replaced by the submitted source;
  - the logs were moved into `logs/`, and the Colab logs and analysis outputs
    were added from Google Drive;
  - three one-time text-edit helpers and two working notes were removed
    (`docs/PAPER_OUTLINE.md`, `docs/WORK_REMAINING.md`).

## Operational rules that still apply

- If anything is ever re-run on CPU, use AMD EPYC 7B12 only. The notebooks
  stop at their device cell otherwise.
- Never `git pull` on a machine that is running a sweep.
- No analysis is changed after its result is seen.
- Never rewrite git history. Run records store the commit they ran on, and
  commit dates show that each amendment was written before its runs.
