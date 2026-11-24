# Project State

**Updated:** 24 November 2026. This is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first. Then read:
- `docs/analysis_plan.md`: binding; the amendments, with records at the end;
- `docs/RESULTS.md`: every number, by tier;
- `docs/CLAIMS_LEDGER.md`: the status of every manuscript claim;
- `paper/`: the manuscript as submitted.

---

## Status

**Submitted to IEEE Access on 21 November 2026. Under review.** The repository
state at submission is the tag `submitted-2026-11-21` (commit `ca5e12d`).

**Data deposit published on 24 November 2026:** Zenodo,
DOI [10.5281/zenodo.24465986](https://doi.org/10.5281/zenodo.24465986)
(all versions: 10.5281/zenodo.24465985). It was made against commit `e0aed52`
(tag `zenodo-data-v1.0`).

**Nothing is running.** All experiments are finished and frozen. Any further
run needs a new amendment, written and committed before it runs.

## Final counts

| What | Count |
|---|---|
| Reported runs reproduced bit for bit on the device type that recorded them | 35,740 (GPU 20,220; CPU 15,520) |
| GPU runs in which clipping acted, re-run without clipping (Amendment 21) | 1,998 (plus 524 re-run though not required; see the Record of 24 November 2026 in the plan) |
| CPU runs in which clipping acted, re-run without clipping | 6 (full data, Amendment 27) + 2,480 (Amendments 30–32, all step-level identical up to the first clipped step; 805,769 steps) |
| Confirmatory tests | 23: 14 supported, 7 not, 1 opposite, 1 difference without prediction |
| Exploratory checks with predictions written before they ran | 19: 8 held, 8 failed, 3 held in part (paper, Table 17) |
| Amendments | 33 (1–32, plus 3a) |

## Next

1. **Reviews.** Answer them from the records. Any new run needs a new
   amendment first.
2. **Revision of the manuscript** (with the answers to the reviews):
   - put the DOI into Data and Code Availability;
   - cite arXiv:2611.10544 (Bhalgami *et al.*), read on 22 November 2026:
     related, not pre-empting;
   - report that Amendment 21's prediction P2 failed (59 circuit runs exceeded
     the clipping norm; no confirmatory verdict changed);
   - correct the processor sentence: the original full-data runs ran on the GPU
     server's CPU, whose model is not recorded; their Colab re-execution ran on
     AMD EPYC 7B12;
   - widen the statement on AI assistance.
3. **After acceptance:** add the article DOI to `CITATION.cff`, `README.md` and
   the Zenodo record ("is supplement to"), and tag the accepted version.

## Where everything is

| What | Where |
|---|---|
| Code, docs, logs, small analysis outputs | GitHub, `feature/journal-expansion` (the default branch) |
| Manuscript as submitted | `paper/` |
| Run logs | `logs/` (see `logs/README.md`); GPU job logs in the Zenodo deposit |
| Analysis outputs of Amendments 26–32 | `results/analysis/` |
| Run records, predictions, step norms, weights, caches (GPU and CPU) | Zenodo, 10.5281/zenodo.24465986; also the backups of 10 and 21 October and Google Drive `QML-backup-run/` |
| Intel session (not analysed) | Zenodo, `notcounted_quarantined_intel_session.zip` |
| Notebooks for the Colab runs | `scripts/entangling_colab.ipynb`, `week_colab.ipynb`, `a30_colab.ipynb`, `a31_colab.ipynb`, `a32_colab.ipynb`, `cpu17_colab.ipynb`; the copies with saved outputs are in the Zenodo deposit (`support_logs.zip`, `notebooks_as_run/`) |

## Repository changes after the experiments

- **21 November 2026, clean-up** (no code path changed):
  - the earlier draft's files were removed; they remain in git history;
  - the old manuscript source was replaced by the submitted source;
  - the logs were moved into `logs/`, and the Colab logs and analysis outputs
    were added from Google Drive;
  - three one-time text-edit helpers and two working notes were removed.
- **24 November 2026, documentation only** (no code changed):
  - the DOI added to `README.md`, `CITATION.cff`, `paper/README.md`,
    `logs/README.md` and the claims ledger;
  - `README.md` rewritten as the front page of the paper;
  - `data/README.md` removed: it described the earlier draft, and the code
    caches MedMNIST in `artifacts/data_cache/`, not in `data/`;
  - a dated record added to the analysis plan (deposit, and a correction about
    the 524 extra unclipped runs).

## Operational rules that still apply

- If anything is ever re-run on CPU, use AMD EPYC 7B12 only. The notebooks
  stop at their device cell otherwise.
- Never `git pull` on a machine that is running a sweep.
- No analysis is changed after its result is seen.
- Never rewrite git history. Run records store the commit they ran on, and
  commit dates show that each amendment was written before its runs.
