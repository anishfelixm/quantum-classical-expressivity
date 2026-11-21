# Run and analysis logs

Console output of the study's runs and analyses, kept for provenance. Logs
are evidence of what ran, when, and on which commit. The reported numbers come
from the run records and per-sample predictions (Zenodo), recomputed by the
analysis scripts, not from these logs.

## Layout

| Folder | What it holds | Where it ran |
|---|---|---|
| `logs/` (top level) | every log committed during the study, under the file names that `docs/` and the analysis plan use (`logs_lr.txt`, `logs_confirm.txt`, `logs_stats_*.txt`, ...) | mostly the GPU server (A100 MIG 3g.20gb), August–October 2026; a few Colab analyses |
| `logs/colab_cpu_reruns/` | Amendments 27–32: CPU runs, CPU re-executions (`C_`), CPU runs without clipping (`CU_`), rate tuning and analyses | Google Colab CPU, November 2026 |
| `logs/colab_exp31/` | Amendment 26: the entangling-range runs (`31_entangling_range`), their pilot and audit | Google Colab CPU, 31 October – 1 November 2026 |

Until 21 November 2026 the top-level logs sat in the repository root; the two
Colab folders were on Google Drive only. They were moved here unchanged. Older
documents name the top-level logs without a folder; they are in `logs/`.

Two finished one-off scripts, `scripts/part2_colab.py` and
`scripts/predict_cold_runs.py`, read logs from the repository root, where they
were when the scripts ran. They are left unchanged. To re-run them, copy the
logs they name back to the root.

## Reading the top-level logs

- **Run logs** start with the run count, the device and the commit, for
  example `Experiment 1 | 1200 cells | device=cuda | sha=4771f9a4`.
- **Analysis logs** (`logs_stats_*.txt`, `logs_family.txt`,
  `logs_exploratory*.txt`, `logs_followup_analysis.txt`) print the commit, B
  and the tests computed.
- **Launch logs** (`logs_gap_closure.txt`, `logs_gap_status.txt`,
  `logs_followup.txt`) record when each step of a launch script started and
  ended.
- Some logs come from diagnostic or superseded runs and analyses, kept so that
  nothing is hidden:
  - `logs_diag.txt` and `logs_matched.txt` are the pre-plan diagnostic sweeps;
  - a log that prints `SEED-LEVEL FALLBACK IN USE` (for example
    `logs_stats_01.txt`) predates per-sample predictions and is not a reported
    analysis.

  `docs/RESULTS.md` and `docs/analysis_plan.md` say which runs are reported.
- `logs_stats_replay_identity.txt` and `logs_stats_replay_results.txt` are the
  Amendment 21 checks: bit-for-bit identity of the GPU re-executions, and the
  results recomputed without clipping.
- `logs_stats_budget.txt`, `logs_stats_trained_heads.txt`,
  `logs_stats_hardware_tuned.txt` and `logs_barren_plateau.txt` are the
  Amendment 20, 22, 23 and 24 analyses (22 October 2026).

## Reading the Colab logs

Each Colab session had a role, set in cell 1 of its notebook.

| Roles | Amendment | Notebook |
|---|---|---|
| `A0`, `A1`, `C`, `D1`, `D2`, `E1`, `E2` | 27–29 | `scripts/week_colab.ipynb` |
| `F1`, `F2`, `F3` | 30 | `scripts/a30_colab.ipynb` |
| `G1`–`G3`, `H1`–`H3`, `J1`–`J3` | 31 | `scripts/a31_colab.ipynb` |
| `K1`, `K2`, `L`, `M` | 32 | `scripts/a32_colab.ipynb` |
| sessions of Amendment 26 | 26 | `scripts/entangling_colab.ipynb` |

**What each file holds:**
- `sessions.txt`: one line per session, giving the start time (UTC), role,
  processor, cores and commit. A session is logged before its processor check,
  so a session refused for running on the wrong processor leaves one line and
  no runs.
- `gates_<role>.txt`: the checks each session ran before any training.
- `environment*.txt`: the installed package versions.
- `replay<NN>_*.txt`: CPU re-executions (`C_`) of namespace `NN`.
- `replay<NN>u_*.txt` and `cu<NN>_*.txt`: re-runs without clipping (`CU_`).
- `a3x_views.txt` and `a3x_check_steps.txt`: the views without clipping, and
  the step-level prefix checks.
- `tune*.txt`: learning-rate tuning runs.
- `squash_*`, `nosq36_*`, `gaps37_*`, `fc39_*`: the training runs of
  namespaces 32, 36, 37 and 39.
- `stats_*.txt`: analysis outputs. The matching JSON files are in
  `results/analysis/`.

**Quarantined session.** One session (role `F1`, 9 November 2026, 06:39 UTC)
ran on an Intel Xeon processor before the processor check existed. Its runs
diverged from the first step and were quarantined. They are not analysed and
are not in this repository.
