# Project State

**Updated:** 16 October 2026. Keep this current — it is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first, then `docs/RESULTS.md` (every
number), `docs/analysis_plan.md` (binding; amendments; records at the end) and
`docs/CLAIMS_LEDGER.md` (every manuscript claim and its source). This file says
what is running and what is next.

---

## Running now

Amendments 20, 21 and 23 on the pod, once launched (`scripts/launch_v57.sh`).
All three were written before any run of them. Launch happens in three steps:

1. **Gate.** `bash scripts/launch_v57.sh gate J` re-executes one cheap run per
   configuration of every source and commit. Every one must reproduce its
   record bit for bit.
2. **Wave 1**, in the background, in priority order:
   1. the primary namespace re-executed at the original protocol;
   2. the 1,000-epoch budget check;
   3. the follow-up and other confirmatory namespaces;
   4. the remaining reported namespaces;
   5. hardware feasibility at the tuned rate;
   6. full data.
3. **Wave 2.** Unclipped re-executions of every run in which a step exceeded
   20.

Every re-executed run keeps the norm of every optimisation step and its
trained weights (`artifacts/step_norms/`, `artifacts/weights/`).

Progress: `tail -f logs_v57/status.txt`. Projection: `logs_v57/plan_wave1_*.txt`.
**Do not `git pull` on the pod while a wave is running.**

## Gate of 16 October (before wave 1)

20 of 22 checks passed. The two failures are explained in `analysis_plan.md`,
"Records of 16 October": an error in the replay adapter for `03` (fixed), and
runs that built the feature cache (699 of the logged runs; they need the cache
absent to be reproduced). No reported number changes. The replay now handles
the cache state, and the gate gains one cache-building run per source and
commit. Re-run the gate; launch wave 1 only if every check passes.

## Done on 15 October

- **Run-record verification** (`check_paper_facts`). It confirmed the
  early-stopping rule and the learning rates. It found that clipping binds on
  Fourier heads and that most runs, and 71 of 80 full-data runs, reach the
  100-epoch cap.
- **Amendment 18** (frozen compression): both predictions failed.
- **Amendment 19** (n = 5 budget): both predictions held.
- **Line-by-line audit of the manuscript** (`docs/CLAIMS_LEDGER.md`). Wrong
  values corrected:
  - reordered pairs 7, not 24;
  - PennyLane residuals;
  - NumPy residuals;
  - adjoint agreement;
  - PathMNIST full-data size 10,002;
  - early-stopping floor values;
  - shot retention being a ratio of cell means.
- **Clip threshold basis.** It was set before the run-record system existed
  (git `7be3dcf`, 10 Aug 19:30), so it was never recorded. It is replaced by
  measured step norms.
- **Pod**: 66 tests passed; backprop vs adjoint agree to a relative 4e-7; no
  `SEED-LEVEL FALLBACK`; reporting tables re-run with identical console values
  (CSV line endings being checked).

## To do on the pod before and alongside the launch

- `python scripts/verify_claims.py`, sections A–E: environment, data counts,
  training records, the search for the threshold's source, single numbers.
- `python scripts/verify_claims.py --sections F`: the analysis code
  recomputed on the real data must reproduce the published family table
  exactly.
- The CSV line-ending check.

## Where everything is

| What | Where |
|---|---|
| Code and docs | GitHub, `feature/journal-expansion` |
| Old-commit code for re-execution | `../replay_trees/<commit>` (git worktrees, created by the launcher) |
| Paper figures and tables | `paper/figures/`, `paper/tables/*.csv` |
| Manuscript draft (model text) | outside the repository, by design |
| Backups | `backup_2026-10-10/` on the pod, `C:\Users\Anish\quantum-classical-expressivity-remote-backup`, Google Drive; the two 15 Oct archives (`5d89ac77…`, `d9acd367…`) |

## Next, in order

1. Gate → wave 1 → wave 2.
2. `python scripts/analyse_replay.py identity`, then
   `python scripts/analyse_replay.py results`.
3. `python scripts/analyse_budget.py`.
4. Write the Amendment 22 (trained-head sensitivity) and Amendment 23(b)
   (per-layer noise) analyses on the saved weights, exactly as specified.
5. Record every result in `analysis_plan.md`, `RESULTS.md` and this file, as
   the amendments say.
6. Back up shards, predictions, step norms, weights and logs, with SHA-256.
7. Manuscript v4 from the ledger.
8. Release items, citation check, cover letter.
