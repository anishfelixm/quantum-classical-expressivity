# Project State

**Updated:** 11 October 2026. Keep this current — it is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first, then `docs/RESULTS.md` (every
number) and `docs/analysis_plan.md` (binding; amendment log; follow-up results
at the end). This file says what is running and what is next.

---

## Running now

**Nothing.** All experiments finished on 6 October. The GPU cluster has been
returned.

## Where everything is

| What | Where |
|---|---|
| Code and docs | GitHub, `feature/journal-expansion`; last run commit `17506e8` |
| Backup (results, feature cache, checkpoints, data cache, latents, repo snapshot, conda env, pip freeze, git state) | `C:\Users\Anish\quantum-classical-expressivity-remote-backup`, dated 2026-10-10; SHA-256 verified; second copy on Google Drive |
| Bootstrap caches for figures | inside `results_2026-10-10.tgz`: `artifacts/family_table_cache`, `artifacts/exploratory_cache`, `family_table.json`, `exploratory_table.json` |

Restarting any frozen-backbone run elsewhere needs `feature_cache_2026-10-10.tgz`
plus the `qml_v2` environment (`conda_env_qml_v2.yml`).

## Open checks

1. **Is `C:\Users\Anish` itself a git repository?** The backup folder showed
   `(master)` in Git Bash. Run `git rev-parse --show-toplevel` inside it. If it
   prints `C:/Users/Anish`, a stray `git init` was run in the home folder —
   harmless to the project, but nothing should be committed there.
2. **`git_diff.txt` in the backup** (640 bytes): confirm it is only a tracked log
   file and not unpushed code.

## Next, in order

1. Commit the corrected documents (this file, `RESULTS.md`, `PAPER_OUTLINE.md`
   v3.1, `AUDIT_REPORT.md`, `analysis_plan.md`, `WORK_REMAINING.md`,
   `MASTER_RESEARCH_DOCUMENT.md`).
2. Extract `results_2026-10-10.tgz` into the local repo's `artifacts/` (check
   `git check-ignore artifacts` first so 3.9 GB is never staged).
3. Write `src/eval/generate_paper_plots.py` — torch-free, reads the caches only.
4. Rewrite the conference `paper/main.tex` into the journal manuscript.
5. Citation check, cover letter, submit.

No further experiments are planned. A new idea goes in the paper's future-work
paragraph, not a sweep.
