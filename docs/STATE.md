# Project State

**Updated:** 14 August 2026. Keep this current — it is the handoff document.
Read this first in any new session, then `docs/HYPOTHESES.md` and
`docs/MATH_VERIFICATION.md` for detail.

---

## The question

At a fixed small parameter budget and with very few labelled images, does a
variational quantum head extract more from a compressed feature vector than a
classical head — and if it does, is the advantage quantum, or merely
trigonometric?

## The thesis (as currently supported)

The quantum head's advantage, where it exists, is a **regularization effect from
its restricted function class** — a 24-dimensional manifold inside an
81-dimensional trigonometric span — not a quantum computational advantage. It is
confined to the extreme-scarcity regime and reverses once data is sufficient.

---

## Answered

| Q | Question | Answer | Runs |
|---|---|---|---|
| Q0 | Does compression to d=4 cost anything? | No — 0.002 AUC when the encoder adapts | 144 |
| Q1 | Efficiency at 24 vs 24 parameters? | Tie overall (31/40 cells) | 400 |
| Q2 | Can the VQC exploit its own function class? | No — a direct fit over the same basis wins on multi-class | 1,200 |
| Q3 | Does the encoder absorb the bottleneck? | Yes, strongly; shrinks head differences 3–5× | 1,200 |

## Open

| Q | Question | Status |
|---|---|---|
| Q4 | Is narrow frequency support the cause? | **running**, ~Mon 17 Aug |
| Q5 | Differential noise robustness | not started |
| — | Confirmatory sweep, 40 seeds | blocked on `analysis_plan.md` commit |

## The finding to confirm

Frozen encoder, Δ = AUC(quantum) − AUC(matched classical), averaged over 4 datasets:

| shots/class | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ | +0.039 | +0.023 | −0.025 | −0.023 | −0.020 |
| datasets favouring quantum | 4/4 | 3/4 | 1/4 | 0/4 | 0/4 |

Monotone crossover between n=10 and n=20. **Post hoc** — pre-registered in
`docs/analysis_plan.md`, to be confirmed at 40 seeds.

---

## Verified facts (do not re-derive)

- VQC output lies exactly in the 3^d trigonometric span; residual 1e-16
  (`tests/test_fourier_equivalence.py`). Matches Schuld/Sweke/Meyer 2021.
- `freeze_policy`: `"all"` = frozen, `"layer3_only"` = adaptive. There is no `"frozen"`.
- Gradients propagate correctly with `diff_method="backprop"`; adjoint agrees to 3e-7.
- Head parameter counts at d=4: `quantum_vqc` 24, `quantum_reupload` 24,
  `matched_param` 24, `matched_param_fullrank` 24, `fourier_rff` 324.
- `fourier_rff` = 80 independent features after the canonical-frequency fix.
- Gradient clipping at 20.0 never binds (largest observed p95 = 9.62).
- Runtime: ~1 min/run classical, ~2.5 min quantum, ~6.5 min re-uploading.

## Known defects, unfixed

- `MatchedParamHead` is rank-limited (width 3 regardless of d). Retained for
  diagnostic reproducibility; `MatchedParamFullRankHead` supersedes it.
- `fourier_rff` results before the canonical-frequency fix used a 68-dimensional
  basis and must be regenerated.
- ETA display in `01_frozen_backbone_ablation.py` underestimates badly (it averages
  in instantly-returning cached cells).
- `src/03_robustness_evaluation.py`, `src/04_statistical_analysis.py`,
  `src/eval/generate_paper_plots.py` are conference-era and will crash on import.

## To delete

`src/02_end_to_end_finetuning.py` (redundant — `01` with `layer3_only` is the
end-to-end experiment) · `src/models/classical_resnet.py` · `tests/test.py` ·
`tests/test2.py` · `results/*.json` → move to `artifacts/` as archive

---

## Environment

```
python 3.10 · torch 2.4.1+cu118 · pennylane 0.42.3 · numpy 1.26.4
conda env: qml_v2       (fallback: qml_journal)
GPU: A100-SXM4-40GB MIG 3g.20gb, driver 470 → CUDA capped at 11.4
simulator: default.qubit + backprop, all dimensions
```

Edit and commit on Windows; pull and run on Linux.
Artifacts live outside the repo via the `artifacts/` symlink.

## Timeline

| | |
|---|---|
| Q4 completes | ~17 Aug |
| Confirmatory + Q5 | ~24 Aug |
| Analysis, figures | ~27 Aug |
| Draft | ~6 Sep |
| Review, submit | ~10 Sep |

Target venue: IEEE Access. APC covered by the university.
