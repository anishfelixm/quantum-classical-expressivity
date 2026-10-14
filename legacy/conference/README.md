# Conference-version files (kept for provenance)

These files come from the conference version of this project, *Expressivity and
Robustness of Hybrid Quantum Neural Networks for Constrained Medical Image
Classification*. **They are not results of the journal study.** They were
produced under the conference protocol:
- 3 seeds;
- fractional data scarcity (10 % and 1 %);
- two datasets;
- fixed learning rates (5e-3 for the quantum head, 1e-3 for classical heads);
- 50 epochs, without early stopping;
- Linear and MLP baselines, with no parameter-matched or function-class control.

The scripts that produced them are no longer in the repository.

They are kept, unchanged, because the journal study re-tested the claims they
illustrated, and a reader should be able to see what was claimed and what
replaced it. The main [`README.md`](../../README.md) lists each withdrawn claim
and what the controls showed instead; every number is in
[`docs/RESULTS.md`](../../docs/RESULTS.md).

| File | What it is | Status |
|---|---|---|
| `figures/fig01_bottleneck_gap.png` | conference figure for the "Bottleneck Gap" | withdrawn: compressing to d = 4 costs almost nothing against d = 256 (RESULTS §3, premise check) |
| `figures/fig02_expressivity_dynamics.png` | conference figure ("expressivity dynamics") | superseded: the head's output is a classical trigonometric function, and the VQC loses to a direct fit over that function class at n ≥ 10 (H-S2) |
| `figures/fig03_precision_paradox.png` | conference figure for the "Precision Paradox" | withdrawn as a VQC-specific effect: the F1 collapse under noise happens to every small head (H-S3, RESULTS §3) |
| `logs/frozen_ablation_logs.json` | conference frozen-backbone training logs | conference protocol; not comparable |
| `logs/end_to_end_logs.json`, `logs/02_end_to_end_logs.json` | conference end-to-end fine-tuning logs | conference protocol; not comparable |
| `logs/robustness_e2e_logs.json` | conference noise-robustness logs | conference protocol; the noise was also clamped in normalised coordinates, which the journal study corrected |

The conference model checkpoints (`results/*.pt`) and downloaded data
(`data/*.npz`) were never tracked by git and are not part of the repository.
