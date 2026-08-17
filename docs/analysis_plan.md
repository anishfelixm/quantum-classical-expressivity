# Pre-Registration / Analysis Plan

**Written:** 14 August 2026
**Status:** must be committed BEFORE the confirmatory sweep produces any result.
**Binding on:** every number that appears in the manuscript.

---

## 0. Why this document exists

The exploratory diagnostic (1,200 + 400 runs, 10 seeds) has already been
inspected. One of the findings below — the scarcity crossover — was noticed *by
looking at the data*, not predicted in advance. It therefore cannot be claimed
on that data.

This document fixes, before any confirmatory number exists: the hypotheses, the
arms, the statistics, the correction, the decision rule, and what would falsify
each claim. Everything not listed here is exploratory and will be labelled as
such in the manuscript.

---

## 1. What is already known (exploratory — none of this is claimable yet)

| Finding | Evidence |
|---|---|
| Compression 256→4 costs ≈0.002 AUC when the encoder adapts | 144 cells |
| Encoder adaptation is large and explains that null | 1,200 cells |
| At 24 vs 24 parameters the VQC ties overall (31/40 cells) | 400 runs |
| Against a 324-parameter Fourier head over the same basis, the VQC loses on multi-class | 1,200 cells |
| **Scarcity crossover**: frozen-encoder Δ(VQC − matched) is +0.039, +0.023, −0.025, −0.023, −0.020 at n = 5, 10, 20, 50, 100 | 400 runs, **post hoc** |
| VQC Macro-F1 degrades far more than its AUC | 1,200 cells |

The crossover has a mechanism: the VQC reaches a 24-dimensional manifold inside
an 81-dimensional trigonometric span (proved analytically, verified to 1e-16).
A restricted function class regularises when data is scarce and limits when it
is not. That mechanism predicts the observed direction — but it was formulated
after seeing the pattern, so it requires independent confirmation.

---

## 2. Primary hypothesis

> **H-P.** With a frozen encoder at d=4, the quantum head's advantage over a
> parameter-matched classical head decreases monotonically with the number of
> training shots per class, being positive at extreme scarcity and negative once
> data is sufficient.

Formally, with Δ(n) = AUC(quantum_vqc) − AUC(matched_param_fullrank):

- **H-P1:** Δ(5) > 0
- **H-P2:** the slope of Δ on log₂(n) is negative

Both must hold for H-P to be supported.

### Design

| Axis | Value |
|---|---|
| Arms | `quantum_vqc`, `matched_param_fullrank` |
| Parameters | 24 each, both full-rank at d=4 |
| Encoder | frozen (`freeze_policy="all"`) |
| Bottleneck | d = 4 |
| Datasets | all four |
| Shots/class | 5, 10, 20, 50, 100 |
| Seeds | **40** (`ALL_SEEDS` extended; list fixed in `config.py` before launch) |
| Augmentation | off (required for feature caching; identical for both arms) |
| Runs | 4 × 5 × 2 × 40 = 1,600 |

Seeds are fixed in advance. **No interim analysis. No stopping early. No adding
seeds after seeing results.**

### Primary statistic

Nested paired bootstrap, B = 2000, resampling **both** test indices and seeds:

```
for b in 1..B:
    I_b = resample test indices with replacement
    S_b = resample seeds with replacement
    Δ_b(n) = mean_{s in S_b} [ AUC_q(s, I_b) − AUC_c(s, I_b) ]
```

Pairing is on seed: both arms see identical splits and identical initialisation
seeds, so seed-level variance largely cancels.

- **H-P1** is supported if the 95% CI on Δ(5), pooled across datasets, excludes 0
  and is positive.
- **H-P2** is supported if the 95% CI on the bootstrap slope of Δ against log₂(n)
  excludes 0 and is negative.

A Welch t-test over seeds is **not** used: it captures training variance only,
and with n_test = 156 on BreastMNIST the AUC standard error (≈0.03–0.04) exceeds
the effects at stake.

### Decision rule, fixed in advance

| Outcome | Manuscript claim |
|---|---|
| H-P1 **and** H-P2 supported | Scarcity-dependent quantum advantage, attributed to function-class restriction acting as a regulariser |
| H-P2 only | Monotone trend reported; no claim of positive advantage at any n |
| Neither | Crossover reported as an **unreplicated exploratory observation**; headline reverts to "no parameter-efficiency advantage" |

### What would falsify H-P

Δ(5) ≤ 0, or a non-negative slope. Either outcome is reported with the same
prominence as a positive one.

---

## 3. Secondary hypotheses

Declared now; each tested once; all enter the same correction family.

**H-S1 (spectral richness, Q4).** Widening the spectrum from 3^d = 81 to 5^d = 625
at identical parameter count will *hurt* at n ∈ {5,10} and *help* at n ∈ {50,100}.
Test: paired bootstrap on AUC(`quantum_reupload`) − AUC(`quantum_vqc`), frozen, d=4.
This prediction was recorded on 12 August 2026, **before** the Q4 data existed.

**H-S2 (dequantization, Q2).** The VQC does not match a direct fit over its own
function class. Test: paired bootstrap against `fourier_rff`, on the corrected
canonical-frequency implementation. Prior results used a 68-effective-dimension
basis and are superseded.

**H-S3 (calibration, Q5).** Under AWGN the VQC's Macro-F1 degrades
disproportionately to its AUC — a calibration failure, not a ranking failure.
Test: at each σ, the ratio of relative F1 loss to relative AUC loss, plus ECE and
predicted-probability spread. Arms: `quantum_vqc`, `matched_param_fullrank`,
`fourier_rff`, `linear`.

**H-S4 (encoder adaptation, Q3).** Head choice matters less when the encoder can
adapt. Test: the interaction between encoder policy and |Δ| between arms.

---

## 4. Correction

Benjamini–Hochberg FDR at α = 0.05 across the declared family:

- H-P1, H-P2 (2 tests)
- H-S1 at 5 shot levels (5)
- H-S2 at 5 shot levels (5)
- H-S3 at 4 noise levels (4)
- H-S4 (1)

**Family size = 17.** Raw and adjusted p-values are both reported.

Anything outside this list — per-dataset breakdowns, d=8/16, adaptive-encoder
cells, the diagnostic tables — is exploratory, is labelled exploratory, and is
excluded from the correction family.

---

## 5. Effect sizes and practical significance

Cohen's d accompanies every p-value. Where a difference is statistically
significant but smaller than 0.01 AUC, the manuscript will say so explicitly:
below that threshold the difference is smaller than the test-set sampling error
on the smallest dataset and carries no clinical meaning.

---

## 6. Threats to validity, and how each is handled

| Threat | Status |
|---|---|
| Unmatched parameters | Both primary arms at exactly 24, verified programmatically |
| Rank handicap in the classical control | **Found and fixed** — full-rank variant used for confirmation |
| Regularization asymmetry | Identical weight decay, LR, clipping across arms |
| Gradient clipping binding per-arm | Threshold raised to 2× largest observed p95; never binds |
| Validation leakage | Val subsampled to match training scarcity |
| Non-stratified sampling | Per-class stratified draws; all classes asserted present |
| Augmentation confounded with freezing | Augmentation off on both sides of every frozen comparison |
| Undertrained quantum arm | Convergence audited: best-epoch 52.9 vs 56.9/57.9 |
| Noise injected in wrong coordinates | Injected in physical pixel space; round-trip tested |
| Test-set sampling variance ignored | Nested bootstrap resamples test indices |
| Multiplicity | BH-FDR over a declared family of 17 |
| Post-hoc hypothesis | This document, committed before confirmatory data |
| Untuned angle scale favouring one arm | Swept {π/2, π}; sweep reported |
| Circuit depth untuned | L ∈ {1,2,4} swept; selection reported |
| Simulator ≠ hardware | Finite-shot ablation at 1024 shots |
| Irreproducible runs | Exact pins, seeded RNGs, cuDNN deterministic, git SHA in every shard |

### Accepted limitations, to be stated in the manuscript

1. **No quantum advantage can be demonstrated** at 4–16 qubits on a state-vector
   simulator; the model is classically simulable by construction.
2. **Single-encoding spectrum.** Without re-uploading the spectrum is the most
   restricted possible. H-S1 probes this; a full re-uploading study is out of scope.
3. **Ceiling effects.** BloodMNIST, PathMNIST and PneumoniaMNIST sit at 0.94–0.98
   AUC at d=4, compressing the range in which any head can differ.
4. **One backbone, one dataset family.** ResNet-18 and MedMNIST only.
5. **The diagnostic used the rank-limited control.** Diagnostic and confirmatory
   numbers are reported separately and never pooled.

---

## 7. Execution order

1. Q4 completes (running).
2. `matched_param_fullrank` added; parity verified programmatically.
3. **This document committed.** Commit SHA recorded here: `________`
4. Confirmatory sweep, 40 seeds, 1,600 runs (~35 h).
5. Q5 robustness, all four arms, AUC + F1 + ECE at every σ.
6. Statistics per §3–5.
7. Figures.
8. Manuscript.

No confirmatory analysis begins before step 3 is committed and pushed.

---

## 8. Authorship of this plan

Drafted by the analysis assistant, reviewed and approved by the PI. Any
deviation after step 3 must be recorded as an amendment in this file, with a
date and a reason, and disclosed in the manuscript.
