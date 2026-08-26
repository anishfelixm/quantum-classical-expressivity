# Pre-Registration / Analysis Plan

**Written:** 14 August 2026
**Amended:** 17, 19, 26 August 2026 — see §9. Every amendment is dated, gives a
reason, and is disclosed in the manuscript.
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

**All numbers in this table were produced under Macro-F1 checkpoint selection
and are not comparable with anything produced after Amendment 2 (§9).**

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
| Bottleneck | d = 4, learned projection (`bottleneck="learned"`) |
| Datasets | all four |
| Shots/class | 5, 10, 20, 50, 100 |
| Seeds | **40** (`config.CONFIRMATORY_SEEDS`, fixed before launch) |
| Learning rate | per-arm, selected under Amendment 3 (§9) |
| Checkpoint selection | validation AUC (Amendment 2, §9) |
| Augmentation | off (required for feature caching; identical for both arms) |
| Shard namespace | `01_frozen_tuned` — never mixed with diagnostic shards |
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

This statistic requires per-sample predictions. If
`04_statistical_analysis.py` reports `SEED-LEVEL FALLBACK IN USE`, the
pre-registered analysis has **not** been computed and no number from that run may
be quoted.

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

**H-S5 (mechanism — restriction is sufficient).** *Added 26 Aug, Amendment 4.*
If restriction rather than quantumness produces the crossover, a purely classical
head with *fewer* parameters must reproduce it. With
Δ_r(n) = AUC(`low_rank`, rank r) − AUC(`low_rank`, rank 8), frozen, d=4:

- **H-S5a:** Δ₀(5) > 0 — the most restricted classical head helps at extreme scarcity
- **H-S5b:** the slope of Δ₀ on log₂(n) is negative

Prediction recorded before `10_capacity_sweep.py` was run. A classical head at
16 parameters reproducing the quantum crossover is a **stronger** result than any
quantum advantage: it would show the effect is classically reproducible by
restriction alone.

**H-S6 (the head, not the projection).** *Added 26 Aug, Amendment 5.*
The primary contrast is unchanged in sign when the 256→d bottleneck is frozen
rather than learned. Test: Δ(5) and the log₂(n) slope recomputed under
`bottleneck="pca"` and under `bottleneck="random"`.

Motivation: with a learned bottleneck the head holds 24 of 1,062 trainable
parameters (2%); under a frozen projection it holds 24 of 34 (~70%). If the sign
of Δ survives both an optimal projection and a random one, the result is a
property of the heads and not of a 1,028-parameter learned compressor adapting to
whichever head follows it.

---

## 4. Correction

Benjamini–Hochberg FDR at α = 0.05 across the declared family:

| Hypothesis | Tests |
|---|---|
| H-P1, H-P2 | 2 |
| H-S1, 5 shot levels | 5 |
| H-S2, 5 shot levels | 5 |
| H-S3, 4 noise levels | 4 |
| H-S4 | 1 |
| H-S5a, H-S5b *(Amendment 4)* | 2 |
| H-S6, 2 bottleneck policies *(Amendment 5)* | 2 |

**Family size = 21** (was 17 before Amendments 4 and 5).

Enlarging the family costs power on the primary test, and that cost is accepted
deliberately: an under-declared family is anti-conservative, which is the worse
error. `04_statistical_analysis.py --family-size 21` is the invocation used for
every reported table.

Anything outside this list — per-dataset breakdowns, d=8/16, adaptive-encoder
cells, the diagnostic tables, the depth and angle-scale sweeps, the tanh ablation
— is exploratory, is labelled exploratory, and is excluded from the correction
family.

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
| Unmatched parameters | Both primary arms at exactly 24, asserted in `tests/test_parity.py` |
| Rank handicap in the classical control | **Found and fixed** — full-rank variant used for confirmation |
| Parity unreachable at d>4 | **Fixed** — `low_rank` at rank 2 gives 6d parameters at any d |
| Regularization asymmetry | Identical weight decay and clipping across arms |
| Learning rate favouring one arm | **Tuned per arm** on validation, disjoint seeds (Amendment 3) |
| Gradient clipping binding per-arm | Threshold raised to 2× largest observed p95; never binds |
| Validation leakage | Val subsampled to match training scarcity |
| Non-stratified sampling | Per-class stratified draws; all classes asserted present |
| Augmentation confounded with freezing | Augmentation off on both sides of every frozen comparison |
| Undertrained quantum arm | Convergence audited: best-epoch 52.9 vs 56.9/57.9 |
| Selection metric interacting with calibration | **Fixed** — selection on AUC, F1-selection reported as sensitivity (Amendment 2) |
| **Backbone not actually frozen** | **Now proven** — `11_flow_verification.py` compares every parameter and buffer bit-exactly, with a negative control |
| **Gradients may not reach the encoder from the quantum head** | **Now proven** — per-module gradient norms and layer3 weight displacement, per arm |
| **The bottleneck, not the head, doing the work** | **Now controlled** — H-S6, frozen PCA and random projections |
| Noise injected in wrong coordinates | Injected in physical pixel space; round-trip tested |
| Test-set sampling variance ignored | Nested bootstrap resamples test indices |
| Prediction files unreadable by the analysis | **Fixed** — one naming function shared by writer and reader; fallback loudly labelled |
| Multiplicity | BH-FDR over a declared family of 21 |
| Post-hoc hypothesis | This document, committed before confirmatory data |
| Untuned angle scale favouring one arm | Swept {π/2, π}; sweep reported |
| Circuit depth untuned | L ∈ {1,2,4} swept; selection reported |
| Simulator ≠ hardware | Finite-shot and depolarizing ablations, reported as feasibility only |
| Irreproducible runs | Exact pins, seeded RNGs, cuDNN deterministic, git SHA in every shard |

### Accepted limitations, to be stated in the manuscript

1. **No quantum advantage can be demonstrated** at 4–16 qubits on a state-vector
   simulator; the model is classically simulable by construction.
2. **Single-encoding spectrum.** Without re-uploading the spectrum is the most
   restricted possible. H-S1 probes this; a full re-uploading study is out of scope.
3. **Ceiling effects.** BloodMNIST, PathMNIST and PneumoniaMNIST sit at 0.94–0.98
   AUC at d=4, compressing the range in which any head can differ.
4. **The crossover lives where both arms perform poorly** (AUC 0.60–0.68 at n=5).
   A reviewer may fairly observe that it compares two weak models. Disclosed.
5. **One backbone, one dataset family.** ResNet-18 and MedMNIST at 28×28
   upsampled to 224 — an artificial setting for medical imaging.
6. **The diagnostic used the rank-limited control and F1 selection.** Diagnostic
   and confirmatory numbers are reported separately and never pooled.

---

## 7. Execution order

1. ~~Q4 completes~~ *(done)*
2. ~~`matched_param_fullrank` added; parity verified~~ *(done)*
3. Validity gate — must pass before any confirmatory run:
   - `pytest tests/` green, including `test_parity.py` and `test_freezing.py`
   - `11_flow_verification.py` — frozen bit-identical, negative control detects
     drift, gradient reaches the encoder from every arm
   - prediction round-trip: two cells, then `04` with **no** fallback banner
4. **This document committed.** Commit SHA recorded here: `________`
5. LR selection (`09`), per Amendment 3. Results recorded in §9.
6. Confirmatory sweep, 40 seeds, 1,600 runs, namespace `01_frozen_tuned`.
7. Q5 robustness, all four arms, all five scarcity levels, AUC + F1 + ECE at every σ.
8. H-S5 capacity sweep, H-S6 bottleneck ablation.
9. Statistics per §3–5, `--family-size 21`.
10. Figures. Manuscript.

No confirmatory analysis begins before steps 3 and 4 are complete.

---

## 8. Authorship of this plan

Drafted by the analysis assistant, reviewed and approved by the PI. Any
deviation after step 4 must be recorded as an amendment in §9, with a date and a
reason, and disclosed in the manuscript.

---

## 9. Amendments

### Amendment 1 — 17 August 2026. Q4 restricted to the frozen encoder.

**Change.** H-S1 is tested with a frozen encoder only, not both encoder regimes.

**Reason.** Q3 established that an adaptive encoder compresses head-level
differences by 3–5×, so a spectral-richness effect is only observable with the
encoder frozen. The adaptive half would have cost four additional days of compute
to measure a difference the same experiment predicts will be absent.

**Effect on claims.** H-S1 is a statement about the frozen setting. Stated as such.

### Amendment 2 — 17 August 2026. Checkpoint selection changed from Macro-F1 to AUC.

**Change.** The best epoch is selected on validation **AUC**. The F1-selected
model is still evaluated and reported as a stated sensitivity analysis.

**Reason.** The manuscript's primary endpoint is AUC, and Macro-F1 depends on the
argmax threshold. The VQC has a documented calibration failure — probability mass
collapsing toward a point — which makes its validation F1 nearly flat across
epochs, so F1-based selection was close to arbitrary *for the quantum arm
specifically*. A selection criterion that behaves differently across arms is part
of the comparison, not a neutral choice.

**Effect on claims.** Every result produced under F1 selection is **not
comparable** with results produced after this date. The affected Q4 shards were
archived to `artifacts/shards/_superseded_f1selection/` and re-run. The
diagnostic tables in §1 predate the change and are reported only as diagnostics.

### Amendment 3 — 26 August 2026. Per-arm learning-rate selection.

**Change.** Each arm's learning rate is selected from a grid rather than shared.

**Protocol, fixed before any tuning run:**

| | |
|---|---|
| Grid | {3e-4, 1e-3, 3e-3, 1e-2}, identical for every arm |
| Criterion | mean **validation** AUC, aggregated over all tuning cells |
| Tuning seeds | 90001–90005, asserted disjoint from `CONFIRMATORY_SEEDS` at import |
| Scope | frozen encoder, d=4 |
| Selection | one global LR per arm, not per cell |
| Reporting | the full LR × AUC sweep appears in the appendix, not only the winners |

**Reason.** Measured mean gradient norms at d=4 differ several-fold across arms
(quantum_vqc 0.48–0.76; linear 0.97–1.39; fourier_rff 1.27–2.55; deep_funnel
2.88–4.79). At a shared learning rate the quantum arm takes systematically
smaller effective steps, so "the VQC underperforms" and "the VQC was
under-stepped" are indistinguishable — the cheapest available objection to a
negative result.

**One global LR per arm, not per cell:** at n=5/class the validation set is 10–20
images, so per-cell selection would mostly fit noise and would let each arm
cherry-pick favourable configurations. The per-regime breakdown is reported as a
sensitivity check and is not used for selection.

**Anticipated risk, recorded in advance.** Tuning may strengthen, weaken or
eliminate the crossover. That is precisely why it runs *before* the confirmatory
sweep; doing it afterwards would mean choosing hyperparameters with knowledge of
the outcome.

**Selected values** (filled in after `09_lr_selection.py`, before step 6):

| Arm | LR | Mean val AUC |
|---|---|---|
| linear | ______ | ______ |
| matched_param_fullrank | ______ | ______ |
| fourier_rff | ______ | ______ |
| quantum_vqc | ______ | ______ |

### Amendment 4 — 26 August 2026. H-S5 added: the mechanism test.

**Change.** A capacity sweep over a classical head (`low_rank`, ranks 0/1/2/4/8)
is added as H-S5. Family size 17 → 19.

**Reason.** The paper's central claim is that the advantage is a *regularization*
effect of restriction. Every existing arm controls something adjacent —
`matched_param_fullrank` controls capacity at one fixed value, `fourier_rff`
controls function class, `quantum_reupload` controls spectral richness — but
nothing varied restriction itself. The mechanism was inferred from the crossover
and then used to explain the crossover, which is circular. H-S5 varies
restriction directly, classically, and asks whether the same crossover appears.

**Why `low_rank`.** Capacity must vary without rank varying, or the two are
confounded — the flaw in `MatchedParamHead`. `I + UVᵀ` is generically invertible
at every rank including 0, and a width-w MLP cannot go below 2d² parameters while
remaining full rank, so it cannot reach the restricted end at all.

**Prediction, recorded before the run.** Low ranks help at n ∈ {5,10} and hurt at
n ∈ {50,100}, with a negative slope. A null result refutes the regularization
explanation and the mechanism claim is revised rather than retained.

### Amendment 5 — 26 August 2026. H-S6 added: frozen-bottleneck control.

**Change.** The primary contrast is repeated under `bottleneck="pca"` and
`bottleneck="random"`. Family size 19 → 21.

**Reason.** Freezing the backbone does not isolate the head. At d=4 with two
classes the trainable budget of the "frozen" experiment is: bottleneck 1,028
(97%), head 24 (2%), classifier 10 (1%). The experiment designed to isolate the
head's function class is dominated by a learned projection forty times its size,
which can reshape the latent space to suit whichever head follows — the same
absorption effect measured at the encoder in Q3, one layer down, and previously
uncontrolled.

Under a frozen projection the head holds ~70% of trainable capacity. Two
projections are used because one alone is attackable: **PCA** is optimal linear
compression, so "the projection was badly chosen" is unavailable; **random**
(Johnson–Lindenstrauss) is arm-agnostic by construction. Agreement between them
is what makes the head ordering a property of the heads.

**Effect on claims.** If the sign of Δ(5) reverses under either frozen policy,
the primary result is reported as contingent on a learned bottleneck — which
would itself be the paper's most interesting finding, and would be stated as
such rather than buried.
