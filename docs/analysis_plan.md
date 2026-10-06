# Pre-Registration / Analysis Plan

**Written:** 14 August 2026
**Amended:** 17, 19, 26, 27 August 2026 — see §9. Every amendment is dated, gives a
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

**H-S7 (readout richness — was the state ever used?).** *Added 27 Aug, Amendment 6.*
The project's founding hypothesis was that superposition gives access to a
2^d-dimensional state. The default readout extracts only d numbers from it —
4 out of 16 at d=4. `quantum_rich` measures every 2-local ⟨X_i X_j⟩ as well,
giving 10 observables from an **identical circuit with identical 24 parameters**.

- **H-S7a:** AUC(`quantum_rich`) − AUC(`quantum_rich_padded`) > 0, pooled
- **H-S7b:** AUC(`quantum_rich`) − AUC(`quantum_vqc`) > 0, pooled

`quantum_rich_padded` is the control. Widening the readout also widens the
shared classifier (`Linear(4,C)` → `Linear(10,C)`, 10 → 22 parameters at C=2), so
a bare rich-versus-vqc gain could be classifier capacity rather than information.
`padded` repeats the same 4 expectations to the same width: identical classifier,
zero new information. **H-S7a is therefore the informative test** and H-S7b is
reported with the parameter delta disclosed.

This does **not** escape dequantization: ⟨X_i X_j⟩ is quadratic in the amplitudes
and lies in the same 3^d span. What changes is how much of that span the
measurement reaches. Prediction recorded before the run.

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
| H-S7a, H-S7b *(Amendment 6)* | 2 |

**Family size = 23** (17 originally; +2 each from Amendments 4, 5 and 6).

Enlarging the family costs power on the primary test, and that cost is accepted
deliberately: an under-declared family is anti-conservative, which is the worse
error. `04_statistical_analysis.py --family-size 23` is the invocation used for
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
| **Optimization steps confounded with scarcity** | **Measured and benign** — at n=5 one epoch is one step, so n=100 gets ~4-7x more steps. Re-running n=5 with a 1000-epoch budget left AUC unchanged (best epochs 13-65, well inside the original cap). Reported with the data. |
| **Readout discards most of the state** | **Now tested** — H-S7, with a padded control at matched parameters |
| **Anatomically invalid augmentation** | **Fixed** — horizontal flip disabled for chest radiographs (Amendment 8) |
| **Frozen projection estimated from 10 images** | **Fixed** — PCA fitted on the unlabelled training pool (Amendment 7) |
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

### Amendment 6 — 27 August 2026. H-S7 added: readout richness.

**Change.** Two arms added, `quantum_rich` (all 2-local ⟨X_i X_j⟩ measured
alongside the singles) and `quantum_rich_padded` (the same 4 values tiled to the
same width). Family size 21 → 23.

**Reason.** The project's founding hypothesis was that superposition gives access
to a larger state. The dequantization result explains why the *function class*
is classical, but it does not address a separate and more basic point: the
default readout extracts 4 numbers from a 16-dimensional state. Twelve
dimensions are never measured at all. "We measured four numbers out of sixteen"
is not an adequate answer to a reviewer asking whether the state was used, and
the alternative costs nothing in circuit parameters.

**Why the padded control is required.** Widening the readout widens the shared
classifier from `Linear(4,C)` to `Linear(10,C)` — 10 to 22 parameters at C=2, a
35% increase in total trainable parameters under a frozen bottleneck. Without a
control, a positive result is uninterpretable. `padded` holds classifier size and
information content fixed while varying nothing, so **H-S7a (rich − padded)** is
the test that isolates measurement richness.

**Scope.** This does not escape dequantization. ⟨X_i X_j⟩ is quadratic in the
amplitudes and lies in the same 3^d span. The claim under test is about
*measurement*, not about function class, and the manuscript states that
explicitly.

### Amendment 7 — 27 August 2026. PCA fitted on the unlabelled training pool.

**Change.** For `bottleneck="pca"`, the projection is fitted on the **full
training split** at a fixed pool seed, not on the n-shot subset.

**Reason.** Measured variance retained on PneumoniaMNIST at d=4:

| regime | samples | variance retained |
|---|---|---|
| n=5 | 10 | **0.8281** |
| n=20 | 40 | 0.6065 |
| n=100 | 200 | 0.6020 |

The 0.83 is an artifact of fitting four components to ten points, not a better
projection. Fitting on the subset would therefore confound "frozen projection"
with "projection estimated from almost nothing" — precisely at n=5, which is
where the effect under test lives.

**No leakage.** Only the feature matrix is read; labels are never touched. It
also matches practice, since unlabelled medical images are cheap and labels are
not: an institution deploying this would fit its projection on everything it has
and spend its annotation budget on the classifier.

**Effect on claims.** H-S6 is now a statement about a projection fitted on
unlabelled data. Stated as such.

### Amendment 8 — 27 August 2026. Horizontal flip disabled for chest radiographs.

**Change.** `config.NO_HFLIP_DATASETS = ["pneumoniamnist"]`. Random rotation
(±10°) is retained for every dataset.

**Reason.** Mirroring a chest radiograph moves the heart to the right side. That
is situs inversus — a rare congenital condition, not a benign second view of the
same patient. Training on mirrored chests teaches the model that laterality
carries no information, and it is a standard objection from medical-imaging
reviewers.

**Scope.** Augmentation is off in `01` and `03`; it is enabled only in
`06_premise_check.py`. The affected numbers are therefore the premise-check
compression curve, which is reported as a diagnostic. BreastMNIST (ultrasound),
BloodMNIST (cell microscopy) and PathMNIST (histology) have no comparable
laterality constraint and are unchanged.

### Validity finding — 27 August 2026. Step budget is not a confound.

Not an amendment; no hypothesis or family changes. Recorded because it closes a
threat that would otherwise be an unanswered reviewer question.

At n=5 with two classes the training set is 10 images, one batch, so **one epoch
is one gradient step**. At n=100 it is seven. Models across the scarcity axis
therefore receive 4–7× different amounts of optimization, and "performance
improves with n" could partly mean "performance improves with more steps."

Re-running PneumoniaMNIST n=5 with `MAX_EPOCHS=1000`, `PATIENCE=200`:

| arm | AUC (3 seeds) | best epoch |
|---|---|---|
| quantum_vqc | 0.8817, 0.9006, 0.9153 | 43, 49, 38 |
| matched_param_fullrank | 0.8878, 0.9141, 0.9183 | 13, 15, 65 |
| linear | 0.9025, 0.8813, 0.8998 | 8, 108, 34 |

Best epochs sit well inside the original 100-epoch cap and the AUCs match the
capped runs. The extra budget changes nothing, so the confound is benign — and
now evidenced rather than assumed. Reported in Limitations with this table.

---

## Amendments written 29 September 2026

Amendments 3a and 9–11 record changes made in late August and early September
that were implemented and used but never written into this plan. Each states
the date the change was made and the date it was written. Amendments 12–14
were written **before** any of the runs they describe.

### Amendment 3a — change made 28 August; written 29 September. Learning-rate grid extended.

**Change.** Grid extended from {3e-4, 1e-3, 3e-3, 1e-2} (Amendment 3) to
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}.

**Reason.** On the registered grid the selected rate sat at the upper boundary,
so the optimum was not bracketed and "the arm was under-stepped" could not be
excluded — the objection Amendment 3 exists to close. The grid was extended in
the direction of the boundary only, with the same tuning seeds (90001–90005),
cells and criterion.

**What informed the change.** Validation AUC on the tuning seeds only. No
confirmatory run existed; no test-set data was consulted.

**Result.** On the registered grid, **all four arms** selected 1e-2 — the upper
boundary. On the extended grid every optimum is interior.

| Arm | Registered-grid pick | Extended-grid pick | Mean val AUC | Changed? |
|---|---|---|---|---|
| linear | 1e-2 | **3e-2** | 0.9048 (1e-2: 0.9010) | yes |
| matched_param_fullrank | 1e-2 | 1e-2 | 0.8845 | no |
| fourier_rff | 1e-2 | 1e-2 | 0.8979 | no |
| quantum_vqc | 1e-2 | 1e-2 | 0.8785 | no |

**Consequence for the confirmatory results.** The extension changed one arm's
rate, `linear`, by +0.004 validation AUC. The primary pair (`quantum_vqc`,
`matched_param_fullrank`) and the H-S2 control (`fourier_rff`) receive exactly
the rate the registered grid alone would have selected. H-P1, H-P2 and H-S2 are
therefore invariant to this amendment.

`linear`'s selected rate lies outside the registered grid. The full sweep is
reported in the appendix. The `grid` field in `lr_selection.json` still lists
the four registered values; its `full_sweep` field holds all six and is the
authoritative record.

### Amendment 9 — change made 28 August; written 29 September. Predictions stored as float32.

**Change.** Per-sample probabilities stored as float32 rather than float16.

**Reason.** Float16 rows summed to 0.9996–1.0004, outside scikit-learn's
tolerance, so every multi-class AUC raised and the analysis silently fell back to
seed-level resampling — the statistic §2 forbids. Recomputed-versus-recorded AUC
differed by up to 4.7e-3.

**Effect.** `12_bottleneck_ablation` and `10_capacity_sweep` were re-run in
float32. Every number reported in the manuscript post-dates this change, and the
integrity check reports `max |recomputed − recorded| = 0.00e+00` on every
namespace used.

### Amendment 10 — change made 29 August; written 29 September. Conditional renormalisation.

**Change.** Stored probability rows are renormalised on load only when they
deviate from 1 by more than `RENORM_TOLERANCE = 1e-6`.

**Reason.** Unconditional renormalisation perturbed float32 scores enough to
break ties differently and shift AUC. Rows that are already valid probabilities
are now read exactly as written.

### Amendment 11 — change made 9 September; written 29 September. Explicit keying of depth and tanh.

**Change.** Circuit depth `L` and the tanh setting enter the shard key whenever
they are explicitly set, not only when they differ from the configuration default.

**Reason.** Keying against a mutable default meant the baseline of each sweep
(L=2; tanh on) carried no key, landed in a different cell from the conditions it
anchors, and could not be compared. It also meant any change to the default
would have silently turned old shards into cache hits for the new setting — the
same latent fault already fixed for learning rates.

**Effect on data.** None. `18_depth` and `20_tanh` were run after the change.

### Amendment 12 — written 29 September. What the reported analyses are.

**(a) Exploratory status, as §4 already specified.** The following are
exploratory, excluded from the confirmatory family, and labelled as such:
d=8 (`15_dim8`), d=16 (`16_dim16`), the full-data row (`17_fulldata`), depth
(`18_depth`), angle scale (`19_angle`), tanh (`20_tanh`), hardware noise (`07`),
Lipschitz constants (`08`), and the leave-one-dataset-out analysis.

**(b) Per-dataset cells are exploratory.** §4 excludes per-dataset breakdowns from
the family. The per-cell tables printed by `04_statistical_analysis.py` are
per-dataset breakdowns. In particular, "BloodMNIST, n=5, Δ = +0.0446" is an
exploratory observation, not a confirmatory result.

**(c) A correction error, found and fixed.** `benjamini_hochberg` used the
declared family size as m even when more tests had been computed, which is
anti-conservative. It affected every table with more than 23 cells: `20_tanh`
(100 cells), `19_angle` (40), and `12_bottleneck` cross-condition tables. m is now
`max(declared, computed)`; the declared size is a floor, never a ceiling. The
affected tables are re-run, and only the corrected values are reported. No table
with 23 or fewer cells changes, including the primary per-cell table.

**(d) The confirmatory table did not exist.** §4 specifies one test per declared
hypothesis, pooled across datasets, with a single BH correction over 23.
`13_family_table.py` produces that table. It is the only source of confirmatory
p-values in the manuscript.

### Amendment 13 — written 29 September, before the runs. Completing the confirmatory family.

Audit of the 23 declared tests:

| Test | Status on 29 Sept | Action |
|---|---|---|
| H-P1, H-P2 | computed | — |
| H-S1 ×5 | **not run under the final protocol** | run now |
| H-S2 ×5 | **`fourier_rff` never run in the confirmatory namespace** | run now |
| H-S3 ×4 | **v1 noise sweep invalid** (below) | run now |
| H-S4 | **not run under the final protocol** | run now |
| H-S5a/b, H-S6 ×2, H-S7a/b | data exist | assembled by `13_family_table.py` |

**H-S1.** `quantum_reupload` added to `01_frozen_tuned`, the same 40 confirmatory
seeds, learning rate inherited from `quantum_vqc` (1e-2) under
`config.LR_INHERITANCE`. One test per shot level: pooled nested bootstrap on
AUC(`quantum_reupload`) − AUC(`quantum_vqc`). Prediction unchanged from
12 August: hurts at n ∈ {5, 10}, helps at n ∈ {50, 100}.

**H-S2.** `fourier_rff` added to `01_frozen_tuned`, 40 confirmatory seeds, tuned
learning rate 1e-2. One test per shot level: pooled nested bootstrap on
AUC(`quantum_vqc`) − AUC(`fourier_rff`). The hypothesis is that the VQC does not
match a direct fit over its own function class, i.e. the contrast is negative.

**H-S3.** The v1 sweep (`03_robustness`) added Gaussian noise to the network
input after 28→224 upsampling and after grayscale→RGB replication. That is a
valid perturbation, applied identically to every arm, so arm-versus-arm
comparisons in v1 are fair — but it is not a model of sensor noise: it gives 64
independent draws per native pixel, and three per pixel on single-channel data,
most of which the ResNet stem averages away. Separately, three arms that H-S3
does not use (`low_rank`, `quantum_rich`, `quantum_rich_padded`) had no tuned
learning rate and trained at the 1e-3 default; the four H-S3 arms were correctly
tuned. The v2 sweep (`03_robustness_v2`) injects noise at native resolution and
channel count and assigns every arm a declared rate. **v2 is the version reported
for H-S3.** v1 is retained, and disclosed as an earlier input-perturbation variant.

The statistic is pinned now, before the v2 data exist. The registered wording is
"the ratio of relative F1 loss to relative AUC loss"; a ratio is undefined as the
relative AUC loss approaches zero, which it does at small σ by construction. The
difference carries the same claim and is defined everywhere:

    G(σ) = [F1(0) − F1(σ)] / F1(0)  −  [AUC(0) − AUC(σ)] / AUC(0)

H-S3 at σ ∈ {0.05, 0.10, 0.15, 0.20}: G_vqc(σ) > 0, pooled nested bootstrap
across datasets. Four tests, as registered. G for `matched_param_fullrank`,
`fourier_rff` and `linear`, and the contrast G_vqc − G_mpfr, are reported
alongside as exploratory. ECE and probability spread are reported descriptively.

**H-S4.** Legacy adaptive-encoder runs predate Amendment 2 and learning-rate
tuning and are not used. New namespace `25_encoder`: `quantum_vqc` and
`matched_param_fullrank`, both freeze policies in one namespace so frozen and
adaptive are paired on seed, tuned rates, 10 seeds (`ALL_SEEDS`), d=4, no
augmentation on either side. Statistic pinned now:

    I = mean over (dataset, n) of |Δ_frozen| − |Δ_adaptive|,   Δ = AUC(vqc) − AUC(mpfr)

nested bootstrap resampling test indices and seeds within each cell. Supported if
the 95% CI on I is positive.

**Family size remains 23.** It is not reduced for any reason.

### Amendment 14 — written 29 September, before the run. Ansatz check; and follow-ups deliberately not run.

**E1 — Ansatz** (`21_ansatz`, 10 seeds, exploratory). `quantum_basic`
(BasicEntanglerLayers ×6, 24 parameters) against `quantum_vqc`
(StronglyEntanglingLayers ×2, 24 parameters). Motivated by the literature — a
published QCNN study reports that performance varies with the circuit — not by
any pattern in our data. Both ansätze sit behind the same single-upload
AngleEmbedding(Y), so both occupy the same 3^d span; this tests whether the
conclusion depends on which coefficient manifold inside it is reached.
*Prediction:* no pooled difference at any n. Corrected by BH within the analysis;
not a member of the confirmatory family.

**Considered and deliberately not run.** The exploratory depth sweep showed
L=2 is not the best depth for the quantum arm on BloodMNIST (both L=1 and L=4
did better at n ≥ 10), and the tanh ablation showed the control's largest tanh
cost at BloodMNIST n=5, the one cell with an exploratory quantum advantage.
Parameter-matched comparisons at L=1 and L=4, and a primary contrast without tanh
on the control, were designed and then **not run**: each would be a comparison
chosen after seeing the data and aimed at the pattern that prompted it. Held-out
seeds guard against seed noise, not against that selection. L=2 remains the
pre-registered depth because it is the only depth at which the VQC and the
classical control have identical parameter counts (24 = 24). The depth and tanh
sweeps are reported as exploratory, with these patterns stated plainly, and
matched comparisons at other depths are named as future work.

### Amendment 15 — written 1 October 2026, before any family-level analysis. How the confirmatory table is computed.

Every choice below is fixed before `13_family_table.py` is run for the first
time. Recorded honestly: per-cell and per-regime H-P, H-S5, H-S6 slope and H-S7
results have been seen; H-S1, H-S2, H-S3 (v2) and H-S4 results have not.

**One statistic per declared test, one correction.** `13_family_table.py`
computes exactly the 23 tests of §4 and applies Benjamini–Hochberg once across
them, m = 23. A test whose data are missing or incomplete is reported as
*not computed*; m is not reduced.

**Verdict.** A test is *supported* when its BH-adjusted p ≤ 0.05 **and** the
observed effect lies in the predicted direction. The unadjusted 95% CI is
reported alongside. This applies §4 to every member of the family, H-P1
included: H-P1's verdict depends on where its p-value ranks among the 23.

**p-values** use the +1 correction, p = 2·min((k≤0 + 1)/(B+1), (k≥0 + 1)/(B+1)),
so no p-value is exactly zero. Slightly conservative; it changes only effects
whose bootstrap distribution lies entirely on one side of zero.

**All statistics are nested bootstraps** (§2): every replicate resamples test
indices and seeds within each cell, with equal weight per cell, B = 2000.

- **H-P2** is the slope of pooled Δ(n) on log₂ n computed *inside* the nested
  bootstrap. `04_statistical_analysis.py` computed H-P2 by resampling the four
  per-dataset deltas at each n, which ignores test-set and seed variance — not
  the §2 statistic. Both are reported; the nested one is the confirmatory value.
- **H-S1, H-S2:** pooled Δ(n) across datasets at each n — five tests each.
- **H-S3:** G_vqc(σ) pooled across all 20 dataset × n cells, one test per
  σ ∈ {0.05, 0.10, 0.15, 0.20}; test indices and seeds resampled jointly across
  σ = 0 and σ, so each G is paired.
- **H-S4:** I pooled across all 20 cells; frozen and adaptive share resampled
  test indices and seeds.
- **H-S5a:** Δ₀(5) pooled across datasets. **H-S5b:** nested slope of Δ₀.
- **H-S6:** one test per frozen policy. §3 names both Δ(5) and the slope; the
  test statistic is **Δ(5)**, the sign-bearing quantity that H-P1 itself tests,
  and the slope is reported descriptively. Supported if Δ(5) > 0 under that
  policy.
- **H-S7a, H-S7b:** pooled across all 20 dataset × n cells — the only reading
  of "pooled" that yields one test each.

**Completeness.** Every test requires all 20 cells, or all 4 cells for
per-regime tests, with the full planned seed set. Partial data are reported as
incomplete, never analysed.

### Amendment 16 — written 3 October 2026, before the runs. A powered follow-up to H-S6, and the function-class control for re-uploading.

**Status of what follows.** Neither item is a member of the confirmatory
family, and neither changes any verdict in it. H-S6's confirmatory verdict —
*not supported* — is fixed. Both items are reported alongside the family table
with their own correction.

**(a) Why H-S6 needs a follow-up.** H-S6 was run in `12_bottleneck`: 5 seeds,
at the default learning rate 1e-3, because that experiment never received the
tuned rates. The exploratory companion analysis showed that the same
experiment could not detect the n=5 advantage even under its own *learned*
bottleneck (+0.030, 95% CI [−0.020, +0.088]). H-S6's null was therefore
uninformative: the test lacked the power to see the effect it was testing for.

**Design.** The primary's protocol exactly, with one thing changed:
`quantum_vqc` and `matched_param_fullrank`, d=4, frozen backbone, all 40
`CONFIRMATORY_SEEDS`, learning rate 1e-2 for both arms, bottleneck frozen as
PCA (fitted on the unlabelled full training split, Amendment 7) or as a random
Johnson–Lindenstrauss projection (seeded per run). Namespace
`26_bottleneck_tuned`. The learned-bottleneck reference is the primary itself
(`01_frozen_tuned`), identical in seeds, cells, rate and code path.

**Stated limitation.** The rate 1e-2 was selected with a learned bottleneck.
It is not re-tuned for the frozen policies; it is identical for both arms, so it
cannot favour either, but the absolute performance under a frozen bottleneck
may not be optimal.

**Statistic.** Δ(5) under each frozen policy, pooled across datasets, nested
bootstrap exactly as in Amendment 15; BH across the two policies (m = 2).
Per-n rows and the slope are descriptive.

**Prediction.** Under both frozen policies Δ(5) is **not** positive — the
interval includes zero or lies below it — that is, the n=5 advantage depends on
the learned bottleneck. Basis: the exploratory point estimates in
`12_bottleneck` (learned +0.030, PCA −0.028, random −0.055 at n=5).

**(b) E7 — the function-class control for the re-uploading VQC.** H-S1
showed re-uploading beats the single-encoding VQC at n ≥ 10, and an exploratory
contrast showed it beats the matched classical control at four of five n. The
single-encoding VQC has a function-class control (H-S2, `fourier_rff`): a direct
fit over its own trigonometric span, which beat it at n ≥ 10. The re-uploading
VQC has none. The test suite verifies its output lies in the {−2..2}^d span to a
residual below 1e-8, so the control is `fourier_rff_r2`: all 312 canonical
frequencies of {−2..2}^4, 624 features, 2,500 parameters — a direct fit, not
parameter-matched, exactly as `fourier_rff` is for H-S2. Added to
`01_frozen_tuned` with the same 40 seeds, rate inherited from `fourier_rff`
(1e-2). Exploratory, motivated by the data, and labelled so.

**Statistic.** Pooled Δ(n) = AUC(`quantum_reupload`) − AUC(`fourier_rff_r2`) at
each n; BH within the analysis (m = 5).

**Prediction.** Δ(n) < 0 at n ≥ 10, mirroring H-S2: the re-uploading VQC does
not match a direct fit over its own function class.

### Amendment 17 — written 6 October 2026, before the run. A powered follow-up to H-S5.

**Status.** Not a member of the confirmatory family; changes no verdict in it.
H-S5's confirmatory verdict — *not supported* — is fixed. Reported alongside it.

**Why.** H-S5 ran in `10_capacity` at the default rate 1e-3 with 10 seeds —
outside the primary's protocol, exactly as H-S6 did before Amendment 16. Its
interval for Δ₀(5), [−0.0380, +0.0114], excludes a restriction effect as large
as the primary advantage (+0.0142) but not a smaller one. Amendment 16 placed
one mechanism test on the primary's protocol; this places the other there too.

**Design.** `low_rank` at rank 0 (8 parameters) and rank 8 (72 parameters), d=4,
frozen backbone, learned bottleneck, all 40 `CONFIRMATORY_SEEDS`, rate 1e-2
(inherited from `matched_param_fullrank` under `config.LR_INHERITANCE`).
Namespace `27_capacity_tuned`. The feature cache for these seeds and cells
already exists from the primary, so no backbone pass is repeated.

**Statistics.** Mirroring H-S5a and H-S5b: Δ₀(5) = AUC(rank 0) − AUC(rank 8),
pooled across datasets, and the slope of Δ₀ on log₂ n, both by the nested
bootstrap of Amendment 15; BH across the two (m = 2).

**Prediction.** No restriction effect: Δ₀(5) is not positive and the slope is
not negative. Basis: the original H-S5 (Δ₀(5) = −0.0124, slope +0.0029).
