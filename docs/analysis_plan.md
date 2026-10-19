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

---

## Results of the pre-specified follow-ups — recorded 11 October 2026

Recorded after the runs. Nothing above this line has been edited since it was
written; each prediction stands exactly as dated. Neither follow-up is a member
of the confirmatory family, and no confirmatory verdict changes.

### Amendment 16(a) — H-S6 follow-up (`26_bottleneck_tuned`, run ended 4 October)

| Bottleneck | Δ(5) | 95% CI | p_adj (m = 2) |
|---|---|---|---|
| frozen PCA | −0.0301 | [−0.0421, −0.0179] | 0.002 |
| frozen random | −0.0190 | [−0.0321, −0.0063] | 0.005 |

Δ is negative at every n under both policies. **Prediction confirmed** ("not
positive"): the n=5 advantage depends on the learned projection.

### Amendment 16(b) — E7 (`01_frozen_tuned`, `quantum_reupload` − `fourier_rff_r2`)

Δ(n) = +0.0150, +0.0148, +0.0029, −0.0002, −0.0038 for n = 5, 10, 20, 50, 100;
the intervals exclude zero at n = 5, 10 and 100. **Prediction (Δ < 0 at n ≥ 10)
not confirmed:** wrong in sign at n = 10, null at 20 and 50, held only at 100.

### Amendment 17 — H-S5 follow-up (`27_capacity_tuned`, run ended 6 October)

Δ₀(n) = AUC(`low_rank` rank 0) − AUC(`low_rank` rank 8):

| n | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ₀ | +0.0097 | +0.0118 | +0.0041 | +0.0019 | +0.0059 |
| 95% CI | [+.0007, +.0187] | [+.0035, +.0205] | [−.0031, +.0111] | [−.0040, +.0076] | [+.0016, +.0102] |

| Test (BH m = 2) | Estimate | 95% CI | p | p_adj | Rule's label |
|---|---|---|---|---|---|
| H-S5a: Δ₀(5) > 0 | +0.00973 | [+0.00069, +0.01874] | 0.037 | 0.074 | no restriction effect |
| H-S5b: slope < 0 | −0.00160 | [−0.00369, +0.00051] | 0.143 | 0.143 | no restriction effect |

**Prediction ("Δ₀(5) not positive and slope not negative") — not cleanly
confirmed.** Neither test survives correction, so the decision rule returns "no
restriction effect". But the point estimate of Δ₀(5) moved from −0.0124 in the
original H-S5 to +0.0097, its unadjusted interval excludes zero, and the smaller
head is ahead at every n. The accurate summary:

1. A **scarcity-dependent** benefit of restriction is not supported in either
   run (slope +0.0029, then −0.0016, both intervals spanning zero).
2. A **scarcity-independent** benefit of a smaller head, of up to about
   0.019 AUC at n = 5, is not established after correction and is not excluded.
3. Restriction is therefore **neither established nor excluded** as a
   contributor to the H-P1 advantage (+0.0142), which lies inside that range.
   Earlier documents that called restriction "refuted" overstated the original
   H-S5, whose interval [−0.0380, +0.0114] never excluded an effect of this
   size. That wording is withdrawn.
4. The contrast is between two classical heads. It measures whether restriction
   helps a classical head under this protocol; it does not measure the VQC.


---

## Records of 13 October 2026

Recorded after the reporting analyses; nothing above is edited.

**§7 step 4 — the commit SHA left blank.** This plan was first committed as
`d6e37db` on 17 August 2026 (written 14 August; git verifies only the commit
date). Amendments 4–8 were committed in `ec5cae6` on 27 August, and the file did
not change again until `bd833ba` on 30 September. The confirmatory sweep's
first run was at 19:15 UTC on 31 August, on commit `9dcda12`, whose copy of this
plan contains §2–§4 and Amendments 1–8. Every later experiment ran on a commit
whose copy of this plan already contained its amendment; the table is in
`docs/RESULTS.md` §6.6.

**§6, accepted limitation 4, is inaccurate for the confirmatory data.** It
states that both primary arms sit at AUC 0.60–0.68 at n = 5; that range came
from the diagnostic runs. Confirmatory means at n = 5 are 0.66–0.67
(BreastMNIST), 0.74–0.79 (BloodMNIST), 0.83–0.84 (PathMNIST) and 0.89–0.90
(PneumoniaMNIST). The manuscript reports the confirmatory values.

**H-S3's wording.** The registered sentence calls the F1 collapse "a
calibration failure, not a ranking failure". The test statistic G is unchanged
and its verdict stands. The descriptive ECE promised by Amendment 13 shows the
VQC's calibration error is the lowest or within 0.005 of the lowest of seven
heads at every σ, so "calibration failure" is not supported; the manuscript
calls it decision-threshold degradation.

**Noise model in `07`.** The depolarizing channel is applied once per qubit
after the variational block. AUC invariance holds for that model and not for
noise after every layer (`tests/test_independent_theory.py`). The manuscript
names the model.

---

## Amendment 18 — written 15 October 2026, before the run. The compression check, repeated in the study's own setting.

**Status.** Exploratory. Not a member of the confirmatory family; changes no
verdict in it. Reported in full whatever it shows.

**Why.** The compression check in `docs/RESULTS.md` §3 (`06_premise_check.py`,
commit `538d8361`, 10–12 August) predates this plan and used the end-to-end
protocol of that date: the backbone's layer3 trainable (rate 1e-4),
augmentation on (random rotation of ±10° and horizontal flip on every dataset —
including chest radiographs, since it predates Amendment 8), head and projection
at the then-default rate 1e-3, 3 seeds, and only n = 100 and full data. Every
reported comparison in this study uses a frozen backbone and no augmentation.
With a trainable layer3 the backbone can compensate for a narrow projection, so
that check cannot show how much a 4-dimensional projection of FROZEN features
loses, and it says nothing about n ≤ 50, where the primary effect lives. The
mismatch was found during manuscript verification on 15 October. This design
and its predictions are written before any run of it.

**Design.** `src/01_frozen_backbone_ablation.py`, namespace
`29_compression_frozen`:

| Axis | Value |
|---|---|
| Arm | `linear` (identity head: projection → tanh·π/2 → shared classifier) |
| Backbone | frozen, cached features, no augmentation |
| Projection | learned, width d ∈ {4, 256} — the only thing that differs within a pair |
| Training size | n ∈ {5, 10, 20, 50, 100} per class, and full data (PathMNIST capped at 10,000, as in `17_fulldata`) |
| Datasets | all four |
| Seeds | 10, `config.ALL_SEEDS` (passed explicitly: `config.seeds_for` has no entry for d = 256) |
| Learning rate | the identity head's tuned rate, 3e-2, for both widths (`--use-tuned-lr`) |
| Runs | 4 × 6 × 2 × 10 = 480 |

Within a pair the seed, the split, the cached features and the code path are
identical; only the projection width differs.

**Statistic.** Per dataset × size cell, cost = AUC(d = 256) − AUC(d = 4), by the
nested bootstrap of `04_statistical_analysis.py --compare-key dim
--compare-values 256 4` (test images and seeds resampled), Benjamini–Hochberg
within the analysis (m = 24). Means over the four datasets at each size are
reported descriptively.

**Predictions.**

- **P1.** At n = 100 and with full data, the point estimate of the cost is below
  0.02 AUC on every dataset, as in the earlier check.
- **P2.** At n ≤ 20, the cost averaged over the four datasets is not positive:
  the 256-dimensional projection has 65,792 parameters against 1,028 to fit
  from the same labels.

**What each outcome means for the manuscript.** If P1 holds, the manuscript may
say that compressing frozen features to d = 4 costs little at those sizes. If it
fails on a dataset, the manuscript says that d = 4 is a material constraint on
that dataset with frozen features and does not call compression "nearly free".
P2 is reported as confirmed or not, with the per-dataset values. In every case
the earlier end-to-end check is reported only as context, with its protocol
stated.

**Stated limitation.** The rate 3e-2 was tuned at d = 4 and is not re-tuned for
d = 256. It is identical for both widths.

---

## Amendment 19 — written 15 October 2026, before the run. Does the 100-epoch budget bind at n = 5?

**Status.** Exploratory. Not a member of the confirmatory family; changes no
verdict in it. Reported in full whatever it shows.

**Why.** At n = 5 an epoch is one or two optimisation steps, so the 100-epoch
cap allows at most 200 steps. The only evidence that the cap does not bind is
the "Validity finding — 27 August" above: PneumoniaMNIST only, 3 seeds, three
arms, at the default rate of that date (1e-3), run by editing `config.py`. It
has no script and no log in the repository, and it predates the tuned rate the
primary comparison used. The manuscript cites it. This repeats it under the
final protocol, on every dataset, with records. Found during manuscript
verification on 15 October; written before any run of this design.

**Design.** `scripts/run_budget_check.py`, namespace `30_budget_check`. It runs
`src/01_frozen_backbone_ablation.py` unchanged, with `config.MAX_EPOCHS` raised
from 100 to 1,000 and `config.PATIENCE` from 30 to 200 in-process. Everything
else is the primary protocol: `quantum_vqc` and `matched_param_fullrank`,
frozen backbone, d = 4, learned projection, n = 5, tuned rate 1e-2
(`--use-tuned-lr`), the first ten confirmatory seeds (`config.ALL_SEEDS`), all
four datasets. 80 runs. Each run is paired with the `01_frozen_tuned` run of the
same dataset, seed and arm: same split, initialisation, features and rate.
Because `config.min_epochs_for` clamps at `MAX_EPOCHS // 2`, the minimum before
early stopping rises from 50 epochs to 200 (one training batch) or 100 (two).

**Statistics** (`scripts/analyse_budget_check.py`; descriptive). Per dataset and
arm: mean test AUC under each budget and the mean paired change. Per arm, pooled
over datasets with equal weight: the mean paired change (1,000-epoch minus
100-epoch) with a 95% percentile interval from a bootstrap over seeds within
datasets (B = 2000). Δ(5) = AUC(`quantum_vqc`) − AUC(`matched_param_fullrank`),
pooled, under each budget, with the same bootstrap. Seeds only, not test images:
this checks a protocol constant and enters no test family.

**Predictions.**

- **P1.** For each arm, the pooled change in AUC from the longer budget is
  smaller than 0.01 in absolute value.
- **P2.** Δ(5) is positive under the 1,000-epoch budget.

**What each outcome means for the manuscript.** If P1 fails for an arm, the
100-epoch cap was binding for that arm at n = 5; the manuscript says so in
Methods and Limitations, and the confirmatory verdicts, which were computed
under the declared protocol, stand as declared. If P2 fails, the manuscript
states prominently that the n = 5 advantage does not survive a longer training
budget. Incomplete cells, if compute ends first, are reported as incomplete with
their seed counts.

---

## Records of 15 October 2026 — Amendments 18 and 19 after their runs; corrections found in verification

Recorded after the runs. Nothing above this line is edited. Both amendments ran
on commit `75ad1667`. Logs: `logs_stats_compression_frozen.txt`,
`logs_stats_budget_check.txt`, `logs_paper_facts.txt`,
`logs_independent_theory.txt` (archive `v54_logs_2026-10-15.tgz`, SHA-256
`5d89ac773805c3e08faa22ad2d435a8f3d00be063102a0754594d33ca7b32b10`).

### Amendment 18 — compression with frozen features (`29_compression_frozen`, 480 runs)

Cost = AUC(d = 256) − AUC(d = 4), `linear` arm, 10 seeds, nested bootstrap
(`04 --compare-key dim --compare-values 256 4`), BH within the analysis
(m = 24). Integrity: 480 runs re-scored, maximum difference 0. Bold: 95%
interval excludes zero and BH-adjusted p ≤ 0.05.

| Dataset | n = 5 | 10 | 20 | 50 | 100 | full |
|---|---|---|---|---|---|---|
| BloodMNIST | **+0.0550** | **+0.0180** | **+0.0201** | **+0.0239** | **+0.0211** | **+0.0217** |
| PathMNIST | **+0.0232** | **+0.0180** | **+0.0126** | **+0.0046** | +0.0029 | +0.0012 |
| BreastMNIST | +0.0308 | −0.0083 | +0.0145 | −0.0017 | −0.0004 | +0.0161 |
| PneumoniaMNIST | +0.0036 | +0.0027 | −0.0047 | +0.0002 | −0.0000 | −0.0022 |

Mean over datasets: +0.0281, +0.0076, +0.0106, +0.0068, +0.0059 at
n = 5, 10, 20, 50, 100 (full data: +0.0092). Slope on log₂ n −0.00405
[−0.00797, −0.00032].

- **P1 fails on BloodMNIST** (+0.0211 at n = 100, +0.0217 with full data, both
  above 0.02). It holds on the other three datasets.
- **P2 fails.** The mean cost at n ≤ 20 is positive at every size (+0.0281,
  +0.0076, +0.0106), not ≤ 0.

As the amendment specified: the manuscript says that d = 4 is a material
constraint on BloodMNIST with frozen features, does not call compression
"nearly free", and reports the earlier end-to-end check only as context, with
its protocol. Exploratory observation, found by looking and not tested:
BloodMNIST, where compression costs most, is also the dataset that carries
the H-P1 advantage.

### Amendment 19 — the 100-epoch budget at n = 5 (`30_budget_check`, 80 runs)

All 80 runs trained at (1e-2, 1e-2). All 80 stopped early, at 201–455 epochs;
none reached 1,000.

- **P1 holds.** Pooled change (1,000 − 100 epochs): `quantum_vqc` +0.0012
  [−0.0000, +0.0027]; `matched_param_fullrank` +0.0021 [+0.0002, +0.0042].
- **P2 holds.** Δ(5) on the same ten seeds: +0.0314 [+0.0111, +0.0551] at
  100 epochs, +0.0305 [+0.0101, +0.0540] at 1,000.

On BreastMNIST and PneumoniaMNIST the mean selected epoch is the same under
both budgets (14.3 and 29.4; 15.3 and 17.8). A longer run can only select the
same or a later epoch, so the selected epoch is the same in every one of those
40 runs. Mean test AUC agrees to four decimals, consistent with the
1,000-epoch runs repeating the 100-epoch runs exactly over their shared
epochs.


### Corrections and open items found during manuscript verification

The manuscript was checked line by line against the code, the committed logs
and the run records. Each claim and its source is listed in
`docs/CLAIMS_LEDGER.md`. These items affect this plan or its records.

1. **Reordered pairs under per-layer noise: 7 of 1,770, not 24.**
   - `scripts/print_independent_theory.py` prints the quantity that
     `tests/test_independent_theory.py` asserts on, with the test's own
     circuit, inputs and readout vector (seed 1). The count is 7 of 1,770 at
     p = 0.05.
   - Over 1,000 random readout vectors the count has median 10, 5th–95th
     percentile 1–25 and maximum 39, and is zero for 22 of them.
   - "24" matches no recorded run.
2. **The stated basis of the clip threshold was never written to a file.**
   - `config.py` says the threshold of 20 is twice the largest
     95th-percentile norm seen in early runs (9.62, `deep_funnel` on
     BloodMNIST; maximum 15.87).
   - Git dates the threshold to 10 August, 19:30 (commit `7be3dcf`). The
     shard system that records runs was first committed at 23:55 that day
     (`3e67091`). So those early runs predate any run record.
   - `scripts/verify_claims.py` (section D) searches every shard folder,
     archive and log on the pod for them anyway.
   - The manuscript does not rest on these values. Amendment 21 records the
     norm of every optimisation step of every reported run.
3. **The pre-tuning gradient norms of Amendment 3** ("quantum_vqc 0.48–0.76;
   linear 0.97–1.39; fourier_rff 1.27–2.55").
   - The diagnostic runs (`01_frozen`, frozen backbone, rate 1e-3) carry
     per-epoch pre-clip norms for these arms.
   - `scripts/verify_claims.py` (section C) recomputes the per-dataset and
     per-n means. Until it confirms the source, the manuscript does not quote
     the ranges.
   - The tuning records (`09_lr_selection`, 1e-3) give overall means of 0.632
     (VQC), 0.739 (identity), 1.112 (matched) and 2.734 (Fourier).
4. **Run-record facts** (`scripts/check_paper_facts.py`, `logs_paper_facts.txt`):
   - The early-stopping rule as written matches every run in every
     namespace, except the nine shards of `13_stepbudget` (the 27 August
     check, run at a different budget).
   - Every learning rate in the manuscript's table is the rate recorded.
   - The 100-epoch cap was reached by 57–63% of the runs in `01_frozen_tuned`
     (median selected epoch 79–90), and by 71 of the 80 full-data runs.
   - The epoch-mean pre-clip norm exceeded 20 in:

     | Arm | Epochs above 20 | Maximum |
     |---|---|---|
     | `fourier_rff_r2` | 1,053 of 70,673 | 61.5 |
     | `fourier_rff`, d = 4 | 4 of 69,306 | 26.2 |
     | `fourier_rff`, d = 8 | 2,466 of 17,200 | 112.0 |
     | `fourier_rff`, d = 16 | 428 of 3,613 | 79.6 |
     | `fourier_rff` in `20_tanh` | 1 epoch | — |
     | `linear`, d = 256 (`29_compression_frozen`) | 4 of 20,708 | — |

   - Epoch means came close to 20 elsewhere:
     - `low_rank` rank 8 in `27_capacity_tuned`: 19.9;
     - the matched head with a trainable layer3 in `25_encoder`: 18.3;
     - the matched head in the primary namespace: 15.0.
   - Because only epoch means were recorded, no arm is cleared step by step.
5. **Statements in the manuscript draft that the audit found inaccurate.**
   Each is corrected in the manuscript; the ledger gives the source of each
   correct value.
   - The PennyLane span residuals were 4×10⁻¹⁶ to 4×10⁻¹⁵, not "6×10⁻¹⁶ to
     2×10⁻¹⁵".
   - The NumPy re-derivation residuals were 3.4×10⁻¹⁵ (single encoding),
     1.7×10⁻¹⁵ (two-qubit readout) and 3.4×10⁻¹⁵ (re-uploading), not
     2/2/4×10⁻¹⁵.
   - Adjoint and backpropagation gradients agree to a relative 4×10⁻⁷, not
     3×10⁻⁷.
   - The full-data PathMNIST subsample has 10,002 images, not 10,000:
     per-class rounding of 10,000 × class share.
   - The early-stopping floor takes the values 50, 40, 33, 28 and 20, not
     only 50 and 20.
   - Shot-noise "retention 0.989–1.005" is a ratio of cell means; single runs
     range more widely.
   - The 27 August budget check is replaced by Amendments 19 and 20.

---

## Amendment 20 — written 15 October 2026, before the run. Does the 100-epoch budget bind at any n, for any arm, or with full data?

**Status.** Exploratory sensitivity analysis. Not a member of the confirmatory
family; changes no verdict in it; reported in full whatever it shows. Written
after every primary result was known, which is why it is exploratory. Its
motivation is the training dynamics in the run records (item 4 above), not the
outcome of any contrast.

**Why.**
- Most primary runs, and 71 of 80 full-data runs, reached the 100-epoch cap,
  with validation AUC still improving late.
- Amendment 19 covered only n = 5 and only two arms.
- H-P2, H-S1, H-S2, E7 and the full-data result are all decided where the cap
  binds most.

**Design.** `scripts/replay.py --budget 1000:200`, target `B_<source>`.
- Each run repeats a recorded run with identical keys, re-executed by the code
  of the commit that recorded it (a git worktree).
- `config.MAX_EPOCHS` is set to 1,000 and `config.PATIENCE` to 200
  in-process. Nothing else changes, clipping included.
- The minimum-epoch floor follows `MAX_EPOCHS // 2`, as in Amendment 19.

| Block | Source | Arms | n | Seeds | Runs | Partner (100 epochs) |
|---|---|---|---|---|---|---|
| primary | `01_frozen_tuned` | all five | 5–100 | `config.ALL_SEEDS` (10) | 1,000 | the recorded run |
| full data | `17_fulldata` | `quantum_vqc`, `matched_param_fullrank` | full | its 5 seeds | 40 | `R_17_fulldata` (same GPU, Amendment 21) |

The recorded full-data runs ran on CPU. Their 100-epoch partner is therefore
their GPU re-execution, so that the two budgets differ in nothing else.

**Statistics** (`scripts/analyse_budget.py`):
- The confirmatory nested bootstrap of `13_family_table.py`, B = 2000, with
  the same draws under both budgets. Every change therefore has a paired
  interval.
- (1) Epochs run and selected, and runs reaching 1,000 epochs, per arm and n.
- (2) Pooled AUC change per arm and n.
- (3) Δ under each budget, and its change, for VQC − matched, re-upload − VQC,
  VQC − Fourier, re-upload − Fourier {−2..2}^d and re-upload − matched, per n;
  plus the slope of VQC − matched.
- (4) Full data: AUC per arm and dataset under each budget, and pooled
  VQC − matched under each (descriptive).

**Predictions.**
- **P1.** For every arm at every n, |pooled change| < 0.01.
- **P2.** Wherever a contrast's 100-epoch interval on these ten seeds excludes
  zero, its 1,000-epoch estimate has the same sign.
- **P3.** The VQC − matched slope is negative under the 1,000-epoch budget.
- **P4.** With full data, pooled VQC − matched is not positive under either
  budget.

**Conditional extension, fixed now.** If P1 fails for any arm at any n, the
same re-execution at 1,000 epochs is run for the follow-up namespaces whose
arms and sizes match the failure: `26_bottleneck_tuned`, `27_capacity_tuned`
and `14_readout`, ten seeds. They are analysed the same way.

**What each outcome means for the manuscript.**
- **P1 fails:** Methods and Limitations state the size of the change and where
  it occurs.
- **P2 fails:** the manuscript states, beside the confirmatory verdict and
  with the same prominence, which contrast does not survive the longer budget.
- **P3 fails:** the manuscript states that the shrinking of the advantage
  depends on the 100-epoch budget.
- **P4 fails:** "absent with full data" is withdrawn.

Confirmatory verdicts stand as computed under the declared protocol. Nothing
is replaced.

**Stated limitations.**
- Ten seeds (five at full data), not forty.
- d = 8 and 16, the ablations and the noise sweep stay at 100 epochs unless
  the conditional extension triggers.

---

## Amendment 21 — written 15 October 2026, before the run. Every reported training run re-executed, step by step; clipping removed where it acted.

**Status.** A reproducibility audit plus an exploratory sensitivity analysis.
It changes no confirmatory verdict, and is reported in full whatever it shows.

**Why.**
- Methods says clipping at 20 never acts as an arm-specific learning rate.
  The records show the epoch-mean norm above 20 in Fourier heads and never in
  a circuit (item 4 above).
- The records hold epoch means only, so no arm can be cleared step by step.
- The threshold's stated basis was never recorded (item 2).
- The trained weights of the reported models were never stored. Any question
  about the trained models themselves (Amendments 22 and 23) needs them back.

**Design.** `scripts/replay.py`.
- **Re-execution at the original protocol (`--clip 20`, target `R_<source>`).**
  - Each recorded run is executed again by the code of the commit that
    recorded it (`--code-at`, a git worktree), through the same `run_cell` of
    the same study script, with arguments reconstructed exactly from the
    run's keys.
  - In-process recorders (`scripts/recorders.py`) keep the pre-clip norm of
    every optimisation step and the AUC-selected trained weights. They change
    no computation.
  - Each re-execution is compared with its record: every number in the two
    records (timings excepted) and every stored test probability.
- **Sources.**
  - Through `01`: `01_frozen_tuned`, `14_readout`, `15_dim8`, `16_dim16`,
    `17_fulldata`, `18_depth`, `19_angle`, `20_tanh`, `21_ansatz`,
    `25_encoder`, `26_bottleneck_tuned`, `27_capacity_tuned`,
    `29_compression_frozen`.
  - Through their own scripts: `03_robustness_v2`, `09_lr_selection`,
    `10_capacity`, `12_bottleneck`.
  - This is every namespace from which the manuscript reports a number,
    except the pre-plan compression check (`06_premise`, reported only as
    context) and E8 (`28_latent_saturation`, itself a re-execution).
- **Unclipped re-execution (`--clip inf`, target `U_<source>`)** of every run
  whose `R_` record shows a step above 20 or a non-finite norm. Clipping at
  20 changes a step only when its norm is at least 20. A run with no such
  step is therefore identical with or without clipping, and needs no
  unclipped counterpart.
- **Gate.**
  - Before anything long is launched, one cheap run per configuration of
    every source and commit is re-executed (`replay.py --quick`).
  - Wave 1 starts only if every one is bit-identical to its record.
  - `17_fulldata` is exempt: it ran on CPU and is re-executed on GPU, so it
    is reported as a cross-device comparison, not an identity test.
- **Order** (`scripts/plan_v57_jobs.py`; jobs start in this order):
  1. `R_01_frozen_tuned`;
  2. Amendment 20 primary block;
  3. `R_` of 26, 27, 14, 03, 10, 12, 25;
  4. `R_` of 15, 16, 21, 18, 19, 20, 29, 09;
  5. Amendment 23;
  6. `R_17_fulldata` and Amendment 20's full-data block;
  7. then wave 2, the `U_` runs.

**Statistics** (`scripts/analyse_replay.py`):
1. **Identity.** Per source: re-executed, identical, different, with the
   first differing field.
2. **Steps.** Per source, arm, configuration, dataset and n: runs with a step
   above 20, steps above 20, and the largest step norm.
3. **Unclipped condition.** For each source, a read-only view `X_<source>` of
   hard links is built: the `U_` run where clipping acted, the recorded run
   otherwise.
4. **Recomputation.**
   - Every analysis of `13_family_table.py` (all 23 tests) and of
     `14_exploratory_table.py` is recomputed on the views, with the same
     bootstrap draws as on the recorded namespaces. Each change therefore has
     a paired interval.
   - The family's verdicts follow, BH m = 23.
   - Analyses whose namespaces contain no run in which clipping acted are
     unchanged by construction, and are reported as such.
5. **Learning-rate selection** recomputed on `X_09_lr_selection`.
6. **Everything else.** For every other source, the number of changed runs.
   Where any changed, its reported numbers are recomputed on the view by the
   script that produced them.

**Predictions.**
- **P1.** Every re-execution except `17_fulldata` is bit-identical to its
  record.
- **P2.** No step of any `quantum_vqc` or `quantum_reupload` run exceeds 20.
- **P3.** No confirmatory verdict changes when clipping is removed.
- **P4.** E7's pattern is unchanged: the sign at each n, and whether the
  interval excludes zero.
- **P5.** Amendment 18's P1 and P2 outcomes are unchanged.
- **P6.** The selected learning rate of every arm is unchanged.

**What each outcome means for the manuscript.**
- **P1 holds:** Methods states that every reported run was re-executed
  bit-for-bit, with the count.
- **P1 fails:** the runs that differ, and by how much, are reported. The
  identity argument for the unclipped condition is withdrawn for their
  sources, and those sources' `U_` and `R_` runs are compared directly.
- **Clipping acted in a contrast** (on one arm or both): the unclipped result
  is reported beside the recorded one with the same prominence.
  - A registered confirmatory verdict stands as registered. If the unclipped
    verdict differs, the abstract and conclusion use the weaker of the two.
  - For exploratory analyses (E7, d = 8 and 16 absolute AUCs, compression),
    the text describes the unclipped result, with the clipped one beside it.
- **Methods** reports the step-level clipping facts as measured, and drops
  the unverifiable basis.

**Stored for every re-executed run:**
`artifacts/step_norms/<namespace>/<shard>.npz` and
`artifacts/weights/<namespace>/<shard>.pt`, named in the shard's `recorded`
field.

---

## Amendment 22 — written 15 October 2026, before any trained weights exist. Sensitivity of the trained heads.

**Status.** Exploratory; descriptive.

**Why.** The manuscript's sensitivity result was measured at initialisation
only, because no trained weights had been kept. Amendment 21 regenerates them.

**Data.**
- The `R_03_robustness_v2` weights: seven heads at d = 4, four datasets,
  n = 5–100, ten seeds. These are the heads of the noise analysis.
- The `R_15_dim8` and `R_16_dim16` weights of `quantum_vqc` and `low_rank`.

**Measure.** The estimator of `src/08_lipschitz.py`, applied to each trained
head (projection and classifier excluded):
- 512 inputs, 32 random unit directions per input, ε = 1e-3, the largest
  finite-difference ratio;
- divided by the largest absolute head output over the same inputs.

It is computed on two input sets:
- (a) uniform on [−π/2, π/2]^d, as at initialisation;
- (b) the run's own test latents: the squashed projection of the test images,
  from its saved projection and the cached features, up to 512 of them.

**Reported.** The mean per head, per input set and per n, and the paired ratio
VQC / matched per run.

**Prediction.** The VQC's normalised sensitivity exceeds the matched head's on
both input sets (pooled paired ratio above 1), as at initialisation.

---

## Amendment 23 — written 15 October 2026, before the run. Hardware feasibility of the models as trained in the study.

**Status.** Exploratory feasibility; no classical counterpart; descriptive.

**Why.** The hardware analysis (`07_hardware_noise`) trained its circuits at
the default rate of 1e-3, not the tuned 1e-2 of every reported comparison. Its
noise model also acted only at the end of the circuit, where the AUC is
invariant by construction.

**Design.**
- **(a)** `scripts/replay.py --hardware-tuned`: every cell of
  `07_hardware_noise` is re-run by `07` with the rate set to 1e-2 in-process,
  in namespace `33_hardware_tuned`. Cells:
  - BreastMNIST and PneumoniaMNIST at n ∈ {5, 100}, five seeds;
  - the BloodMNIST cell.

  Shots {exact, 8192, 1024, 256, 64}; end-of-circuit depolarising
  p ∈ {0, 0.001, 0.005, 0.01, 0.05}; up to 500 test images. The weights are
  kept.
  - Consistency check: those weights are compared with the `R_01_frozen_tuned`
    weights of the same dataset, n and seed, which come from the same
    protocol.
- **(b)** Depolarising noise after every layer, on the trained weights from
  (a).
  - It is applied with the independent NumPy simulator of
    `tests/test_independent_theory.py`, at the same p values and on the same
    test images.
  - The simulator must first reproduce the PennyLane head outputs of those
    weights to within 1e-6, or (b) is not reported.

**Predictions.**
- (i) At 1,024 shots, the cell-mean AUC is between 0.98 and 1.02 of exact.
- (ii) End-of-circuit noise leaves binary AUC exactly unchanged.
- (iii) Per-layer noise at p ≤ 0.01 changes binary AUC by less than 0.005 in
  every cell.

---

## Records of 16 October 2026 — the gate, before wave 1

Written after the gate and before any wave-1 run. Nothing here changes a
reported number, a hypothesis, an arm or a statistic.

**Gate (J = 3; J = 8 had been killed by the pod's 16 GiB memory limit).**
20 of 22 checks passed. Two failed, for two different reasons:

1. **`03_robustness_v2` — an error in the replay script, not in the runs.**
   `03`'s `main()` passed the tuned rate as both `lr_head` and `lr_quantum`
   for every arm, and the record stores both under `"lr"`. The key carries
   `lrq` only for circuit arms, and the replay adapter took the rates from the
   key, so for classical arms it passed `lr_quantum=None`. Classical arms have
   no quantum parameter group, so the value does not enter training, but the
   stored field differs and the comparison correctly reported it. The adapter
   now passes exactly the recorded `"lr"` values. The re-run gate decides
   whether that was the only difference.

2. **`09_lr_selection` at `8c80cce` — the feature cache.** The differing run
   (`breastmnist`, n = 5, seed 90001, `linear`, 3e-4) is line `[1/960]` of
   `logs_lr.txt`: the first run of the sweep, preceded by three data-load
   lines. It built the feature cache for that dataset, n and seed.
   `01.get_cached_features` builds a missing cache by constructing the
   ResNet-18 (torchvision), whose random initialisation draws from the global
   PyTorch stream before the pretrained weights are loaded. The head is then
   initialised from the same stream (`build_arm` does not re-seed it). A run
   that finds the cache on disk skips that draw. So a cache-building run's
   head starts from a different, equally random, point of the seeded stream.
   The replay found the cache on disk; the original did not.

**Scope, from the original logs** (`scripts/predict_cold_runs.py`,
`docs/cold_cache_prediction.json`, written now from the committed logs).
Cache-building runs among the reported namespaces:

| log | source | commit | runs | built the cache |
|---|---|---|---|---|
| `logs_confirm.txt` | `01_frozen_tuned` | `9dcda129` | 1,600 | 600 — every `quantum_vqc` run of the 30 seeds outside `ALL_SEEDS` (the first arm run for each new dataset, n and seed) |
| `logs_lr.txt` | `09_lr_selection` | `8c80cce7` | 960 | 60 — `linear` at 3e-4, the first run of each dataset, n and seed |
| `logs_compression_frozen.txt` | `29_compression_frozen` | `75ad1667` | 479 logged | 19 |
| `logs_full.txt` | `17_fulldata` | `e07459a5` | 80 | 20 (cross-device; see below) |
| every other reported log (19 logs) | | | 17,640 | 0 |

**What this means for the results: nothing changes.** Each of these runs was
initialised from the seeded stream, by a draw that depends on neither the data
nor the arm; it is as valid a repetition as any other. Every reported number
was computed from the records, and Section F of `verify_claims.py` reproduced
the published family table exactly from them. What it changes is how a run is
reproduced: a cache-building run is reproduced only if its cache is absent
again.

**What it means for the gate.** The quick sets draw the first seed of
`ALL_SEEDS`, which never built a cache in the confirmatory sweep, so the gate
did not test this case except by accident in `09`. The gate is extended
before wave 1 (below).

**Procedure from now on** (`replay.py`, "FEATURE-CACHE STATE").
- Each run is first re-executed in the state its log line predicts: **cold**
  (the cache is rebuilt for that run in a private temporary folder, which is
  then deleted; the shared cache is never written) or **warm** (the cache on
  disk is read). Runs with no log line start warm.
- If that execution is not identical, the other state is tried once. The
  record keeps the state that reproduced the run, the logs' prediction and
  every attempt.
- On a cold execution, the rebuilt features are compared bit for bit with the
  cache file. If they differ (a different device or library), the file's
  features are used, since those are the ones the original run computed and
  saved, and the record says so.
- `U_` (unclipped) runs use the state their `R_` run established, so `U_`
  and `R_` differ only in the clip.
- `B_` (1,000-epoch) runs are now checked, not just executed. Up to the epoch
  at which the reference run stopped, a long run is the same computation (the
  scheduler's patience is a constant 15, independent of the budget), so every
  per-epoch series must agree bit for bit over those epochs. A long run that
  fails this is retried in the other cache state once.
- `17_fulldata` ran on CPU and is re-executed on GPU, always warm. Identity
  is not expected there (Amendment 21); its `B_` runs are checked against
  `R_17` on the same GPU.

**Prediction P0** (added to Amendment 21's P1–P6, before any of these runs):
every run the logs show building its cache is reproduced bit for bit in the
cold state, and every run they show reading it, in the warm state.
`analyse_replay.py identity` tabulates prediction × state used × verdict.

**Gate additions.**
- One cheapest predicted-cold run per source and commit (`gate_cold_*`): it
  must be bit-identical.
- The `B_` gate now applies the epoch-by-epoch check.

**Clipped side of the clipping contrast.** As Amendment 21 already specifies
for the case where P1 fails, the contrast is computed against `R_` wherever a
namespace has any run that did not reproduce its record. Where every run
reproduced its record, `R_` and the record are the same numbers.
`analyse_replay.py results` applies this rule namespace by namespace and says
which side it used.

**Parallelism.** Each job holds about 3.2 GB; the pod's limit is 16 GiB.
Wave 1 runs with J = 4. A job killed for memory is marked FAIL in
`logs_v57/status.txt`; launching the wave again re-runs only what is missing.

**Methods text that follows from this (one sentence, factual):** "Runs that
built the feature cache drew the head's initialisation after the backbone's
constructor had consumed random numbers; the replay script reproduces them by
rebuilding the cache (Section X: n of n runs re-executed bit for bit)."

---

## Records of 17 October 2026 — wave 1 under way; amendments written before their runs

**Wave 1, interim** (`analyse_replay.py identity`, 17 October 00:30 IST; 311 jobs
finished, 8 marked FAIL and under diagnosis):
- `01_frozen_tuned`: **4,000 of 4,000 re-executions bit-identical** to their
  records, every metric and every stored test probability. The 600 runs the
  logs show building their feature cache were reproduced in that state, every
  one; the 3,395 the logs show reading it, in that state; 5 with no log line,
  warm.
- Every other source re-executed so far (813 runs): identical. P0 and P1 hold
  so far.
- `17_fulldata` on GPU differs from its CPU records, as Amendment 21 expected.

### Amendment 25 — the full-data runs re-executed on the device that recorded them

**Why.** `17_fulldata` ran on CPU. Its GPU re-execution cannot be bit-identical,
so it cannot show the step norms of the reported full-data runs themselves.
Re-executed on CPU, it can.

**Design.** `replay.py` run with CUDA hidden (`CUDA_VISIBLE_DEVICES=`) trains on
CPU and writes `C_17_fulldata`. The planner adds these 80 runs as tier 7, plus a
gate check (`gate_cpu_*`). Cache state as predicted by the logs (20 runs built
their cache: the first arm, `quantum_vqc`, of each dataset and seed).

**Prediction.** Every `C_17_fulldata` run is bit-identical to its record.
- **If it holds:** the full-data runs join the identity count, and their step
  record is that of the reported runs.
- **If it fails:** `17_fulldata` stays as Amendment 21 treats it, a GPU
  comparison. Its clipping contrast is `U_17` against `R_17`, both on the GPU.

**Unchanged:** the budget check's partner for full data stays `R_17` (same GPU).

### Amendment 22 — analysis details fixed before any analysis
`scripts/analyse_trained_heads.py`.
- **Faithful rebuild.** Each model is rebuilt with the code of the commit that
  trained it, and its weights are loaded.
- **Integrity check.** A run is used only if the rebuilt model reproduces the
  test probabilities its re-execution stored:
  - within 1e-5 for `R_15` and `R_16`, which stored them from the same cached
    features;
  - within 1e-4 for `R_03`, which stored them from images, so only the
    floating-point order differs.
  Runs that fail are listed and excluded, never repaired.
- **Pairing.**
  - `R_03`: `quantum_vqc` with `matched_param_fullrank`.
  - `R_15` and `R_16`: `quantum_vqc` with `low_rank`, the parity control
    defined at every d.
- **Pooled ratio.** The geometric mean over runs of S(VQC)/S(partner), with the
  share of runs above 1, per input set and per n.

### Amendment 23 — analysis details fixed before any analysis
`scripts/analyse_hardware_tuned.py`.
- **Simulator convention check.** Before the trained weights exist,
  `scripts/check_simulator.py` checks the NumPy simulator against PennyLane on
  random weights, noise-free and with per-layer depolarising noise. It must
  agree within 1e-6 on both.
- **Per-run requirement, unchanged.** On each trained model, the simulator must
  agree with the PennyLane head within 1e-6.
- **Weight-consistency partner.** The `R_01_frozen_tuned` `quantum_vqc` run with
  the same dataset, n and seed, at `lrh = lrq = 1e-2`, frozen backbone, learned
  projection and default settings.
- **Prediction (iii) is evaluated on cell means**, as worded.
- **Two further outputs.**
  - The simulator's end-of-circuit AUC against 07's recorded AUC, as a sanity
    check.
  - The original 1e-3 cells beside the tuned ones, descriptively.

### Amendment 24 — gradient variance at initialisation (barren plateaus)
**Why.**
- The table in `MATH_VERIFICATION.md` §7 has no producing script. It is
  withdrawn there.
- The loop's logged `quantum_grad_var` is the variance across one circuit's
  parameters within a step, which is not the barren-plateau quantity.

**Design** (`scripts/barren_plateau.py`).
- The quantity: the variance over initialisations of the derivatives of a cost
  with respect to the circuit parameters.
- The head: the project's `VQCHead` at d ∈ {4, 8, 16} and L ∈ {1, 2, 4}.
- Initialisations, 200 draws each:
  - the project's N(0, 0.1²);
  - uniform on [0, 2π).
- Inputs: one fixed batch of 32 latents drawn uniformly from the head's
  operating range.
- Cost: the batch mean of ⟨X₀⟩.

**Predictions.**
- **P1.** Under uniform initialisation at L = 4, the variance is lower at
  d = 16 than at d = 4.
- **P2.** The project's initialisation gives a larger variance than uniform
  initialisation at every (d, L).

**Status.** Exploratory; reported only as context for the d = 16 results.

### Corrections
- **Fig. 2 fit residual.** It is 2.15×10⁻¹⁵ at the fit points and 2.33×10⁻¹⁵ on
  the plotted grid (`figure_data.json`). The text says "below 3×10⁻¹⁵", not
  "≤ 2×10⁻¹⁵".

### Literature check of 17 October

**Method.** Web search over 2025–2026 for hybrid quantum heads on medical
images, matched classical baselines, dequantization and inductive-bias
explanations. Each item was read from its abstract or record page, except one
the server refused (below).

**Newly found, to cite and position against.** None reports this paper's
result: an advantage attributed, under exact parameter parity and
function-class controls, to the learned projection rather than the circuit.

| Item | What it reports | How it relates to this paper |
|---|---|---|
| arXiv:2607.21186 (Rubiños Rodríguez et al., Jul 2026), "Do emulated quantum circuits change what CNNs look at?" | Parameter-matched hybrid vs CNN on retinal OCT and OASIS-1 MRI; the hybrid leads at about 800–2,000 images, the lead narrows and reverses with more data; removing entanglement changes little; "inductive bias" offered as an untested interpretation | The closest empirical pattern (an advantage that shrinks with data). This paper tests the attribution with function-class controls and frozen-projection controls instead of offering it as interpretation. |
| arXiv:2610.17722 (Yamaoka, Chan, Mitarai, 13 Oct 2026), "Representation Geometry and Inductive Biases in Quantum Classifiers across Classical Visual Benchmarks" | Quantum classifiers on strong classical feature extractors; effectiveness depends on input-representation geometry and data scale | The closest conceptual neighbour. It does not report parameter-parity or function-class controls in its abstract. |
| arXiv:2609.24666 (Anthony et al., Sep 2026) | QCNN in the small-data regime, digits and BreastMNIST; a parameter-matched CNN near chance on digits; on BreastMNIST every classical baseline beats the QCNN | Agrees that low parameter count alone does not explain generalisation, and that generalisation is not quantum advantage. |
| *Electronics* 15(20) 4723 (Oct 2026), "Two methodological pitfalls in matched-parameter quantum–classical benchmarking" | Class-imbalance collapse; readout offset at initialisation that no circuit parameter can recentre | Both are already controlled here: balanced n-shot training sets, class weights, AUC as the primary metric, and a shared classifier with a bias term after every head. To be stated in Methods. |
| *Quantum Mach. Intell.* 8:96 (2026), Barbato et al. | Knowledge distillation into a 233-parameter hybrid student, MNIST/Fashion-MNIST; gains attributed to Hilbert-space embedding without a function-class control | An example of the attribution this paper's controls test. |
| Zenodo 10.5281/zenodo.23207872 (Oct 2026) | Code for "dressed quantum heads on frozen biomedical transformers", text classification, pre-registered protocol | Convergent practice; different modality. No results on the record page. |

**Not read.** arXiv:2610.16153, "Architectural Robustness to Label Noise Using
Variational Quantum Circuits for Medical Image Classification" (Oct 2026). The
server refused the request (rate limit). It must be read before submission.

**Verdict.** Nothing found makes the paper unpublishable or pre-empts its
contribution.
- The field is converging on matched baselines, so "we matched parameters" is
  not a contribution.
- The distinct parts remain:
  - exact parity at three dimensions;
  - function-class controls for both encodings;
  - capacity accounting and the frozen-projection result;
  - one pre-registered confirmatory family;
  - every reported run re-executed bit for bit.
- The check is repeated in the week of submission (`WORK_REMAINING.md` item 12).

---

## Records of 20 October 2026 — wave 1 status; how the remaining runs are scheduled

**Wave 1 so far.**
- Every `R_` source except full data is complete: 20,120 of 20,120
  re-executions bit-identical to their records.
- P0 holds for every source. All 699 runs the logs predicted to have built
  their cache were reproduced in that state.
- The 8 FAIL lines in `logs_v57/status.txt` are from the first gate run of
  15 October, when 8 parallel jobs exceeded the memory limit. Every one of those
  checks passed when re-run; no wave job has failed.

**The hardware runs (Amendment 23) stalled.**
- One run took 1,039 minutes. The same cell took about 15 minutes in the
  original 07 run (`logs_hwnoise.txt`: 20 runs in about 5 hours).
- Four ran at once, each with 257 threads, and the pod's memory stood at 16.4
  of 17.2 GB.
- These jobs are stopped. Nothing they would have written exists yet, so
  nothing is lost.
- The full-data jobs queued behind them had not started.

**From now on, each kind of run has its own job list** (`launch_v57.sh` modes):
- `wave1`: the remaining `R_` and `B_` runs, on GPU.
- `cpu17`: Amendment 25, on CPU, one process at a time, as the original ran.
  Its thread count is left at the default, because the original's was.
- `hardware`: Amendment 23, with 8 threads per job (`OMP_NUM_THREADS=8`).
  - The thread count changes only the floating-point order of the CPU
    evaluation.
  - No identity is tested for these runs: they train at a different rate by
    design.
  - The setting is recorded here before the runs.
- `wave2`: the unclipped runs, after wave 1.

No design, prediction or analysis changes.
