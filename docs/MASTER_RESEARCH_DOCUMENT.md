# Master Research Document

**Project:** Quantum–Classical Expressivity Under Extreme Latent Compression
**Repository:** `github.com/anishfelixm/quantum-classical-expressivity`, branch `feature/journal-expansion`
**Version 3.0 — 26 August 2026.** Supersedes v2.0 (19 Aug): the thesis changed.
**Target venue:** IEEE Access. Submission target ~26 September 2026.

> Read this first. Then `docs/analysis_plan.md` (binding pre-registration,
> including the amendment log) and `docs/STATE.md` (what is running now).

---

## 1. The finding

> **The quantum head's apparent advantage under extreme data scarcity is an
> artifact of the learned compression layer, not a property of the head.**

At d=4 with two classes, the "frozen backbone" experiment's trainable budget is:

| Component | Parameters | Share |
|---|---|---|
| bottleneck `Linear(256,4)` | 1,028 | **97%** |
| head | 24 | 2% |
| classifier `Linear(4,2)` | 10 | 1% |

The experiment designed to isolate the head's function class was dominated by a
learned projection forty times its size. Freeze that projection and the head
holds ~70% of trainable capacity — and the result inverts:

| shots/class | learned bottleneck | frozen PCA | frozen random |
|---|---|---|---|
| 5 | **+0.014** | **−0.007** | **−0.181** |
| 100 | −0.011 | −0.051 | −0.075 |

*(Δ = AUC(quantum_vqc) − AUC(matched_param_fullrank), pilot: 2 datasets, 3 seeds.
Full sweep pending.)*

The advantage at n=5 exists only when a 1,028-parameter projection is free to
reshape the latent space around whichever head follows it. This is the same
absorption effect measured at the encoder in Q3, one layer further down.

### Two hypotheses, both refuted by their own controls

**"Superposition gives access to more states."** Refuted analytically and
numerically. With `AngleEmbedding(rotation='Y')` and no re-uploading, the measured
output lies exactly in a 3^d = 81-dimensional classical trigonometric span —
verified to 1e-16 across six configurations, with a wrong-frequency negative
control failing at 0.908. The 16-dimensional state exists; only 4 expectation
values are read out, and those provably live in a classically constructible span.
This reproduces Schuld/Sweke/Meyer (2021) for this architecture.

**"Restriction acts as a regulariser."** The replacement thesis, and the capacity
sweep does not support it either. Varying a classical head's capacity from 8 to
72 parameters at fixed full rank gives Δ = −0.004 at n=5 and +0.006 at n=100 —
restriction *hurting* under scarcity, the opposite of the prediction, though both
magnitudes are within noise. *(Pilot; full sweep pending.)*

### What the paper therefore claims

A dequantization-controlled evaluation finding **no parameter-efficiency
advantage** for a variational quantum head, and identifying the learned
compression layer as the source of an apparent advantage reported in the
authors' own prior work. Negative, mechanistic, and supported by controls that
are stronger than this subfield's norm.

**Working title:** *Where Did the Quantum Advantage Go? Learned Compression, Not
Quantum Computation, Explains Few-Shot Gains in Hybrid Medical Image Classifiers.*

---

## 2. Architecture

```
image (28x28, upsampled 224x224)
  -> ResNet-18 truncated after layer3, ImageNet-pretrained
  -> global pool -> h (256-d)
  -> bottleneck 256 -> d        [learned | frozen PCA | frozen random]
  -> z_tilde = tanh(z) * ANGLE_SCALE        applied to EVERY arm
  -> HEAD                       the only thing that varies
  -> Linear(d, C) classifier    shared by every arm
```

| `freeze_policy` | Meaning | Trainable |
|---|---|---|
| `"all"` | backbone frozen | 1,038 |
| `"layer3_only"` | layer3 unfrozen | 2,100,750 |

There is no `"frozen"` value; using it raises.

**Manuscript note.** tanh is applied to every arm, so `linear` is really
tanh-then-linear. Call it an *identity head*; `config.ARM_DISPLAY_NAMES` holds
the labels.

---

## 3. The arms

| Arm | Role | Head params (d=4) |
|---|---|---|
| `linear` | capacity floor | 0 |
| `mlp` | non-linearity, zero parameters | 0 |
| `deep_funnel` | failure is not a depth problem | — |
| `matched_param` | **rank-limited — DIAGNOSTIC ONLY** | 24 |
| `matched_param_fullrank` | capacity control, full rank **at d=4 only** | 24 |
| `low_rank` (rank=2) | capacity control, full rank **at any d** | 24 |
| `fourier_rff` | function-class control | 324 |
| `fourier_exact` | function-class ceiling (d ≤ 8) | 328 |
| `quantum_vqc` | treatment, spectrum 3^d = 81 | 24 |
| `quantum_reupload` | spectrum 5^d = 625, same parameters | 24 |
| `pca_svm` | non-neural reference, outside the test family | — |

### The parity identities

```
MatchedParamFullRankHead:   d² + 2d = 6d   ⟺  d = 4   ONLY
LowRankHead:             2·d·r + 2d = 6d   ⟺  r = 2   ANY d
```

At d=8 a dense d×d matrix costs 64 against a 48-parameter budget, so exact parity
and full rank are unreachable in dense form above d=4. `low_rank` at rank 2 fixes
that — though d=8/16 is now out of scope (§8).

---

## 4. Status

### Structural claims — PROVEN, 26 Aug (`11_flow_verification.py`)

| Claim | Evidence |
|---|---|
| Frozen backbone bit-identical | 0 params, 0 buffers changed, max delta 0.00e+00, six arms |
| The check is not vacuous | negative control without `set_bn_eval()`: 45 buffers drift |
| Gradient reaches encoder from every head | `quantum_vqc` backbone norm 0.2008, layer3 displacement 0.5259 |
| Frozen blocks receive none | no gradient in the frozen regime, all arms |

BatchNorm running statistics are buffers, not parameters, so `requires_grad=False`
does nothing to them — only `eval()` mode does. That is the failure mode this
proof exists to exclude.

### Experiments

| Q | Question | Status |
|---|---|---|
| Q0 | Compression cost | answered: ≈0.002 AUC |
| Q1 | Efficiency at 24 vs 24 | answered: tie (31/40) |
| Q2 | Does the VQC exploit its own class? | answered: no |
| Q3 | Encoder absorption | answered: yes, 3–5× |
| Q4 | Spectral richness | complete — **read `--summary-only`** |
| — | Validity gate | ✅ **PASS** |
| Q6 | Capacity sweep (H-S5) | pilot done, **not supported**; full sweep pending |
| Q7 | Bottleneck ablation (H-S6) | pilot done, **sign flips**; full sweep pending |
| — | LR selection | pending, ~7h |
| — | Confirmatory sweep | pending, ~22h |
| Q5 | Input noise, all five regimes | pending, ~12h |

Everything before 17 Aug used Macro-F1 checkpoint selection and is **diagnostic
only** — not comparable with anything after Amendment 2.

---

## 5. Open questions that must be resolved before writing

**Below-chance AUC.** PneumoniaMNIST, n=5, random projection, quantum:
0.257 / 0.305 / 0.325 across three seeds. Three seeds all strongly
anti-predictive is not noise. Either the model reliably learns an inverted
relationship on 10 training images, or checkpoint selection on a 20-image
validation set is picking anti-correlated epochs. Diagnose before reporting.

**PCA fitted on 10 images.** At n=5 the frozen projection is estimated from 10
samples for 4 components, so "frozen" is confounded with "badly estimated."
**Fix: fit PCA on the full unlabelled training pool** — no labels, no leakage,
and it matches practice, since unlabelled medical images are cheap and labels are
not. This materially strengthens H-S6.

**Selection noise.** Taking the maximum over 100 epochs on a 10–20 image
validation set is heavy selection, and it does not bias all arms equally: an arm
whose validation AUC fluctuates more gains a larger optimistic bias. Same class
of problem as the F1/AUC mismatch. Report `best_val_auc` minus test AUC per arm
as a diagnostic.

**Classifier size on multi-class.** `Linear(4,9)` is 45 parameters — nearly twice
the head. On PathMNIST the head is under a third of post-bottleneck capacity even
with a frozen bottleneck. Shared across arms, so not an arm-vs-arm confound, but
report the capacity table per dataset.

---

## 6. Verified facts — do not re-derive

- VQC output lies exactly in the 3^d trigonometric span; residual 1e-16.
- Re-uploading widens the spectrum to (2R+1)^d, verified numerically.
- `diff_method="backprop"` propagates input gradients; adjoint agrees to 3e-7,
  260× slower at d=16.
- Barren plateau: gradient variance falls ~62× from d=4 to d=16.
- Head parameters at d=4: `quantum_vqc` 24, `quantum_reupload` 24,
  `matched_param` 24, `matched_param_fullrank` 24, `low_rank(2)` 24,
  `fourier_rff` 324.
- `fourier_rff` = 80 independent features after the canonical-frequency fix.
- Gradient clipping at 20.0 never binds (largest observed p95 = 9.62).
- Capacity at d=4: learned bottleneck → head 2.3%; frozen → head 70.6%.
- PCA variance retained at d=4: breastmnist 0.622, pneumoniamnist 0.700.
- Encoded amplitudes are **real**. The conference claim about a "complex Hilbert
  space" is wrong and must not reach the draft.

### Runtime

| Setting | Per run |
|---|---|
| Classical head, frozen, cached features | seconds |
| `quantum_vqc`, frozen | ~2.5 min |
| `quantum_reupload`, frozen | 50 s (Breast) → ~13 min (Path) |
| Any arm, adaptive encoder | up to 46 min |

Cost scales strongly with dataset size. BreastMNIST estimates have under-predicted
PathMNIST by 5–20× twice. **Always project from the largest dataset in the sweep.**

---

## 7. Methodology — the non-negotiables

**Scarcity is absolute**, `n ∈ {5,10,20,50,100}` per class, never fractional.
**Sampling is stratified** — random subsets drop classes on 8- and 9-class sets.
**Validation is subsampled** to `min(2n, available)` per class.
**Checkpoint selection is on validation AUC**, F1-selection reported as a stated
sensitivity check.
**Augmentation is off on both sides** of every frozen/adaptive comparison.
**Noise is injected in physical pixel space**: inverse-normalize → inject → clamp
[0,1] → re-normalize.
**Regularization parity is absolute** — identical weight decay and clipping.
**Learning rates are tuned per arm** on validation, disjoint seeds.
**Every run saves per-sample predictions.** `shards.save_predictions()` /
`load_predictions()` are the single naming authority.

**Metrics:** AUC (primary), Macro-F1, average precision, sensitivity, specificity,
balanced accuracy, ECE, probability spread. Per-epoch uses `light=True`; clinical
metrics on test only.

**Statistics:** nested paired bootstrap, B=2000, resampling test indices *and*
seeds, paired on seed. BH-FDR over a declared family of **21**
(`--family-size 21`). Cohen's d with every p-value. Differences below 0.01 AUC
labelled negligible.

Not a t-test over seeds: with n_test=156 the AUC standard error is ≈0.03–0.04,
larger than any effect at stake.

---

## 8. Scope — FROZEN 26 August. No additions.

**In:** capacity sweep · bottleneck ablation (pool-fitted PCA) · LR selection ·
confirmatory sweep · Q5 noise · statistics · figures · manuscript. **≈50 GPU-hours.**

**Out, permanently — each becomes one line in Limitations:**
d=8 and d=16 · full-data reference row · hardware noise (shot, depolarizing) ·
depth sweep · angle-scale sweep · latent probe · tanh ablation.

Every audit so far found something real, which is why the scope kept growing.
It stops here. The claim the paper makes is supported by what is In; everything
Out would strengthen it but is not required for it.

### Limitations to state explicitly

1. No quantum advantage is demonstrable at 4–16 qubits on a state-vector
   simulator; the model is classically simulable by construction.
2. Single encoding — the most restricted possible spectrum.
3. Ceiling effects: three of four datasets sit at 0.94–0.98 AUC at d=4.
4. The crossover lives where both arms perform poorly (AUC 0.60–0.68 at n=5).
5. One backbone, one dataset family, 28×28 upsampled to 224.
6. d=4 only; parity at higher d is implemented but unswept.
7. Simulation only; hardware feasibility untested.

---

## 9. Environment

```
python 3.10 · torch 2.4.1+cu118 · pennylane 0.42.3 · numpy 1.26.4 · medmnist 3.0.2
conda env: qml_v2
GPU: A100-SXM4-40GB MIG 3g.20gb, driver 470 -> CUDA capped at 11.4
simulator: default.qubit + backprop
```

**GPU currently unavailable** — `nvidia-smi` returns `Failed to initialize NVML`.
Restart the pod before launching anything; the 26 Aug pilots ran on CPU.

Edit and commit on Windows; pull and run on Linux.
**Never `git pull` on Linux while a sweep is running** — the git SHA is read at
shard-write time.
**Commit and push before launching anything long.** Four days of compute were
lost once to an uncommitted editor undo.

---

## 10. Plan to submission

| Days | Work |
|---|---|
| 1 | Fix GPU. Diagnose below-chance AUC. Switch PCA to pool-fitted. |
| 2–3 | Capacity sweep + bottleneck ablation, full |
| 4 | LR selection |
| 5–6 | Confirmatory sweep, 40 seeds |
| 7–8 | Q5 noise |
| 9–10 | Statistics, `--family-size 21` |
| 11–13 | Figures (`generate_paper_plots.py` rewrite — last file needed) |
| 14–24 | Manuscript |
| 25–26 | Review, submit |

**Decision point at day 10.** Whatever the numbers say, the paper is written
around them. No new experiments after that date.

---

## 11. Publication assessment — honest

**In favour.** IEEE Access reviews for soundness. The dequantization control has
not been applied to a hybrid CNN+VQC medical pipeline before. The freezing and
gradient-flow proofs answer questions most papers in this subfield leave
unanswered. A paper that refutes its own prior claim with a control it built
reads as careful.

**Against.** It is a negative result. Some reviewers want positive findings.
The subfield is crowded. Single bottleneck dimension, single backbone,
simulation only.

**Cannot be promised.** No methodology guarantees acceptance. What is defensible:
the standard failure modes — unmatched baselines, underpowered tests,
uncorrected multiplicity, unverified mechanism claims, hyperparameters chosen
after seeing results, unproven freezing, unverified gradient flow — have each
been found and closed.

---

## 12. Session conventions

- **Paste the file when in doubt.** The repo is the source of truth, not the chat.
- Verify parity programmatically after touching `heads.py` or `registry.py`:
  `quantum_vqc`, `matched_param_fullrank`, `low_rank(2)` must all report 24 at d=4.
- After touching prediction I/O: two cells, then `04`. **If
  `SEED-LEVEL FALLBACK IN USE` appears, stop.**
- Commit before running anything long.
- **Scope is frozen.** A new idea goes in a "future work" list, not the sweep.
