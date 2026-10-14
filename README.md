# Variational quantum vs. parameter-matched classical heads under data scarcity

A controlled attribution study on MedMNIST. A frozen ResNet-18 feeds a learned
4-dimensional projection, and only the classification head changes between
arms: a 24-parameter variational quantum circuit (VQC), or classical heads
matched to it on parameter count or on function class.

**Status.** All experiments are complete and the results below are final. The
journal manuscript (IEEE Access) is in preparation. This repository began as the
code for a conference paper; the journal study re-tested that paper's claims
with stronger controls, and several did not survive
([changes from the conference version](#changes-from-the-conference-version)).

> **What this is not.** The circuits have 4–16 qubits and run on a state-vector
> simulator. They are classically simulable by construction, and the trained
> head computes an ordinary trigonometric function (proved and verified below).
> Nothing here is evidence of quantum advantage. The question is narrower: when
> a VQC head beats a matched classical head, where does that advantage come
> from?

---

## Findings

Three evidential tiers, never mixed. **Confirmatory**: the 23 tests declared in
[`docs/analysis_plan.md`](docs/analysis_plan.md), corrected once (BH-FDR,
m = 23). **Pre-specified follow-up**: reruns whose design and prediction were
written before they ran. **Exploratory**: everything else.

| # | Finding | Tier |
|---|---|---|
| 1 | At 5 labelled images per class, the VQC head beats the parameter-matched classical head: ΔAUC **+0.0142** [+0.0025, +0.0256], p_adj = 0.026. | confirmatory |
| 2 | The advantage shrinks with data: slope **−0.0046** per doubling of n [−0.0070, −0.0022], p_adj = 0.002. At n ≥ 50 the matched head is ahead (descriptive). | confirmatory |
| 3 | BloodMNIST carries the advantage. Without it, Δ(5) = +0.0041 [−0.0071, +0.0155]. It reverses at d = 8, is null at d = 16 and is absent with full training data. | exploratory |
| 4 | It exists **only with a learned projection**. With the projection frozen (PCA or random), the VQC head is worse than the matched head at every n: −0.0301 and −0.0190 at n = 5. | pre-specified follow-up |
| 5 | Function-class **restriction** as the cause is neither established nor excluded. A smaller classical head shows no scarcity-dependent benefit in two runs. | confirmatory + follow-up |
| 6 | A direct fit over the VQC's own function class (the 81-term trigonometric basis) beats the VQC at n ≥ 10, by 0.016–0.021 AUC. | confirmatory |
| 7 | Data re-uploading (a wider frequency spectrum) helps at n ≥ 10. At n = 10 this is the **opposite** of the prediction written before the run. | confirmatory |
| 8 | Reading out two-qubit correlators ⟨XᵢXⱼ⟩ as well as ⟨Xᵢ⟩ helps (+0.011). | confirmatory |
| 9 | Under sensor noise, macro-F1 collapses while AUC holds for **every** small head, classical or quantum. The cause is the decision threshold, not miscalibration: the VQC's calibration error is among the lowest of seven heads. | confirmatory + descriptive |
| 10 | Compressing the 256 features to d = 4 costs almost nothing: mean AUC gap +0.0018 against d = 256 at n = 100 and full data. | exploratory (pre-plan, 3 seeds) |
| 11 | Under the VQC head, the projection leaves about one fewer of four latent coordinates unsaturated at n = 5, at the same learning rate. This does not track the advantage. | exploratory |

Of the 23 confirmatory tests, 14 were supported, 1 came out opposite to its
prediction, 1 showed a difference where none was predicted, and 7 were not
supported. Every number, with its interval, is in
[`docs/RESULTS.md`](docs/RESULTS.md).

**Absolute performance at n = 5** (test AUC, mean over 40 seeds):

| Dataset | Classes | Test images | Matched head | VQC head |
|---|---|---|---|---|
| BreastMNIST | 2 | 156 | 0.658 | 0.668 |
| PneumoniaMNIST | 2 | 624 | 0.889 | 0.895 |
| BloodMNIST | 8 | 3,421 | 0.742 | 0.786 |
| PathMNIST | 9 | 7,180 | 0.838 | 0.834 |

**In one sentence.** A small, confirmatory, scarcity-dependent advantage of a
VQC head over a parameter-matched classical head exists, but only when a learned
projection adapts upstream of it. Its cause is not identified, and the head
itself computes a classical function.

---

## Pipeline

```
image 28×28  →  upsampled to 224×224
             →  ResNet-18 to layer3, ImageNet weights, frozen   (0 trainable)
             →  global pool → h ∈ ℝ²⁵⁶
             →  projection Linear(256, d)                       learned | frozen PCA | frozen random
             →  z̃ = tanh(z)·π/2                                 every arm
             →  HEAD                                            the only thing that varies
             →  Linear(outputs, C) classifier                   every arm
```

Trainable parameters at d = 4, binary task: projection 1,028 (97 %), head 24,
classifier 10. The learned projection dominates the budget. This is why
finding 4 matters: the head is a small part of a system that adapts around it.

## Arms

| Arm | Role | Head parameters (d = 4) |
|---|---|---|
| `quantum_vqc` | treatment: RY angle embedding, two StronglyEntanglingLayers, ⟨Xᵢ⟩ readout | 24 |
| `matched_param_fullrank` | primary capacity control | 24 |
| `low_rank` | capacity control at any d (2dr + 2d) | 24 at r = 2; 48 and 96 at d = 8, 16 |
| `fourier_rff` | function-class control: one frequency per ±ω pair of {−1,0,1}⁴, giving 81 = 3⁴ basis functions | 324 |
| `quantum_reupload` | VQC with the input encoded twice; spectrum {−2..2}ᵈ | 24 |
| `fourier_rff_r2` | function-class control for the re-uploading VQC | 2,500 |
| `quantum_rich`, `quantum_rich_padded` | VQC with 2-local readout (10 outputs) | 24 |
| `quantum_basic` | alternative ansatz (BasicEntanglerLayers) | 24 |
| `linear`, `mlp` | floors and references | 0 |

Two kinds of parity, deliberately. `matched_param_fullrank` and `low_rank` match
**parameter count**: is the VQC more efficient per parameter? `fourier_rff`
matches the **function class**: is any advantage specific to the circuit, or
available to any learner over the same trigonometric basis? Exact parity holds
at 24 = 24 = 24 (d = 4), and is checked by `tests/test_parity.py`.
`models/registry.py` also defines diagnostic arms that the reported results
do not use. Among them is `matched_param`, a rank-limited control that turned
out not to be a fair match (see below).

---

## The head computes a classical function

With `AngleEmbedding(rotation='Y')` and no re-uploading, the encoded state has
real amplitudes. Every measured expectation is exactly

```
v_i(z̃) = Σ_{s ∈ {0,c,s}^d} c_s(Θ) ∏_j f_{s_j}(z̃_j),     f_0 = 1, f_c = cos, f_s = sin
```

This is a 3ᵈ-dimensional classical trigonometric span (81 functions at d = 4).
It is the spectrum Schuld, Sweke & Meyer (2021) predict from the encoding
generator. The 24 circuit parameters reach at most a 24-dimensional subset of
that span. With re-uploading the span becomes 5ᵈ. Two-qubit correlators
⟨XᵢXⱼ⟩ stay in the 3ᵈ span.

Checked two independent ways:

```bash
python -m pytest tests/test_fourier_equivalence.py -v   # PennyLane; residuals 6e-16 to 2e-15, wrong basis fails
python -m pytest tests/test_independent_theory.py -v    # plain numpy, no PennyLane or project code
```

The independent rebuild also verifies the depolarising-noise result: noise
applied once after the circuit contracts every ⟨Xᵢ⟩ by (1 − 4p/3) and leaves
binary AUC exactly unchanged. Noise after every layer does not; there, 24 of
1,770 test pairs reorder. Derivations: [`docs/MATH_VERIFICATION.md`](docs/MATH_VERIFICATION.md).

---

## Why the comparison is fair

Each of these was found and fixed during the project, and the reported results
use only the fixed versions. Each one, left in place, could have produced or
hidden an effect.

| Confound | Fix |
|---|---|
| The first "parameter-matched" control was rank-limited | `matched_param_fullrank`, plus `low_rank`, whose parity identity holds at every d |
| Gradient clipping at 1.0 bound on classical arms and never on the VQC, acting as a per-arm learning-rate multiplier | clip at 20.0, twice the largest observed p95 norm, so it never binds; norms are logged |
| Checkpoints selected on macro-F1 while AUC was reported | selection on validation AUC (Amendment 2); F1-selected results reported as a sensitivity check |
| One shared learning rate, while gradient scales differ about 5× across arms | per-arm rates tuned on validation, on seeds disjoint from the confirmatory seeds (`09`) |
| Duplicated Fourier frequencies (both ω and −ω) | one representative per ±ω pair |
| Weight decay applied to classical arms only | identical weight decay for every arm, quantum parameters included |
| Sensor noise clamped in normalised coordinates, and added after upsampling | noise added at native resolution, in physical pixel space: inverse-normalise, add, clamp to [0, 1], re-normalise |
| BH correction too lenient if more tests were computed than declared | m = max(declared, computed); m = 23 |

Other design choices:
- **Scarcity is absolute** (n ∈ {5, 10, 20, 50, 100} per class), never a
  percentage. 1 % of BreastMNIST and 1 % of PathMNIST are unrelated experiments.
- **Sampling is stratified.**
- **Validation is capped** at 2n per class, so model selection never sees more
  labels than training.
- **The backbone stays frozen.** Training changes 0 of its parameters and 0
  BatchNorm buffers, while a deliberately broken negative control drifts
  (`tests/test_freezing.py`, `src/11_flow_verification.py`).
- **Every run saves per-sample test predictions,** and metrics recomputed from
  them match the recorded ones exactly.

## Statistics and pre-specification

- **Analysis plan in version control before the data.** The plan
  ([`docs/analysis_plan.md`](docs/analysis_plan.md)) was committed on 17 August
  2026 (`d6e37db`). The confirmatory sweep started on 31 August. Every later
  experiment ran on a commit whose copy of the plan already contained its
  amendment ([`docs/RESULTS.md`](docs/RESULTS.md) §6.6). This is a pre-specified
  analysis plan, not a third-party registration. Amendments are dated, and those
  written retrospectively say so. The primary crossover was first seen in
  diagnostic data before the plan; the plan says this too.
- **Statistic.** A nested paired bootstrap that resamples test images and
  training seeds within each dataset × n cell (B = 2000, +1-corrected p-values).
  Datasets are weighted equally and treated as fixed. Benjamini–Hochberg is
  applied once across all 23 tests (`src/13_family_table.py`).
- **Verdict.** p_adj ≤ 0.05 **and** the predicted sign. Negligible means
  |ΔAUC| < 0.01.
- **Seeds.** 40 confirmatory seeds, with learning rates selected on 5 disjoint
  tuning seeds.

---

## Reproducing

Four levels, from seconds to GPU-days.

**0 — theory and code checks** (CPU, about a minute)

```bash
python -m pytest tests/test_independent_theory.py -q    # numpy only
python -m pytest tests -q                               # full suite: torch + PennyLane
```

**1 — every figure from the released results** (CPU, numpy + matplotlib)

The script first re-derives every confirmatory and exploratory row from the
bootstrap caches. It checks them against the tables to 1e-9 and stops before
drawing anything if any row disagrees.

```bash
python src/eval/generate_paper_plots.py --check-only
python src/eval/generate_paper_plots.py                 # writes paper/figures/
```

**2 — every statistic from the per-sample predictions** (CPU)

```bash
python src/13_family_table.py --selftest                # statistics core on synthetic data
python src/13_family_table.py --recompute               # the 23-test confirmatory table
python src/14_exploratory_table.py
python src/15_reporting_tables.py                       # absolute AUC, effect sizes, noise, provenance
```

**3 — retraining** (GPU)

```bash
python src/09_lr_selection.py                           # per-arm learning rates → artifacts/lr_selection.json
python src/01_frozen_backbone_ablation.py --confirmatory --use-tuned-lr --experiment 01_frozen_tuned
bash scripts/run_gap_closure.sh                         # H-S1, H-S2, H-S3, H-S4, E1
bash scripts/run_followup.sh                            # E7 and the H-S6 follow-up
bash scripts/run_capacity_followup.sh                   # the H-S5 follow-up
```

Every run writes one resumable shard keyed on its configuration and records the
git commit that produced it. Commit before launching anything long, and never
`git pull` during a sweep. Measured on one A100 MIG 3g.20gb slice, PathMNIST
included:
- 800 VQC runs: 9 h 23 min.
- 800 classical runs: 34 min.
- The 160-run E8 reproduction: 1.1 h.

On the same hardware and software, retraining is bit-exact. `17` retrained 160
confirmatory runs, and `16` retrained the function-class figure's run; every
recorded test AUC matched, with a difference of 0.0. Other GPUs may differ in the last digits.

**Released data.** Per-sample predictions, run records, bootstrap caches, latents
and reporting tables: *archive DOI to be added at submission*. Extract it into
`artifacts/` to run levels 1 and 2 without a GPU.

### Shard namespaces

| Namespace | Contents |
|---|---|
| `01_frozen_tuned` | primary comparison and the other arms at tuned rates, 40 seeds |
| `10_capacity_sweep`, `27_capacity_tuned` | H-S5 and its follow-up |
| `21_ansatz` | E1, ansatz check |
| `25_encoder` | H-S4, frozen vs adaptive encoder |
| `26_bottleneck_tuned` | H-S6 follow-up, frozen PCA and random projections |
| `28_latent_saturation` | E8 |

---

## Repository layout

```
src/
  config.py                     every constant, in one place
  shards.py                     resumable result I/O with provenance
  data/                         MedMNIST loader (stratified, GPU-resident), sensor-noise model
  models/                       backbone, heads, VQC, Fourier controls, registry
  train/                        shared training loop; metrics incl. calibration
  01_frozen_backbone_ablation   head comparison on cached features (frozen or adaptive encoder)
  03_robustness_evaluation      Gaussian sensor noise, σ ∈ [0, 0.20]
  04_statistical_analysis       per-cell breakdowns (exploratory)
  06_premise_check              does compression to d = 4 cost anything?
  07_hardware_noise             shot noise and depolarising noise (quantum only, feasibility)
  08_lipschitz                  head sensitivity
  09_lr_selection               per-arm learning rates on disjoint seeds
  10_capacity_sweep             low_rank across ranks (restriction)
  11_flow_verification          freezing and gradient-flow checks
  12_bottleneck_ablation        learned vs frozen PCA vs frozen random projection
  13_family_table               THE confirmatory table: 23 tests, one correction
  14_exploratory_table          exploratory analyses, corrected within each
  15_reporting_tables           absolute AUC, effect sizes, noise metrics, provenance
  16_reproduce_for_figure       retrains the function-class figure's run exactly
  17_latent_saturation          E8
  eval/generate_paper_plots.py  every figure, verified against the tables first
tests/                          parity, freezing, data pipeline, Fourier equivalence,
                                re-upload spectrum, independent numpy re-derivation
scripts/                        sweep runners, backup, maintenance and one-off patches
docs/
  RESULTS.md                    every number, by evidential tier
  analysis_plan.md              the binding plan and its dated amendments
  MATH_VERIFICATION.md          derivations and theory alignment
  PAPER_OUTLINE.md, AUDIT_REPORT.md, MASTER_RESEARCH_DOCUMENT.md, STATE.md, WORK_REMAINING.md
  HYPOTHESES.md                 historical (August 2026), superseded by the plan
paper/                          manuscript source, figures and tables
logs_*.txt                      console output of every run, kept for provenance
legacy/conference/              files from the conference version (see below)
artifacts/                      git-ignored: data cache, features, shards, predictions, caches
```

---

## Changes from the conference version

The conference paper was *Expressivity and Robustness of Hybrid Quantum Neural
Networks for Constrained Medical Image Classification*. The journal study re-ran
its comparisons with the controls above. Its own controls refuted several of its
claims, and they are withdrawn:

| Conference claim | What the controls show |
|---|---|
| The VQC maps the latent into a 16-dimensional complex Hilbert space and gains expressivity there; classical heads suffer a "Bottleneck Gap" | the head's output is a classical trigonometric function; compression to d = 4 costs almost nothing (finding 10) |
| Classical heads suffer "topological collapse" at d = 4 | against a parameter-matched head, the VQC is ahead only at n = 5, by +0.014 |
| "Latent Reshaping": quantum gradients reshape the encoder favourably | not supported (H-S4) |
| "Precision Paradox": the hybrid's F1 collapses under noise while AUC holds | the same happens to the classical control; a decision-threshold effect common to every small head |
| "Zombie State", exclusive to the hybrid and caused by tanh saturation | tanh·π/2 is applied to every arm here; the effect is not exclusive |
| Data abundance regularises the VQC | no VQC advantage at full data |

The protocol changed as well:
- **Learning rates:** per-arm tuned rates replaced fixed ones.
- **Checkpoint selection:** on validation AUC instead of training for a fixed
  number of epochs.
- **Seeds:** 40 instead of 3.
- **Scarcity:** absolute n per class instead of fractions.
- **Datasets:** four instead of two.
- **Baselines:** parameter-matched and function-class controls instead of Linear
  and MLP.

The conference figures and training logs are kept, unchanged, in
[`legacy/conference/`](legacy/conference/) for provenance. They are not results
of this study.

---

## Environment

Python 3.10 · torch 2.4.1+cu118 · PennyLane 0.42.3 · NumPy 1.26.4 · medmnist 3.0.2.
Simulator: `default.qubit`, `diff_method="backprop"` (agrees with adjoint to 3e-7).

```bash
conda create -n qml_v2 python=3.10 -y && conda activate qml_v2
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
python -m pytest tests -q
```

`requirements.lock.txt` and `environment.yml` record the full environment. The
cu118 wheels were needed because the training host's driver supports CUDA ≤ 11.4.
MedMNIST is downloaded automatically into `artifacts/` on first use.

## Limitations

- **Simulation only:** shot and depolarising noise are simulated; no hardware runs.
- **Scale:** 4–16 qubits.
- **Data:** 28×28 MedMNIST images, upsampled; four datasets.
- **Model:** one backbone.
- **Mechanism:** the cause of the n = 5 advantage is not identified beyond its
  need for a learned projection.

## References

- Schuld, Sweke & Meyer (2021). Effect of data encoding on the expressive power of variational quantum machine learning models. *Phys. Rev. A* 103, 032430.
- Pérez-Salinas et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum* 4, 226.
- Bowles, Ahmed & Schuld (2024). Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059.
- Yang et al. (2023). MedMNIST v2. *Scientific Data* 10, 41.

## Citation and licence

Citation details for the journal article will be added on publication. Licence:
see [`LICENSE`](LICENSE).
