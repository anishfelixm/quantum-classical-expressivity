# Hybrid Quantum-Classical Classification Under Extreme Latent Compression
## Master Research Document

**Version 1.0 — 8 August 2026**
**Status:** Design frozen. Implementation not started.
**Purpose:** Complete, self-contained specification. If all conversational context is lost, this document alone is sufficient to reconstruct the project.

---

# PART I — WHAT THIS RESEARCH IS

## 1.1 One-paragraph summary

A pre-trained ResNet-18 is truncated after its third residual block and its 256-dimensional pooled feature vector is compressed to a very small latent dimension `d ∈ {4, 8, 16}`. Different classification heads are then attached to that latent vector: linear, MLP, deep funnel, three trigonometric-feature heads, and a variational quantum circuit (VQC). The question is whether the VQC's particular way of expanding a low-dimensional vector into a high-dimensional function space confers a measurable advantage under (a) extreme compression, (b) severe data scarcity, and (c) analog input noise — and, critically, whether any such advantage survives comparison against classical baselines that occupy the *same function class*.

## 1.2 The history

The work began as a conference manuscript covering two datasets (BreastMNIST, PneumoniaMNIST), one bottleneck dimension (`d=4`), and three seeds. It reported three claimed contributions:

1. **"Latent Reshaping"** — quantum gradients backpropagating into the ResNet reorganize classical features into a quantum-friendly geometry.
2. **"The Precision Paradox"** — the VQC is superior on clean data but catastrophically fragile under Gaussian input noise.
3. **"Data abundance as a topological regularizer"** — the fragility vanishes when trained on the full dataset.

Supervisors recommended expansion to a journal (target: IEEE Access). Adversarial review of the conference manuscript and codebase identified that all three claims, as stated, were unsupported by the evidence presented. This document specifies the expanded work that would support them — or falsify them.

## 1.3 What was wrong with the original work

These are recorded because they define what the expansion must fix, and because a reviewer would find them.

| Problem | Consequence |
|---|---|
| No control for function class | The VQC's advantage could be entirely explained by "trigonometry beats linear algebra at low dimension" |
| No control for parameter count | VQC had 24–96 more trainable parameters than baselines |
| Regularization asymmetry | Classical heads got `weight_decay=1e-4`, quantum got `0.0`, then the paper concluded quantum generalizes better |
| Effect sizes inside test-set noise | Headline gap was 0.011 AUC on a 156-image test set (SE ≈ 0.03–0.04) |
| Wrong statistical test | Welch's t-test over seeds measures training variance only; blind to test sampling variance |
| No multiple-comparison correction | ~168 tests, ~8 expected false positives at α=0.05 |
| Mechanism claim with no mechanism measurement | "Latent Reshaping" inferred solely from downstream AUC; latent space never examined |
| Metaphor presented as mathematics | "Topological collapse", "Zombie State", "Glass Cannon" — no topology computed anywhere |
| Validation leakage | Trained on 54 images, selected checkpoints on 78 validation images across 100 epochs |
| Non-stratified scarcity sampling | Random subsets could omit entire classes in multi-class datasets |
| Incoherent motivation | Bottleneck justified by edge-device constraints; solution requires a quantum processor that cannot exist on an edge device |

**None of these invalidate the underlying engineering.** The pipeline works. The gradient path was verified correct. The problems are all in experimental control, statistics, and framing.

---

# PART II — THE MATHEMATICS

## 2.1 The shared pipeline

Every architecture shares an identical front end. This is what makes the comparison meaningful.

**Backbone.** Input image `x ∈ ℝ^{3×224×224}`. ResNet-18 pre-trained on ImageNet, truncated after `layer3` (discarding `layer4`, the global pool, and the FC head):

```
F(x) ∈ ℝ^{256×14×14}
h = AdaptiveAvgPool2d(1,1)(F(x)) ∈ ℝ^256
```

**Bottleneck.** A single dense projection:

```
z = W_c h + b_c ,    W_c ∈ ℝ^{d×256},  b_c ∈ ℝ^d
```

The compression ratio is `256:d` — 64:1 at `d=4`, 16:1 at `d=16`.

**Bounded rescaling.** Because rotation gates are `2π`-periodic, unbounded `z` would cause phase wrap-around (distinct classical values mapping to identical quantum states, destroying gradient information). We therefore apply:

```
z̃ = tanh(z) · (π/2)  ∈  [−π/2, π/2]^d
```

**This rescaling is applied identically to every head**, including all classical baselines. This is essential: if only the quantum arm received it, the comparison would confound the head with the input transform.

## 2.2 The quantum head

**State preparation.** `n = d` qubits initialized to `|0⟩^{⊗n}`, then angle embedding via Pauli-Y rotations:

```
RY(θ) = [[cos(θ/2), −sin(θ/2)], [sin(θ/2), cos(θ/2)]]

|ψ(z̃)⟩ = ⊗_{j=1}^{n} RY(z̃_j)|0⟩
        = ⊗_{j=1}^{n} [ cos(z̃_j/2)|0⟩ + sin(z̃_j/2)|1⟩ ]
```

Amplitude of computational basis state `|b⟩`, `b ∈ {0,1}^n`:

```
ψ_b(z̃) = ∏_{j=1}^{n} [cos(z̃_j/2)]^{1−b_j} · [sin(z̃_j/2)]^{b_j}
```

All amplitudes are real. The state lives in `ℂ^{2^n}` but occupies a real `2^n`-dimensional submanifold at this stage.

**Ansatz.** `L` repetitions of PennyLane's `StronglyEntanglingLayers`. Each layer applies an arbitrary single-qubit rotation to every wire followed by a cyclic CNOT cascade:

```
Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)
U(Θ) = ∏_{ℓ=1}^{L} [ CNOT-ring · ⊗_j Rot(Θ_{ℓ,j,1}, Θ_{ℓ,j,2}, Θ_{ℓ,j,3}) ]
```

Trainable quantum parameters: `Θ ∈ ℝ^{L×n×3}`. At `n=4, L=2` that is **24 parameters**.

**Measurement.**

```
v_i(z̃; Θ) = ⟨ψ(z̃)| U†(Θ) X_i U(Θ) |ψ(z̃)⟩  ∈ [−1, 1],   i = 1..n
```

**Classification.** `ŷ = W_o v + b_o`, `W_o ∈ ℝ^{C×n}`.

## 2.3 The dequantization result — the mathematical heart of this paper

This is the argument the entire experimental design is built to test. It must be stated precisely because it is what a physicist reviewer will check first.

Let `M(Θ) = U†(Θ) X_i U(Θ)`. This is a fixed Hermitian matrix that depends on `Θ` but **not** on the data. Then:

```
v_i(z̃) = Σ_{b, b'} ψ_b(z̃) · M_{bb'}(Θ) · ψ_{b'}(z̃)
```

Each product `ψ_b ψ_{b'}` factorizes over qubits into terms drawn from:

```
cos²(u/2) = (1 + cos u)/2
sin²(u/2) = (1 − cos u)/2
sin(u/2)cos(u/2) = sin(u)/2
```

Every such factor is an affine combination of `{1, cos u, sin u}`. Therefore:

> **Result.** For any `Θ`, the measured expectation value is exactly
> ```
> v_i(z̃) = Σ_{s ∈ {0,c,s}^n} c_s(Θ) · ∏_{j=1}^{n} f_{s_j}(z̃_j)
> ```
> where `f_0 = 1`, `f_c = cos`, `f_s = sin`. Equivalently, in exponential form, `v_i` is a Fourier series with frequency support `Ω = {−1, 0, +1}^n`.

**The basis has exactly `3^n` elements.** 81 at `n=4`; 6,561 at `n=8`; 43,046,721 at `n=16`.

### What this does and does not imply

**It does imply:** the VQC computes a function that lies in an explicitly known, classically constructible `3^n`-dimensional linear span. At `n=4` and `n=8` this span is small enough to construct exactly on a laptop. **No quantum advantage claim can survive at these scales.** Any paper claiming one will be rejected, correctly.

**It does not imply the VQC is useless.** The VQC does not span the full `3^n` space. With only `3Ln` parameters, it reaches a **low-dimensional nonlinear manifold**

```
ℳ_{L,n} = { c(Θ) : Θ ∈ ℝ^{L×n×3} }  ⊂  ℝ^{3^n}
```

At `n=4, L=2`: a 24-dimensional manifold inside an 81-dimensional space. At `n=16, L=4`: a 192-dimensional manifold inside a 43-million-dimensional space.

### The actual scientific question

> **Is `ℳ_{L,n}` a useful inductive bias?**

This is a real, open, empirically answerable question, and it is what this paper is about. Three competing possibilities:

- The constraint **helps** — it acts as a structured regularizer, preventing overfitting in the scarce-data regime where an unconstrained linear fit over `3^n` features would memorize.
- The constraint **hurts** — it is an arbitrary restriction, and a direct fit over the same basis does strictly better.
- The constraint is **irrelevant** — a parameter-matched generic nonlinearity does just as well, and neither trigonometry nor the manifold structure matters.

Each possibility maps to a specific baseline. That is why the arm roster looks the way it does.

## 2.4 The three control arms, and what each isolates

| Arm | Function class | Parameters | Isolates |
|---|---|---|---|
| **Fourier-Exact** | Full `3^n` span, linear fit | `3^n · C` | Ceiling of the VQC's function class |
| **Fourier-RFF** | `2m`-dim random subspace of the span | `2m · C` | Trigonometry without the manifold constraint |
| **Matched-Param MLP** | Generic `d→d` nonlinearity | `≈ 3Ln` | Capacity without trigonometry |
| **VQC** | Manifold `ℳ_{L,n}` | `3Ln` | The treatment |

**Fourier-Exact.** Features `φ(z̃) = ⊗_j [1, cos z̃_j, sin z̃_j] ∈ ℝ^{3^n}`, then a linear classifier. Feasible at `d=4` (81 features) and `d=8` (6,561 features). Infeasible at `d=16`. **This arm has far more parameters than the VQC by construction** — it is a class ceiling, not a fair fight, and must be labelled as such in the manuscript.

**Fourier-RFF.** Sample `m` frequency vectors `ω^{(1)}..ω^{(m)}` uniformly without replacement from `{−1,0,1}^n`, fixed at initialization by seed and stored as a non-trainable buffer. Features:

```
φ_RFF(z̃) = [ cos(ω^{(1)}·z̃), sin(ω^{(1)}·z̃), ..., cos(ω^{(m)}·z̃), sin(ω^{(m)}·z̃) ] ∈ ℝ^{2m}
```

This is the random-Fourier-features dequantization baseline from the theoretical literature. **This is the arm the pre-registered decision rule is evaluated against.** Frequencies are fixed, not learned — that is what makes it RFF rather than a learned feature map.

Budget: `2m = min(3^n, 2048)`. So `m=40` at `d=4` (81 features, essentially exact), `m=1024` at `d=8` and `d=16`. At `d=16` the RFF arm samples a vanishing fraction of the class — this is the honest RFF setting (RFF is by definition a Monte-Carlo kernel approximation) and must be stated.

**Matched-Param MLP.** `z̃ → Linear(d,d) → GELU → Linear(d,d) → classifier`, with hidden width chosen so total parameters ≈ `3Ln`. At `d=4, L=2`: target 24 parameters; `Linear(4,4)` with bias = 20. Controls capacity with no trigonometric structure.

**Why parameter-matching is the wrong axis for the Fourier arms.** Solving for parameter parity in an RFF head gives `8m + 4 = 24`, i.e. `m ≈ 2.5` — two or three frequencies. That would be a rigged comparison in the VQC's favour, because the VQC's *basis* (81 functions) is free, obtained from the embedding, and only its steering within that basis costs parameters. The RFF arm's basis is equally free. **Therefore the Fourier arms are matched on basis dimension, and the MLP arm is matched on parameter count.** Different arms, different parity axes, each stated explicitly.

## 2.5 Barren plateaus — measured, not assumed

Gradient variance of the quantum weights, measured on the project hardware (batch 32, `default.qubit`, backprop, GPU):

| n | L=1 | L=2 | L=4 |
|---|---|---|---|
| 4 | 1.045e+01 | 1.121e+01 | 1.296e+01 |
| 8 | 1.017e+01 | 2.588e+00 | 9.957e−01 |
| 16 | 4.289e+00 | 1.938e+00 | 2.098e−01 |

Variance decays monotonically in both `n` and `L`, by ~62× from (`n=4, L=4`) to (`n=16, L=4`). This is consistent with the onset of barren plateaus. It is **reportable as a measured result**, and it constrains interpretation: if the VQC underperforms at `d=16`, trainability is a candidate explanation distinct from function-class inadequacy.

## 2.6 Noise model

Additive white Gaussian noise applied in **physical pixel space**, not normalized space. This ordering is mandatory and was wrong in the conference version.

```
1. Inverse-normalize:  x_real = x · σ_ImageNet + μ_ImageNet          → [0,1]
2. Inject:             x_noisy = x_real + ε,   ε ~ 𝒩(0, σ²)
3. Clamp:              x_noisy = clip(x_noisy, 0, 1)                 → physical sensor bound
4. Re-normalize:       x' = (x_noisy − μ_ImageNet) / σ_ImageNet
```

RNG is seeded as `seed + int(round(σ·1000))` so every architecture faces bit-identical corrupted tensors. `round` (not truncation) avoids float representation collisions.

Sweep: `σ ∈ {0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20}`.

## 2.7 Statistics

**Primary test: nested paired bootstrap.** Two variance sources must both be captured — test-set sampling and training seed.

```
for b in 1..B (B = 2000):
    resample test indices I_b with replacement (size = |test|)
    resample seed indices S_b with replacement (size = n_seeds)
    Δ_b = mean_{s ∈ S_b} [ metric_A(s, I_b) − metric_B(s, I_b) ]
CI_95 = percentiles(Δ, 2.5, 97.5)
p_boot = 2 · min( P(Δ ≤ 0), P(Δ ≥ 0) )
```

Paired: identical resample indices for both models. This is what makes it sensitive to a small consistent difference while remaining honest about a 156-image test set.

**Why not Welch's t-test over seeds.** It captures only training variance. With `n_test = 156`, the Hanley–McNeil standard error on AUC ≈ 0.03–0.04, while the effect being claimed is 0.011. A seed-level test can report `p < 0.05` on a difference that a different draw of test images would reverse.

**Secondary: DeLong's test.** Applies only to binary AUC — BreastMNIST and PneumoniaMNIST. **Undefined for BloodMNIST and PathMNIST.** Confirmation only, never a substitute.

**Multiple comparisons: Benjamini–Hochberg FDR.** Sort `p_(1) ≤ ... ≤ p_(M)`, find the largest `k` with `p_(k) ≤ (k/M)·α`, reject `H_(1)..H_(k)`. The family is declared in advance in `docs/analysis_plan.md`. Report raw and adjusted p-values side by side.

**Effect size: Cohen's d** across seeds, reported alongside every p-value. Where a significant difference is practically negligible, say so explicitly in the text.

---

# PART III — EXPERIMENTAL DESIGN

## 3.1 The scarcity axis

**Absolute shots per class**, not percentages. `n_per_class ∈ {5, 10, 20, 50, 100}` plus a full-dataset reference row.

**Rationale.** The scientific variable is how many labelled examples the model sees. Percentages are non-comparable across datasets: 1% is 5 images on BreastMNIST and ~900 on PathMNIST, so a single row of a scaling-law figure would mix two unrelated experiments. Absolute counts also match the convention in the medical few-shot literature, making results directly comparable to published baselines.

Sampling is **stratified per class**. Validation is subsampled to the same `n_per_class` and seed. Test sets remain full.

## 3.2 The arm roster

| # | Arm | Role |
|---|---|---|
| 1 | PCA + SVM | Non-neural reference (excluded from primary test family) |
| 2 | Classical Linear | Minimum-capacity floor |
| 3 | Classical MLP (GELU) | Non-linearity at zero extra parameters |
| 4 | Classical Deep Funnel | Proves failure is not a depth problem |
| 5 | Matched-Param MLP | Capacity control |
| 6 | Fourier-RFF | **Function-class control — the primary comparison** |
| 7 | Fourier-Exact | Function-class ceiling (`d=4,8` only) |
| 8 | Quantum VQC | Treatment |

## 3.3 The matrix

| Axis | Values |
|---|---|
| Datasets | BreastMNIST, PneumoniaMNIST (binary); BloodMNIST, PathMNIST (multi-class) |
| Scarcity | `n_per_class ∈ {5, 10, 20, 50, 100}` + full reference |
| Bottleneck | `d ∈ {4, 8, 16}` |
| Seeds | 10 |
| Arms | 8 |
| VQC depth | `L ∈ {1,2,4}` in pilot; best carried forward |

## 3.4 The three experiments

**Experiment 1 — Frozen-backbone ablation.** Entire ResNet immobilized. Isolates head expressivity on static ImageNet features. **Features are cached**: because the backbone is frozen and in eval mode, `h = pool(backbone(x))` is deterministic, so it is extracted once per (dataset, seed) and all heads train on cached vectors. Mathematically identical, orders of magnitude faster.

**Experiment 2 — End-to-end fine-tuning.** `layer3` unfrozen; all other backbone blocks frozen with BatchNorm in eval mode. Differential learning rates. Tests whether head gradients reshape the classical feature extractor.

**Experiment 3 — Robustness.** Experiment 2 checkpoints evaluated across the noise sweep. **Fourier-RFF and Matched-Param arms are included here**, not just in the clean evaluation — differential fragility against a function-class-matched control is the only part of this that is novel.

**Experiment 4 — Latent probe.** Extract `z` on the test set, freeze it, then: train a linear probe; compute Fisher discriminant ratio and silhouette score; compare geometry before vs after `layer3` unfreezing. **This converts "Latent Reshaping" from an interpretation into a measurement.**

**Experiment 5 — Premise check.** Unbottlenecked ResNet-18 (`d=256`) on all four datasets. Gates everything: if `d=4` matches `d=256`, there is no compression penalty to bypass and the paper is about scarcity only.

**Experiment 6 — Shot noise.** Final models re-evaluated with finite measurement shots (1024) instead of exact expectation values. Evaluation only, no retraining.

## 3.5 The pilot

BreastMNIST + PneumoniaMNIST, `d=4`, `n_per_class ∈ {20, 100}`, 10 seeds, all 8 arms, both weight-decay settings, `L ∈ {1,2,4}`. Roughly 1–2 days.

**The pilot is a gate.** Its outcome, evaluated against the pre-registered rule, fixes the manuscript's framing. It is not relitigated afterward.

## 3.6 Training protocol

- Loss: `CrossEntropyLoss` with inverse-frequency class weights, clipped to `[0.1, 10]`. Applied identically to every arm.
- Optimizer: Adam. `η_backbone = 1e-4`, `η_head = 1e-3`, `η_quantum = 1e-3`. Small LR grid per arm on validation for one dataset; grid and selection reported.
- Weight decay: pilot runs both `0.0` for all arms and `1e-4` for all arms. Never asymmetric.
- Model selection: best validation Macro-F1, strict `>` (not `>=`). Selected epoch logged.
- Early stopping: patience 30, `min_epochs = min(max(20, 200 // len(train_loader)), max_epochs // 2)`.
- Max epochs: 100.
- Gradient clipping: `max_norm=1.0`, with **pre-clip gradient norms logged per arm per epoch** to verify clipping is not differentially binding.
- Augmentation: horizontal flip + small rotation, identical across all arms.
- Threshold: argmax throughout. No locked or tuned thresholds. Report Macro-F1 and balanced accuracy.

## 3.7 Hardware and environment

**Verified working configuration:**

```
GPU:            NVIDIA A100-SXM4-40GB, MIG 3g.20gb slice (20GB, 42/108 SMs)
Host driver:    470.199.02  →  CUDA ceiling 11.4  →  CUDA 12 unreachable
pennylane       0.42.3   (PIN — do not upgrade)
pennylane_lightning 0.42.0
numpy           1.26.4
torch           1.12.1+cu113  →  UPGRADE to a cu118 build (see below)
```

**Simulator decision: `default.qubit` with `diff_method="backprop"`, on GPU, for every `d`.** Measured step times (batch 32):

| d | backprop/GPU | adjoint/GPU | lightning/CPU |
|---|---|---|---|
| 4 | 0.026 s | 1.79 s | 4.07 s |
| 8 | 0.054 s | 5.32 s | 7.57 s |
| 16 | 0.357 s | >700 s | 93.1 s |

Backprop on GPU is 260× faster than the previously-used configuration at `d=16`. Peak memory 6.5 GB at `d=16, L=4`, against a 20 GB slice.

**Verified:** adjoint and backprop input-gradients agree to ~5e−7 relative. Gradients reach `layer3`. The original pipeline was genuinely end-to-end — an earlier concern that it was silently frozen proved unfounded.

**torch upgrade.** `torch.load(weights_only=True)` in the robustness script does not exist in 1.12 and would crash. Upgrade to a `cu118` build (CUDA 11.x minor-version compatibility works on driver 470; CUDA 12 does not). Then pin exactly.

---

# PART IV — FILE SPECIFICATION

```
quantum-classical-expressivity/
├── requirements.txt              exact pins, no floors
├── environment.yml               conda spec, python 3.10
├── README.md
├── docs/
│   ├── analysis_plan.md          PRE-REGISTRATION — committed before pilot
│   └── MASTER_RESEARCH_DOCUMENT.md   this file
├── src/
│   ├── config.py
│   ├── data/medmnist_loader.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── heads.py
│   │   ├── classical_fourier.py
│   │   ├── quantum_vqc.py
│   │   └── registry.py
│   ├── train/
│   │   ├── loop.py
│   │   └── metrics.py
│   ├── 01_frozen_backbone_ablation.py
│   ├── 02_end_to_end_finetuning.py
│   ├── 03_robustness_evaluation.py
│   ├── 04_statistical_analysis.py
│   ├── 05_latent_analysis.py
│   ├── 06_premise_check.py
│   ├── 07_shot_noise.py
│   ├── merge_results.py
│   └── eval/generate_plots.py
├── tests/
│   ├── test_gradient_flow.py
│   ├── test_fourier_equivalence.py
│   ├── test_loader_stratification.py
│   └── test_noise_roundtrip.py
└── scripts/run_sweep.sh
```

## 4.1 `src/config.py` — NEW

Single source of truth. Every constant lives here; no script defines its own. Datasets, `n_per_class` grid, bottleneck dims, seed list, arm registry keys, noise levels, LR values, epoch/patience settings, paths, device and diff-method selection. Scripts import from here and never hardcode.

## 4.2 `src/data/medmnist_loader.py` — REWRITE

**Responsibilities.**
- Load any MedMNIST dataset; detect 1- vs 3-channel natively.
- Transform order: `ToTensor → Resize(224, antialias) → repeat-to-3ch if grayscale → augment (train only) → Normalize`. Resize precedes channel repeat (avoids 3× interpolation cost).
- **Stratified sampling by `n_per_class`**, not fraction. Assert every class present; return realized per-class counts for logging.
- **Subsample validation to the same `n_per_class` and seed.**
- Guard against a final batch of size 1 (BatchNorm crash) by adjusting subset size by ±1.
- Explicit `torch.Generator(seed)` on every DataLoader.
- Export `NORM_MEAN`, `NORM_STD` for the noise module.

**Must not:** use fractions; use `np.random.choice` over the flat index range; leave validation at full size.

## 4.3 `src/models/backbone.py` — NEW

Single shared truncated-ResNet definition, replacing the copy-pasted duplication across the three current model files. Constructs ResNet-18, truncates after `layer3`, applies `AdaptiveAvgPool2d`, exposes a freezing policy (`frozen` vs `layer3_only`) and a `forward` returning `h ∈ ℝ^256`. Also provides `set_bn_eval()` to keep frozen BatchNorm layers from updating running statistics during fine-tuning.

**Why:** guarantees every arm has a bit-identical feature extractor. Duplication is how parity silently breaks.

## 4.4 `src/models/heads.py` — NEW

All classical heads, each taking `z̃` and returning logits:

- `LinearHead` — `Linear(d, C)`
- `MLPHead` — `GELU → Linear(d, C)`
- `DeepFunnelHead` — `256→64→16→d` with GELU and **LayerNorm** (not BatchNorm: batch-size independent, stable at `n=5`, and does not weaken the deep stack this arm exists to strengthen)
- `MatchedParamHead` — `Linear(d,d) → GELU → Linear(d,C)`, width solved for `≈3Ln` parameters

All use `kaiming_normal_(mode='fan_in')` for initialization parity.

## 4.5 `src/models/classical_fourier.py` — NEW

`FourierExactHead(d, C)`
- Builds `φ(z̃) = ⊗_j [1, cos z̃_j, sin z̃_j] ∈ ℝ^{3^d}` via iterative Kronecker product.
- Raises if `d > 8`.
- `Linear(3^d, C)`.

`FourierRFFHead(d, C, m, seed)`
- Samples `m` frequency vectors uniformly without replacement from `{−1,0,1}^d`; stores as **non-trainable buffer** (so they persist in checkpoints and are reproducible from seed).
- `φ_RFF(z̃) = [cos(Ωz̃), sin(Ωz̃)] ∈ ℝ^{2m}`.
- `Linear(2m, C)`.
- Default `2m = min(3^d, 2048)`.

Both consume `z̃`, never raw `z`.

## 4.6 `src/models/quantum_vqc.py` — REWRITE

- Configurable `n_qubits`, `n_layers`, `device_name`, `diff_method` (defaults: `default.qubit`, `backprop`).
- `AngleEmbedding(rotation='Y')` → `StronglyEntanglingLayers` → `[expval(PauliX(i))]`.
- Weight init `𝒩(0, 0.1)` — small variance to delay barren plateaus.
- `Linear(n, C)` output head.
- Exposes `quantum_parameters()` and `classical_parameters()` for clean optimizer grouping.
- Exposes `grad_variance()` for the barren-plateau measurement.

**Removes:** the hardcoded `diff_method="adjoint"`, the duplicated backbone code.

## 4.7 `src/models/registry.py` — NEW

`build_arm(name, d, C, L, seed) → nn.Module`. Single factory so every experiment script constructs arms identically. Prevents drift where one script instantiates a head differently from another.

## 4.8 `src/train/loop.py` — NEW

One training loop, shared by Experiments 1 and 2, parameterized by freezing policy. Contains: differential optimizer grouping, class weighting with clipping, BatchNorm eval locking, clipped gradient step with **pre-clip norm logging**, per-epoch train and validation metrics, `>`-strict checkpointing with epoch logging, clamped early stopping, and trainable-parameters-only checkpoint saving.

**Why one loop:** the current codebase duplicates ~150 lines between scripts 1 and 2, which is where the `>=` bug and the `min_epochs` bug diverged.

## 4.9 `src/train/metrics.py` — NEW

Accuracy, balanced accuracy, Macro-F1, ROC-AUC with explicit `average=` and `multi_class='ovr'`, expected calibration error, probability-distribution standard deviation. NaN-safe, RFC-8259-compliant JSON serialization.

## 4.10 Experiment scripts `01`–`07`

All follow the same shape: read `config.py`, accept a single `(dataset, n_per_class, d, seed, arm)` tuple via CLI, write **one result shard** to `results/shards/`, and exit. No monolithic nested loops, no single JSON rewritten at the end.

- `01` — frozen ablation, **with feature caching**
- `02` — end-to-end fine-tuning
- `03` — robustness sweep over `σ`, loading `02` checkpoints, reporting **F1, AUC, and calibration** at every `σ`
- `04` — nested paired bootstrap, DeLong (binary only), BH-FDR, Cohen's d, LaTeX table emission
- `05` — latent probe: extract `z` to `.npy`, linear probe, Fisher ratio, silhouette
- `06` — premise check at `d=256`
- `07` — finite-shot re-evaluation

## 4.11 `src/merge_results.py` — NEW

Collects shards into a single analysis-ready structure. Reports which cells are missing so the sweep can be resumed.

## 4.12 `src/eval/generate_plots.py` — NEW

Five figures: scarcity scaling law; train-vs-validation overfitting matrix; noise decay curves; bottleneck expressivity bars; cross-dataset heatmap. Plus UMAP panels, **captioned as illustration** — the mechanism claim rests on `05`, not on UMAP geometry.

## 4.13 `tests/test_fourier_equivalence.py` — NEW, and important

**This test empirically proves the dequantization claim inside your own codebase.**

For random `Θ` and random `z̃` samples: evaluate the VQC, evaluate the exact `3^n` basis, solve least squares for coefficients, and assert the residual is at machine precision. If it passes, you can state in the manuscript — with a reproducible test as evidence — that the VQC's output lies exactly in the constructed classical span. That is a strong appendix and it preempts the single most dangerous reviewer objection.

## 4.14 Other tests

- `test_gradient_flow.py` — asserts `z.grad` is non-zero and reaches `layer3`; cross-checks diff methods.
- `test_loader_stratification.py` — asserts every class present at every `n_per_class`, and that validation is subsampled.
- `test_noise_roundtrip.py` — asserts `σ=0` is an exact identity through the four-step normalize/denormalize path.

---

# PART V — IS THIS PUBLISHABLE?

## 5.1 The honest ceiling

**This work cannot demonstrate quantum advantage.** At 4–16 qubits on a state-vector simulator, the model is classically simulable by construction — that is literally how it is being run. Any manuscript claiming advantage will be rejected, and correctly so.

What it can demonstrate is whether a specific quantum-derived inductive bias is useful. That is a real contribution, and a modest one.

## 5.2 Competitive landscape

The literature search found the field is crowded, and two of the original contributions have direct precedents:

- A 2026 hybrid blood-cell paper uses a pre-trained ResNet backbone, a low-dimensional latent bottleneck, a VQC head, **and a capacity-matched classical control**. That is the original architecture and one of the planned controls, already published.
- A BreastMNIST hybrid QCNN paper already does parameter matching, multi-seed runs, Wilcoxon tests and Cohen's d. **Statistical rigor is now table stakes, not a contribution.**
- A 2025 paper already reports Gaussian-noise robustness comparisons on MedMNIST finding classical models more robust than quantum. **The "Precision Paradox" as originally framed is published.**

## 5.3 What remains genuinely novel

1. **First empirical application of the RFF/classical-surrogate dequantization control to a hybrid CNN+VQC medical imaging pipeline.** The theory is mature (necessary and sufficient conditions for RFF dequantization of variational QML, extended to classification), but it has only been tested on synthetic and small problems. This is the headline.
2. **Resolution of a published contradiction.** One paper reports VQCs as more noise-fragile; another reports them as more robust. Nobody has explained why. Data scarcity is a plausible resolving variable, and the conference results already hint at it. This reframes contribution 2 from "we found fragility" (published) to "we identify the condition under which the contradiction resolves."
3. **A shots-per-class scaling law across four modalities and three bottleneck dimensions**, with measured gradient-variance decay.

"Latent Reshaping" drops from headline to supporting result.

## 5.4 A citable framing gift

A survey of QML in medicine already levels the exact critique this paper would face: that noise-free simulation produces over-optimistic estimates, and that reported 0.6–4% improvements should be treated as tentative proof-of-concept rather than evidence of advantage. **Cite it in the introduction and position the paper as responding to it.** This converts the work's greatest vulnerability into its stated motivation.

## 5.5 Assessment

Both pilot outcomes yield a publishable paper:

- **Outcome A** (VQC > Fourier-RFF): an advantage-of-inductive-bias result with unusually strong controls.
- **Outcome B** (VQC ≈ Fourier-RFF): a clean dequantization result — arguably the more citable of the two, given how many unfalsified advantage claims the field currently contains.

The real risk is not rejection. It is producing a result that is true but trivial. The controls are what prevent that, and they are also what makes either outcome worth reading.

**No methodology guarantees acceptance.** Where "scientifically right" and "maximizes publication odds" diverge, the divergence gets flagged rather than silently resolved in favour of the second.

---

# PART VI — EXECUTION SEQUENCE

| Stage | Work | Gate |
|---|---|---|
| S0 | ✅ Gradient verification; hardware benchmark; GPU repair | **Complete** |
| S1 | torch upgrade to cu118; exact pinning | — |
| S2 | `backbone.py`, `heads.py`, `classical_fourier.py`, `quantum_vqc.py`, `registry.py` | `test_fourier_equivalence` must pass |
| S3 | `medmnist_loader.py`, `config.py`, `train/` | `test_loader_stratification` must pass |
| S4 | `06_premise_check.py`, run it | **GATE: does `d=4` differ from `d=256`?** If not, reframe |
| S5 | Commit `docs/analysis_plan.md` | **GATE: pre-registration timestamped before any pilot data exists** |
| S6 | Experiment scripts, sharding, caching, `run_sweep.sh` | — |
| S7 | **Pilot** | **GATE: framing fixed by pre-registered rule** |
| S8 | Full sweep | — |
| S9 | Experiments 3–7 | — |
| S10 | Statistics, figures | — |
| S11 | Manuscript — rebuilt from scratch, not edited | — |
| S12 | Supervisor review, submission | — |

---

# PART VII — OPEN QUESTIONS

## 7.1 Requiring a decision

| # | Question | Recommendation |
|---|---|---|
| 1 | Split-computing framing for the bottleneck motivation? | **Yes.** Edge device transmits a `d`-dim latent; VQC lives server-side. The "quantum computer on a portable ultrasound" framing is incoherent and would be caught. |
| 2 | Approve the pre-registered decision rule as drafted? | Approve, with the addition that the gate is evaluated on pilot datasets only and the full-sweep result is reported regardless of agreement |
| 3 | Corresponding author; supervisor review before submission? | Unresolved |
| 4 | MedMNIST per-dataset licensing and citation requirements | Needed for the IEEE Access data availability statement |

**Draft decision rule.** The VQC is declared to outperform Fourier-RFF only if, at matched basis dimension and matched weight decay: (a) the 95% nested-bootstrap CI on ΔAUC excludes zero; (b) after BH-FDR correction across the declared family; (c) in ≥3 of 4 datasets and ≥2 of 5 scarcity regimes. Otherwise the manuscript is Outcome B.

## 7.2 Genuinely open, empirically

**7.2.1 Does the bottleneck bottleneck?** If `d=4` matches `d=256` on clean data, "topological collapse" does not exist and the paper is about scarcity alone. Gated at S4. Currently unknown, and the conference results suggest it may be a real risk — classical Linear *won* at 100% data.

**7.2.2 Is the noise collapse boundary failure or calibration failure?** The "Zombie State" observation (AUC 0.6118 with probability std 0.0057) suggests the F1 collapse is threshold drift, not decision-boundary degradation. If AUC holds while F1 craters, contribution 2 becomes narrower and more technical. Resolved by reporting both metrics plus ECE at every `σ`.

**7.2.3 Is gradient clipping parity-neutral?** Measured `layer3` gradient norm ≈ 665 against `max_norm=1.0` — a 665× rescale — while the quantum arm's natural scale is ≈ 6.4. If clipping binds differentially across arms, the "quantum trains more stably" claim is partly an artifact of the threshold. Pre-clip norms must be logged.

**7.2.4 Does the VQC exhaust its own function class?** Fourier-Exact answers this. If a direct linear fit over the identical `3^n` basis beats the variational optimizer, that is a strong standalone dequantization result about optimization rather than expressivity.

**7.2.5 Is `d=16` trainable at all?** Gradient variance falls ~62× from `d=4`. Compute is no longer the constraint, but trainability may be. If the VQC fails at `d=16`, distinguish trainability failure from expressivity failure using the measured variance.

**7.2.6 Does 10 seeds suffice at `n_per_class=5`?** Variance at 5 shots per class will be extreme. Monitor in the pilot; increase if bootstrap CIs are uninformatively wide.

**7.2.7 Frequency sampling for RFF.** Uniform over `{−1,0,1}^d` is used, for reproducibility. A low-Hamming-weight bias would better match where a shallow circuit concentrates its spectrum, and might be a fairer or an unfairer baseline. Worth one sensitivity check in the pilot.

## 7.3 Deferred

- Persistent homology: **not** computed. Estimating persistent homology from 5–50 points per class in 4–16 dimensions produces noise. The vocabulary is retired instead; "geometric separability" replaces "topological".
- Real QPU deployment: out of scope. The shot-noise ablation is the closest approximation.
- Data re-uploading: would enlarge the frequency spectrum beyond `{−1,0,1}^d` and change the dequantization analysis substantially. Out of scope for this paper; a natural follow-up.

---

# PART VIII — RECORDED DISAGREEMENTS

Kept so decisions are made deliberately rather than by whoever spoke most recently.

| # | Point | Position A | Position B | Resolution |
|---|---|---|---|---|
| 1 | PathMNIST 1% class coverage | "1% drops 4 of 9 classes" | Factually wrong — 1% ≈ 900 samples, stratifies fine. The degenerate case is BreastMNIST (n≈5) | B |
| 2 | Deep Funnel normalization | Strip BatchNorm | Stripping weakens the baseline this arm exists to strengthen; use LayerNorm | B |
| 3 | AUC significance test | DeLong's test | DeLong is binary-only; nested paired bootstrap primary, DeLong secondary | B |
| 4 | Latent Reshaping evidence | UMAP plot | UMAP is illustration; linear probe + Fisher ratio is evidence. Keep the figure, move the claim | B |
| 5 | Job orchestration | Simple sequential bash script | Concern was output granularity, not orchestration — sequential driver + per-config shards + resume | Merged |
| 6 | Adjoint gradient flow | (Claude's suspicion) adjoint silently drops input gradients | **Falsified by measurement.** Agreement to ~5e−7 | Claude wrong |
| 7 | torch target version | cu121+ | Driver 470 caps at CUDA 11.4; cu118 is the ceiling | Corrected |
| 8 | Fourier arm parity axis | Match parameters | Parameter matching gives m≈2.5 frequencies — rigged. Match basis dimension instead | Corrected |

---

**End of document.**
