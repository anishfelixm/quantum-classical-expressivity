# Quantum–Classical Expressivity Under Extreme Latent Compression

Controlled comparison of variational quantum and classical classification heads operating
on a severely compressed latent vector, under data scarcity and analog sensor noise.

> **Scope.** At 4–16 qubits on a state-vector simulator this work cannot demonstrate
> quantum advantage — the model is classically simulable by construction, which is how it
> is run. The question asked is narrower and answerable: *at a fixed parameter budget, is a
> quantum-derived inductive bias useful?*

---

## The setup

A ResNet-18 truncated after `layer3` produces a 256-d pooled feature vector, compressed by
a single dense projection to `d ∈ {4, 8, 16}` and bounded by `z̃ = tanh(z)·(π/2)`.
Interchangeable classification heads then consume the identical `z̃`.

```
image → ResNet-18 (→layer3) → pool → 256-d → bottleneck → z → tanh·(π/2) → z̃ → HEAD → logits
```

Holding everything before the head fixed is what makes the comparison a comparison.

---

## Arms

| Arm | Role | Head params (d=4, C=2) |
|---|---|---|
| `linear` | minimum-capacity floor | 0 |
| `mlp` | non-linearity at zero extra parameters | 0 |
| `deep_funnel` | shows failure is not a depth problem | — |
| `matched_param` | **capacity control** — parameter-matched to the VQC | 24 |
| `fourier_rff` | **function-class control** — dequantization baseline | 324 |
| `fourier_exact` | function-class ceiling (`d ≤ 8` only) | 328 |
| `quantum_vqc` | treatment | 24 |
| `pca_svm` | non-neural reference, excluded from the test family | — |

Two distinct parity axes, deliberately:

- `matched_param` matches **parameter count** → answers *is the quantum head more
  efficient per parameter?*
- `fourier_rff` matches **basis dimension** → answers *is any advantage quantum, or just
  trigonometric?*

Matching the Fourier arm on parameters instead would allow only ~2 frequencies, which
would rig the comparison. See `docs/MATH_VERIFICATION.md` §6.

---

## The central mathematical result

With `AngleEmbedding(rotation='Y')` and no data re-uploading, every measured expectation
value is exactly

```
v_i(z̃) = Σ_{s ∈ {0,c,s}^n} c_s(Θ) ∏_j f_{s_j}(z̃_j),    f_0 = 1, f_c = cos, f_s = sin
```

— a `3^n`-dimensional classical trigonometric span, matching the frequency spectrum
predicted by Schuld, Sweke & Meyer (2021) from the eigenvalue differences of the encoding
generator.

The VQC does **not** span this space. With `3Ln` parameters it reaches a low-dimensional
manifold inside it: 24 dimensions within 81 at `n=4, L=2`. Whether that constraint is a
useful inductive bias is the paper's question.

Verified numerically at machine precision:

```bash
python -m pytest tests/test_fourier_equivalence.py -v -s
# residuals 6e-16 to 2e-15 across six (d,L) configurations
# wrong-frequency negative control fails at 0.908, as required
```

Full derivation and theory alignment: [`docs/MATH_VERIFICATION.md`](docs/MATH_VERIFICATION.md).

---

## Experiments

| Script | Purpose |
|---|---|
| `src/01_frozen_backbone_ablation.py` | Head expressivity, frozen vs adaptive encoder. Feature-cached. |
| `src/02_end_to_end_finetuning.py` | End-to-end fine-tuning through `layer3` |
| `src/03_robustness_evaluation.py` | AWGN sensor-noise sweep, `σ ∈ [0, 0.20]` |
| `src/04_statistical_analysis.py` | Nested paired bootstrap, BH-FDR, effect sizes |
| `src/05_latent_analysis.py` | Latent probe — linear separability, Fisher ratio |
| `src/06_premise_check.py` | Does the bottleneck actually cost anything? |
| `src/07_shot_noise.py` | Finite-shot re-evaluation |

Every script takes a single configuration and writes one result **shard**, so an
interrupted sweep resumes by re-running the same command. Each shard records the git commit
that produced it.

```bash
python src/06_premise_check.py                       # gates the framing
python src/01_frozen_backbone_ablation.py --diagnostic
python src/01_frozen_backbone_ablation.py --summary-only
```

---

## Design decisions that matter

**Scarcity is absolute, not fractional.** `n_per_class ∈ {5,10,20,50,100}`. Percentages are
non-comparable across datasets — 1% is 5 images on BreastMNIST and ~900 on PathMNIST, so a
single row of a scaling curve would mix unrelated experiments.

**Validation is subsampled to match.** Selecting the best of 100 epochs on a full
validation split while training on 54 images means model selection consumed more labels
than training did. Val is capped at `min(2·n_per_class, available)` per class, and the
val/train ratio is logged.

**Sampling is stratified.** Random subsets can drop entire classes on 8- and 9-class
datasets, varying by seed.

**Noise is injected in physical pixel space.** Inverse-normalize → inject → clamp to
`[0,1]` → re-normalize. Clamping in *normalized* coordinates, where `[0,1]` is not a
physical bound, destroys signal instead of modelling a sensor.

**Regularization parity is absolute.** Weight decay, learning rate and gradient clipping
are identical across arms, or the run does not happen. `GRAD_CLIP_NORM = 20.0` was chosen
as 2× the largest observed p95 gradient norm so that clipping never binds during normal
training — at the previous value of 1.0 it bound on classical arms while never touching the
quantum arm, a per-arm learning-rate multiplier disguised as a safety net.

**Augmentation is off for the frozen/adaptive comparison.** Feature caching requires
deterministic features; if only the adaptive side augmented, freezing and augmentation
would vary together and the result would be uninterpretable.

---

## Setup

Verified configuration — host driver 470.199.02 caps CUDA at 11.4, so CUDA 12 builds will
not run.

```bash
conda create -n qml_v2 python=3.10 -y && conda activate qml_v2
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
python -m pytest tests/ -q      # 21 tests
```

Simulator: `default.qubit` with `diff_method="backprop"` at every dimension —
0.026 s/step at d=4, 0.357 s/step at d=16 (batch 32, A100 MIG 3g.20gb), 260× faster than
adjoint at d=16.

Artifacts (datasets, checkpoints, shards, latents) live outside the repository via the
`artifacts/` symlink and are never committed.

---

## Repository layout

```
src/
  config.py                  single source of truth for every constant
  data/                      GPU-resident loader, AWGN noise model
  models/                    backbone, heads, Fourier arms, VQC, registry
  train/                     shared training loop, metrics + calibration
  0*.py                      experiments
  shards.py                  resumable result I/O with provenance
tests/                       gradient flow, Fourier equivalence, data pipeline
docs/
  MATH_VERIFICATION.md       derivations and theory alignment
  HYPOTHESES.md              every hypothesis, its test, and its status
  analysis_plan.md           pre-registration (committed before confirmatory runs)
```

---

## Reproducibility

Exact-pinned dependencies; per-run seeding of Python, NumPy and torch RNGs with cuDNN
deterministic kernels; explicit DataLoader generators; and a git SHA recorded in every
result shard.

---

## References

- Schuld, Sweke & Meyer (2021). *Effect of data encoding on the expressive power of
  variational quantum machine learning models.* Phys. Rev. A **103**, 032430.
- Pérez-Salinas et al. (2020). *Data re-uploading for a universal quantum classifier.*
  Quantum **4**, 226.
- McClean et al. (2018). *Barren plateaus in quantum neural network training landscapes.*
  Nat. Commun. **9**, 4812.
- Yang et al. (2023). *MedMNIST v2.* Scientific Data **10**, 41.

## License

See `LICENSE`.
