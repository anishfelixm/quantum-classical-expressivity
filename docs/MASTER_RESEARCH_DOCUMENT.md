# Master Research Document

**Project:** Quantum–Classical Expressivity Under Extreme Latent Compression
**Repository:** `github.com/anishfelixm/quantum-classical-expressivity`, branch `feature/journal-expansion`
**Version 2.0 — 19 August 2026.** Supersedes v1.0 (8 Aug), which described a design that has since changed substantially.
**Target venue:** IEEE Access. APC covered by the university.

> **Read this first in any new session.** Then `docs/analysis_plan.md` (the
> pre-registration — binding) and `docs/MATH_VERIFICATION.md` (the derivations).
> Everything else is detail.

---

## 1. The question

> At a fixed small parameter budget and with very few labelled medical images,
> does a variational quantum head extract more from a compressed feature vector
> than a classical head — and if it does, is the advantage quantum, or merely
> trigonometric?

### The thesis as currently supported

The quantum head's advantage, where it exists, is a **regularization effect of
its restricted function class** — 24 parameters reaching a 24-dimensional
manifold inside an 81-dimensional trigonometric span — **not** a quantum
computational advantage. It is confined to the extreme-scarcity regime and
reverses once data is sufficient.

**This is a characterization paper, not a "quantum wins" paper.** If anyone
involved expects the latter, reset that expectation now rather than at
submission.

### Working title

> Quantum or Trigonometric? A Dequantization-Controlled Comparison of
> Variational Quantum and Classical Classification Heads for Medical Imaging
> under Data Scarcity and Sensor Noise

Fallbacks — if the crossover confirms: *"Restriction, Not Advantage: Why a
Variational Quantum Head Helps Only in the Few-Shot Limit."* If it does not:
*"A Matched-Parameter Evaluation of Variational Quantum Classification Heads for
Medical Imaging: No Advantage at Equal Capacity."*

### What the conference paper claimed, and what survived

| Conference claim | Status now |
|---|---|
| Topological collapse at d=4 | ❌ Refuted — compression costs ~0.002 AUC |
| The "Bottleneck Gap" | ❌ Refuted — no gap to bypass |
| Latent Reshaping | ⚠️ True, but **every** head does it, not just quantum |
| Precision Paradox | ⏳ Untested in the new pipeline (Q5) |
| Data abundance as regularizer | 🔄 Inverted — the *restricted function class* regularizes |
| Edge-computing motivation | ❌ Dropped — no QPU on a portable scanner |

---

## 2. Architecture

Matches the PI's original specification exactly.

```
image (28x28, upsampled to 224x224)
  -> ResNet-18 truncated after layer3, ImageNet-pretrained
  -> global pool -> h (256-d)
  -> Linear(256, d) bottleneck -> z (d-d)
  -> z_tilde = tanh(z) * ANGLE_SCALE        [applied to EVERY arm]
  -> HEAD  (the only thing that varies)
  -> Linear(d, C) classifier                [shared by every arm]
```

Two encoder regimes, both driven from one script so they cannot drift apart:

| `freeze_policy` | Meaning | Trainable params |
|---|---|---|
| `"all"` | backbone fully frozen | 1,038 |
| `"layer3_only"` | layer3 unfrozen — the end-to-end setting | 2,100,750 |

There is **no** `"frozen"` value; using it raises.

**Manuscript note.** Because tanh is applied to every arm, the arm named
`linear` is really tanh-then-linear. Call it an *identity head* in prose;
`config.ARM_DISPLAY_NAMES` holds the labels for tables and figures.

---

## 3. The arms

| Arm | Role | Head params (d=4) |
|---|---|---|
| `linear` | capacity floor (`IdentityHead`) | 0 |
| `mlp` | non-linearity at zero extra params | 0 |
| `deep_funnel` | failure is not a depth problem | — |
| `matched_param` | **rank-limited — DIAGNOSTIC ONLY** | 24 |
| `matched_param_fullrank` | capacity control, full rank **at d=4 only** | 24 |
| `low_rank` (rank=2) | capacity control, full rank **at any d** | 24 |
| `fourier_rff` | function-class control (dequantization) | 324 |
| `fourier_exact` | function-class ceiling (d ≤ 8) | 328 |
| `quantum_vqc` | treatment, R=1, spectrum 3^d = 81 | 24 |
| `quantum_reupload` | R=2, spectrum 5^d = 625, same params | 24 |
| `pca_svm` | non-neural reference, excluded from the test family | — |

### Two parity axes, deliberately distinct

- **Parameter parity** → `matched_param_fullrank`, `low_rank`. Answers *is the
  quantum head more efficient per parameter?*
- **Basis-dimension parity** → `fourier_rff`, `fourier_exact`. Answers *is any
  advantage quantum, or just trigonometric?*

Matching the Fourier arm on *parameters* would allow only ~2.5 frequencies. The
VQC's 81-function basis is free from the embedding, and so is the RFF basis;
matching on parameters would hand the VQC 81 dimensions and cap its competitor
at 5. The manuscript states this explicitly.

### The parity identities — these constrain the whole design

```
MatchedParamFullRankHead:   d² + 2d = 6d   ⟺  d = 4      ONLY
LowRankHead:             2·d·r + 2d = 6d   ⟺  r = 2      ANY d
```

At d=8 a dense d×d matrix already costs 64 parameters against a 48-parameter
budget, so exact parity and full rank are not simultaneously achievable in dense
form above d=4. **`low_rank` at rank=2 is what unblocks d=8 and d=16.**

---

## 4. Where the research stands

| Q | Question | Status | Evidence |
|---|---|---|---|
| Q0 | Does compression to d=4 cost anything? | **Answered: no** | 144 cells |
| Q1 | Efficiency at 24 vs 24 params? | **Answered: tie overall** | 400 runs |
| Q2 | Can the VQC exploit its own function class? | **Answered: no** | 1,200 runs |
| Q3 | Does the encoder absorb the bottleneck? | **Answered: yes** | 1,200 runs |
| Q4 | Is narrow spectrum the cause? | **running** (183/200) | — |
| Q6 | **Capacity sweep — the mechanism test** | code ready | ~3h |
| Q5 | Input-noise robustness | not run | ~12h |
| — | Confirmatory sweep, 40 seeds | blocked on LR selection | ~22h |
| Q7 | Hardware noise (shot, depolarizing) | code untested | ~8h |
| — | LR selection | code ready | ~7h |
| — | Full-data reference row | not run | ~7h |
| — | d=8 / d=16 via `low_rank` | unblocked, not run | ~20h |

### Findings, with numbers

**Q0 — compression is nearly free.** Mean AUC gap (d=256 − d=4) = **+0.0018**.
BreastMNIST is *better* at d=4. Residual effect is multi-class only: PathMNIST
full-data error 2.91% → 1.24% from d=4 to d=16. *This killed the conference
paper's central premise, and the project was reframed around data scarcity.*

**Q1 — tie at equal parameters.** 31 of 40 cells no difference, 7 classical
better, 2 quantum better. Measured against the rank-limited control, so the
classical side is if anything understated.

**Q2 — the VQC does not exhaust its own function class.** Against a
324-parameter Fourier head over the same basis: BloodMNIST 8/10 cells Fourier
better, PathMNIST 5/6, both binary sets tied. Large effects — PathMNIST n=10
frozen −0.166. The gap scales with class count: 2 classes tie, 8 ≈ −0.08,
9 ≈ −0.15.

**Q3 — the encoder absorbs the constraint.** BloodMNIST n=5, linear head:
0.640 frozen → 0.825 adaptive. Head differences shrink 3–5× once the encoder can
adapt. **Unplanned finding:** at PathMNIST n=100 the direction *inverts* —
frozen 0.9632 beats adaptive 0.9351. Fine-tuning layer3 on 900 images across
9 classes overfits.

**The crossover (post-hoc — must be confirmed).** Frozen encoder,
Δ = quantum − matched, averaged over four datasets:

| shots/class | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ | **+0.039** | **+0.023** | −0.025 | −0.023 | −0.020 |
| datasets favouring quantum | 4/4 | 3/4 | 1/4 | 0/4 | 0/4 |

Monotone, flipping between n=10 and n=20. Sign test on n=5+10 pooled: 7/8
positive, p ≈ 0.035. **Absent with an adaptive encoder** — consistent with Q3.

**Sanity check against a zero-parameter baseline.** Quantum − `linear`, frozen:
excluding PathMNIST the crossover survives (+0.009, +0.018, −0.044, −0.023,
−0.019); PathMNIST is hostile at every n. So the effect was *inflated* by the
broken control but is not purely an artifact of it.

---

## 5. Verified facts — do not re-derive

- **The dequantization result.** With `AngleEmbedding(rotation='Y')` and no
  re-uploading, every measured expectation lies exactly in the `3^d`
  trigonometric span. Verified to 1e-16 across six (d,L) configurations, with a
  wrong-frequency negative control failing at 0.908. Reproduces the spectrum
  theorem of Schuld, Sweke & Meyer (2021): generator `Y/2`, eigenvalues ±½,
  differences {−1, 0, +1}.
- Re-uploading widens the spectrum to `{−R..R}^d`, verified numerically.
- Gradient flow with `diff_method="backprop"` is correct; adjoint agrees to
  3e-7 and is 260× slower at d=16.
- Barren-plateau onset: gradient variance falls ~62× from d=4 to d=16 at L=4.
  Consistent with McClean et al. (2018).
- `fourier_rff` spans 80 independent features at d=4 after the
  canonical-frequency fix (previously 68 effective, from ±ω duplication).
- Gradient clipping at 20.0 never binds. Measured norms span 0.48 (quantum) to
  15.9 (deep_funnel); at the old threshold of 1.0 it bound on classical arms
  while never touching the quantum arm.
- Encoded amplitudes are **real**. The conference claim about mapping into a
  "complex Hilbert space" is wrong and must not reach the journal draft.

### Runtime, measured

| Setting | Cost per run |
|---|---|
| Classical head, frozen, cached features | seconds |
| `quantum_vqc`, frozen | ~2.5 min |
| `quantum_reupload`, frozen | 50 s (BreastMNIST) → ~13 min (PathMNIST) |
| Any arm, adaptive encoder | up to 46 min |

**Per-run cost scales strongly with dataset size.** Estimates taken from
BreastMNIST have under-predicted PathMNIST by 5–20× on two separate occasions.
Always project from the *largest* dataset in the sweep.

---

## 6. Methodology — the non-negotiables

**Scarcity is absolute**, `n_per_class ∈ {5,10,20,50,100}`, never fractional.
1% is 5 images on BreastMNIST and ~900 on PathMNIST; one row of a scaling curve
would mix unrelated experiments.

**Sampling is stratified.** Random subsets can drop entire classes on 8- and
9-class datasets.

**Validation is subsampled to match** — `min(2n, available)` per class. The old
code trained on 54 images and selected the best of 100 epochs on 78, so model
selection consumed more labels than training did. `val_train_ratio` is logged
and reported.

**Checkpoint selection is on validation AUC**, with F1-selection reported
alongside as a stated sensitivity check. Selecting on F1 while reporting AUC let
the criterion interact with the VQC's calibration failure — a confound, not a
preference.

**Augmentation is off on both sides of the frozen/adaptive comparison.** Feature
caching requires deterministic features; if only the adaptive side augmented,
freezing and augmentation would vary together.

**Noise is injected in physical pixel space**: inverse-normalize → inject →
clamp [0,1] → re-normalize. Clamping *normalized* tensors to [0,1] — where the
real range is roughly [−2.1, 2.6] — destroys signal instead of modelling a
sensor. Conference numbers under that protocol are not comparable to these.

**Regularization parity is absolute.** Identical weight decay, learning rate and
clipping across arms, or the run does not happen.

**Every run saves per-sample test probabilities.** The pre-registered statistic
is a *nested* bootstrap resampling test indices as well as seeds, impossible
from scalar metrics. `shards.save_predictions()` / `shards.load_predictions()`
are the single naming authority — three independent conventions once caused `04`
to find nothing and silently fall back to seed-level resampling while printing a
table that looked correct.

### Metrics

AUC (primary, threshold-free), Macro-F1, average precision, sensitivity,
specificity, balanced accuracy, ECE, predicted-probability spread. Reporting all
of them is what distinguishes a **calibration** failure from a **ranking**
failure — the distinction the "Zombie State" reframing depends on. Per-epoch
evaluation uses `light=True` (AUC, F1, ECE, spread); the clinical metrics are
computed only on test evaluations.

### Statistics

Nested paired bootstrap, B=2000, resampling both test indices and seeds, paired
on seed. Benjamini–Hochberg over the family declared in `docs/analysis_plan.md`
(17 tests) — pass `--family-size 17`. Cohen's d with every p-value. Differences
below 0.01 AUC are labelled negligible: smaller than the test-set sampling error
on the smallest dataset.

**Not** a Welch t-test over seeds. With n_test = 156 on BreastMNIST the AUC
standard error is ≈0.03–0.04 — larger than any effect at stake — so a seed-only
test can report significance on a difference a different test draw would
reverse. That is the error the conference version made.

---

## 7. Environment and workflow

```
conda env: qml_v2      (fallback: qml_journal)
python 3.10 · torch 2.4.1+cu118 · torchvision 0.19.1+cu118
pennylane 0.42.3 · numpy 1.26.4 · medmnist 3.0.2
GPU: A100-SXM4-40GB MIG 3g.20gb. Host driver 470 caps CUDA at 11.4 — cu118 max.
simulator: default.qubit + backprop at every dimension
```

Edit and commit on **Windows** (`C:\Users\Anish\quantum-classical-expressivity`);
pull and run on **Linux** (`/home/jovyan/qml_exp_2026/...`).

**Always `source /home/jovyan/qml_exp_2026/miniconda/bin/activate qml_v2` first.**
Several sessions have been lost to `ModuleNotFoundError` from a shell that
dropped the environment.

**Never `git pull` on Linux while a sweep is running.** `config.git_sha()` is
read at shard-write time, so pulling mid-run splits one experiment across two
commits and breaks provenance.

Artifacts live outside the repo via the `artifacts/` symlink — shards,
predictions, feature cache, data cache. All gitignored.

---

## 8. Known defects, unfixed

- `MatchedParamHead` is rank-limited to width 3 regardless of d. Retained for
  diagnostic reproducibility; **invalid for any comparison at d > 4**.
- `fourier_rff` results predating the canonical-frequency fix used a
  68-dimensional basis and must be regenerated.
- All results predating the AUC-selection change are incomparable with later
  ones. Old Q4 shards are archived in
  `artifacts/shards/_superseded_f1selection/`.
- `src/eval/generate_paper_plots.py` is conference-era and crashes on import.
  **The last file needing a rewrite.**
- `07_hardware_noise.py` has never executed. Run `--quick` before committing
  hours — the `torch.as_tensor(circuit(...))` conversion is untested.
- ETA display under-reports badly: it averages in instantly-returning cached
  cells.

---

## 9. Scope limits — state these in the manuscript

1. **No quantum advantage can be demonstrated** at 4–16 qubits on a state-vector
   simulator. The model is classically simulable by construction — that is how
   it is being run.
2. **Single encoding.** Without re-uploading the spectrum is the most restricted
   possible. Q4 probes this; a full re-uploading study is out of scope.
3. **Ceiling effects.** BloodMNIST, PathMNIST and PneumoniaMNIST sit at
   0.94–0.98 AUC at d=4. Only BreastMNIST has real headroom.
4. **The crossover lives where both models perform poorly** (AUC 0.60–0.68 at
   n=5). A reviewer can fairly say it compares two failures. Disclose it first.
5. **One backbone, one dataset family.** ResNet-18, MedMNIST at 28×28 upsampled
   to 224 — an artificial setting for medical imaging.
6. **Exact parameter parity held only at d=4** for most of the project.
   `low_rank` removes that constraint but has not yet been swept.

---

## 10. Publication assessment — honest

**In favour.** IEEE Access reviews primarily for soundness rather than
excitement. The dequantization control has not previously been applied to a
hybrid CNN+VQC medical pipeline. The paper corrects its own prior claim with
data, which reads as careful rather than weak. The controls are stronger than
is typical for this subfield.

**Against.** It is a characterization result, not a positive one. Some reviewers
want novelty over rigour. The subfield is crowded, and two of the original
contributions have close precedents.

**Cannot be promised.** No methodology guarantees acceptance. What *is*
defensible: the standard failure modes — unmatched baselines, underpowered
tests, uncorrected multiplicity, unverified mechanism claims, hyperparameters
chosen after seeing results — have each been found and closed.

---

## 11. Immediate next steps

1. **Q4 finishes** (~4h from 19 Aug 00:00) → `--summary-only`, read the verdict.
   Prediction recorded 12 Aug, before the data existed: a wider spectrum should
   *hurt* at n=5–10 and *help* at n=50–100. If it holds, the regularization
   mechanism is confirmed rather than merely plausible.
2. **Capacity sweep** (`10_capacity_sweep.py`, ~3h) — the mechanism test. Does a
   restricted *classical* head reproduce the crossover? Pre-register before
   quoting as confirmatory.
3. **Lipschitz** (`08`, minutes) then **Q5 noise** (`03`, ~12h).
4. **LR selection** (`09`, ~7h) → `--use-tuned-lr` → **confirmatory sweep**
   (~22h).
5. **Q7 hardware noise**, **full-data row**, **d=8/16 via `low_rank`**.
6. Rewrite `generate_paper_plots.py`. Figures. Draft.

**If the timeline slips, cut in this order:** depth sweep, tanh ablation,
angle-scale sweep, latent probe, Q7. The primary claim needs only the
confirmatory sweep, Q2 and Q5.

---

## 12. Session conventions

- **Paste the file when in doubt.** Long sessions lose earlier file contents;
  asking is cheaper than a wrong patch.
- **The repo is the source of truth, not the conversation.**
- Verify parameter parity programmatically after any change to `heads.py` or
  `registry.py` — `quantum_vqc`, `matched_param_fullrank` and
  `low_rank(rank=2)` must all report **24** at d=4.
- After any change to prediction I/O, run two cells and then
  `04_statistical_analysis.py`. **If `SEED-LEVEL FALLBACK IN USE` appears,
  stop** — nothing downstream is worth running.
- **Commit before running anything long.** Four days of Q4 compute were once
  lost to an uncommitted revert.
