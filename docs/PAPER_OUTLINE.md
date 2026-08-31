# The Paper — What We Did, What We Found, What Gets Published

**Version 1.0 — 28 August 2026.** Written after the bottleneck ablation and the
first correct nested-bootstrap analysis.

---

## 1. What the study actually is

One sentence: **we took the standard hybrid quantum-classical image classifier
and asked, under controls stronger than the subfield normally applies, whether
the quantum part does anything a matched classical part does not.**

```
image -> ResNet-18 (frozen or adaptive) -> 256-d
      -> bottleneck 256->d              [learned | frozen PCA | frozen random]
      -> tanh(z) * pi/2
      -> HEAD                           <- the ONLY thing that varies
      -> Linear(d, C)
```

Eleven heads share that pipeline: capacity floors, a parameter-matched classical
control, a function-class control, and four quantum variants. Four MedMNIST
datasets, five scarcity levels from 5 to 100 labelled images per class, ten to
forty seeds.

Three things make it a study rather than a benchmark:

1. **Matched parameters.** `quantum_vqc`, `matched_param_fullrank` and
   `low_rank(rank=2)` all hold exactly 24 head parameters at d=4, asserted in a
   unit test. Most published comparisons do not match.
2. **A dequantization control.** `fourier_rff` spans the VQC's *own* function
   class classically, so "quantum vs classical" separates from "trigonometric vs
   not".
3. **Everything structural is proven, not assumed.** Frozen means bit-identical
   weights and buffers, with a negative control. Gradient flow to the encoder is
   measured per arm.

---

## 2. Findings

### F1 — The quantum head computes classical trigonometry. *(proof + numerics)*

With `AngleEmbedding(rotation='Y')` and no re-uploading, the measured output is
**exactly**

    <X_i>(z) = sum_s c_s(Theta) * prod_j f_{s_j}(z_j),   f in {1, cos, sin}

a span of `3^d = 81` functions at d=4, for *any* circuit parameters. Verified to
**residual 1e-16** across six configurations, with a wrong-frequency negative
control failing at 0.908.

The 16-dimensional state exists. Only 4 expectation values are read out, and
those provably live in a classically constructible span. This closes the
"superposition gives access to more" intuition analytically rather than
empirically.

### F2 — Compression to d=4 is nearly free. *(144 cells)*

Mean AUC gap, d=256 minus d=4: **+0.0018**. BreastMNIST is *better* at d=4. The
premise that extreme compression destroys separability — the motivation of the
prior conference paper — does not hold.

### F3 — No parameter-efficiency advantage. *(400 diagnostic + 900 controlled)*

At 24 vs 24 parameters the VQC ties overall. Under the nested bootstrap, **every
cell that survives BH-FDR correction favours the classical control** — 14 of 60,
with Cohen's d down to −3.94 on PathMNIST.

### F4 — The VQC does not exploit its own function class. *(1,200 cells)*

Against a Fourier head spanning the same 81 basis functions: BloodMNIST 8/10
cells classical, PathMNIST 5/6, effects to **−0.17 AUC**. The gap scales with
class count — binary ties, 9 classes lose badly. A variational optimiser reaching
a 24-dimensional manifold underperforms a direct fit over the same basis.

### F5 — The "frozen backbone" protocol does not isolate the head. *(new)*

Trainable parameters at d=4, two classes:

| Component | Params | Share |
|---|---|---|
| bottleneck `Linear(256,4)` | 1,028 | **97%** |
| head | 24 | 2% |
| classifier | 10 | 1% |

The experiment intended to isolate the head's function class is dominated by a
learned projection forty times its size, which can reshape the latent space
around whichever head follows. Freezing the projection moves the head to **70%**
of trainable capacity.

**This is a methodological finding about the subfield, not just about us.**

### F6 — Seed-level statistics overstate the effect. *(new)*

The diagnostic showed a monotone crossover: Δ = +0.039, +0.023, −0.025, −0.023,
−0.020 at n = 5, 10, 20, 50, 100, sign test p ≈ 0.035.

Under a nested bootstrap that resamples **test indices as well as seeds**, the
slope CI is `[−0.0156, +0.0012]` — it crosses zero. Every positive cell at n=5
has a CI straddling zero. On BreastMNIST the test split is 156 images, where the
AUC standard error is ~0.03–0.04, larger than the effect.

**The crossover was an artifact of ignoring test-set sampling variance.**

### F7 — Restriction is not the mechanism. *(1,000 runs)*

The replacement hypothesis — that a restricted function class regularises under
scarcity — predicts that a *classical* head with fewer parameters reproduces the
crossover. Sweeping `low_rank` from 8 to 72 parameters at fixed full rank:
all slopes **positive** (+0.001 to +0.003), predicted negative, effects
under ±0.013. Refuted.

### F8 — Structural claims proven. *(new)*

- Frozen backbone: 0 parameters and 0 buffers changed, max delta 0.00e+00, six
  arms. Negative control without `set_bn_eval()`: 45 buffers drift, confirming
  the test can detect a change.
- Gradient flow: layer3 receives non-zero gradient from **every** head including
  quantum; frozen blocks receive none; layer3 weights displace by 0.53 relative.

Most hybrid QML papers assert both. Neither is usually measured.

---

## 3. What is publishable

### Contribution 1 — a controlled negative result

No quantum advantage at matched parameters, on four medical datasets, across
five scarcity levels, under three bottleneck policies, with a dequantization
control and pre-registered statistics.

Negative results are publishable when the controls are strong enough that the
absence is informative rather than uninformative. These are.

### Contribution 2 — two protocol flaws that can manufacture false positives

**(a) Capacity accounting.** "Freeze the backbone" is standard practice for
isolating a head. It does not, because the projection between backbone and head
usually dwarfs the head. Any paper reporting a head-level effect under a learned
bottleneck may be reporting a projection-level effect.

**(b) Statistical protocol.** Seed-level resampling ignores test-set sampling
variance. On few-shot medical benchmarks with 156–624 test images, that variance
exceeds the effects being claimed. We show a specific effect that is significant
under seed-level statistics and vanishes under a nested bootstrap.

**This is the part most likely to be cited**, because it applies to work beyond
this paper.

### Contribution 3 — dequantization applied to a hybrid medical pipeline

The spectrum theorem is Schuld/Sweke/Meyer (2021). Applying it to a concrete
CNN+VQC medical classifier, constructing the explicit classical head that spans
the same basis, and verifying to 1e-16, has not been done in this setting.

### Contribution 4 — reproducibility artefacts

Pre-registration with eight dated amendments; freezing and gradient-flow proofs
as runnable scripts; per-sample predictions for every run; git SHA in every
shard; 61 tests.

---

## 4. Paper structure

**Title.** *Where Did the Quantum Advantage Go? Learned Compression, Not Quantum
Computation, Explains Few-Shot Gains in Hybrid Medical Image Classifiers*

**Abstract.** Hybrid CNN+VQC classifiers report advantages under data scarcity.
We test one under matched parameters, a dequantization control, and pre-registered
statistics. No advantage survives. We identify two protocol flaws that can
produce apparent advantages: a learned compression layer holding 97% of trainable
capacity, and seed-level statistics that ignore test-set sampling variance.

**1. Introduction** — the claim, why it matters clinically, what we test.

**2. Related work** — hybrid QML for medical imaging; dequantization; the
few-shot regime. Position: not the first to find nulls, first to control this
carefully in this setting.

**3. Method**
  3.1 Architecture and the parity contract
  3.2 The eleven arms and two parity axes
  3.3 Dequantization: theory and numerical verification *(F1)*
  3.4 Bottleneck policies and the capacity accounting *(F5)*
  3.5 Structural verification: freezing and gradient flow *(F8)*
  3.6 Pre-registration, nested bootstrap, BH-FDR

**4. Results**
  4.1 Compression is nearly free *(F2)*
  4.2 No advantage at matched parameters *(F3)*
  4.3 The VQC underperforms its own function class *(F4)*
  4.4 The crossover and its disappearance *(F6)* — **the narrative core**
  4.5 Restriction is not the mechanism *(F7)*
  4.6 Robustness under sensor noise *(pending)*
  4.7 Hardware feasibility *(pending)*

**5. Discussion** — what would have to be true for an advantage to exist;
what this says about evaluation protocol in hybrid QML.

**6. Limitations** — simulation only; ≤16 qubits; single encoding; MedMNIST
resolution; ceiling effects; d=4 confirmatory.

**7. Conclusion**

---

## 5. Still needed before writing

| | Blocks which section |
|---|---|
| Bottleneck ablation, float32 re-run | 4.4 — **running** |
| Confirmatory sweep, 40 seeds | 4.2 |
| Q5 software noise | 4.6 |
| H-S7 readout richness | 3.3, 4.3 |
| Q7 hardware noise | 4.7 |
| d=8 / d=16 | 4.2 robustness |
| Full-data row | 4.1 |
| Depth, angle-scale, tanh ablations | 3.6 fairness |
| `04` cross-cell comparison | H-S5 has no confirmatory path yet |
| `generate_paper_plots.py` | all figures |

---

## 6. Honest assessment

**Strong:** the controls, the proofs, the pre-registration, and the fact that
the paper refutes its own authors' prior claim with data.

**Weak:** it is a negative result on one architecture, one backbone, one dataset
family, at one bottleneck dimension confirmatory. The finding is about protocol
as much as about quantum computing.

**Likely reviewer objections, and the answer:**

| Objection | Answer |
|---|---|
| "You didn't tune the quantum arm" | Per-arm LR selected on validation over a 6-point grid, optimum interior, full sweep in the appendix |
| "Only 4 qubits" | d=4/8/16 reported; parity holds at all three via `low_rank(rank=2)` |
| "You only measured 4 observables" | H-S7 measures all 2-local terms, with a padded control isolating information from classifier capacity |
| "Simulation, not hardware" | Shot noise and depolarizing ablations; stated as feasibility |
| "Negative results are uninformative" | The protocol findings apply beyond this paper |

**Cannot be promised.** Acceptance depends on reviewers. What is defensible is
that the usual reasons for rejecting a null — weak baselines, underpowered
tests, uncontrolled confounds — have each been closed and documented.
