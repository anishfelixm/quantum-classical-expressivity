# The Paper — Outline v2.0

**11 September 2026.** All experiments complete: 12 sweeps, ~11,000 runs,
`Integrity: 0.00e+00` on every post-float32 namespace.

Supersedes v1.0. The change is positioning: a literature check found close
precedents for the bottleneck finding, so the paper must *cite and extend* that
work rather than claim to have discovered it.

---

## 1. What the literature already says

Checked September 2026. These must all appear in Related Work.

### Directly adjacent — the bottleneck confound

**Chen & Kuo, "Quantum Adaptive Self-Attention" (arXiv:2504.05336).** The v2
preprint (June 2025) does not contain it, but a later revision adds a section
titled "Is the gain quantum, or just a compact bottleneck?" It replaces the PQC
value map with a rank-2 bilinear layer with bounded tanh — 40 parameters against
the PQC's 36 — on synthetic time-series regression, 5 seeds, and concludes the
gain comes from the low-rank bottleneck structure rather than quantumness.

**This is the closest prior work and must be cited prominently.** Note it is an
arXiv preprint (the PDF carries a Quantum-journal template artifact, not an
acceptance), and the bottleneck control appears to be a revision to a paper whose
headline claim is quantum advantage.

**Also relevant.** A 2026 radar-occupancy paper defines classical control heads
that retain the CNN backbone and 2-D bottleneck "to attribute any observed
differences to the PQC rather than the bottleneck itself" (arXiv:2601.11929). A
March 2026 paper attaches VQCs to frozen convolutional backbones and states its
objective is "not to demonstrate quantum advantage" (arXiv:2603.16973).
arXiv:2605.19417 proposes controlled frozen-backbone benchmarking for quantum
transfer learning.

### Theory we build on, not extend

- Schuld, Sweke & Meyer (2021) — the spectrum theorem giving the trigonometric
  structure of PQC outputs.
- Schreiber, Eisert & Meyer — classical surrogates for quantum learning models.
- Sweke et al., *Quantum* 9, 1640 (2025) — necessary and sufficient conditions
  for RFF dequantization; notes that all PQC models are linear in a trigonometric
  polynomial feature map and that one can optimise directly over that class.
- Berberich et al., *Phys. Rev. Research* 6, 043326 (2024) — Lipschitz bounds for
  quantum models; trainable encodings and robustness regularisation.
- arXiv:2404.16154 — compares QML and classical adversarial robustness via
  Lipschitz bounds, including a classical Fourier network matched to a
  re-uploading circuit.
- McClean et al. (2018) — barren plateaus.
- *Sci. Rep.* 16, 9017 (2026) — first comprehensive MedMNIST QML benchmark on
  127-qubit IBM hardware (no classical neural network).

### What remains ours

1. **Capacity accounting.** Others match the head's size; nobody quantifies that
   the projection holds **97%** of trainable parameters, nor measures the effect
   of removing it (**Cohen's d up to +6.73**).
2. **Removing the projection, two ways.** Frozen PCA (optimal linear) and frozen
   random (Johnson–Lindenstrauss). Prior work matches size; we eliminate
   adaptivity.
3. **Exact parity at three dimensions.** 24/24 at d=4, 48/48 at d=8, 96/96 at
   d=16 via `LowRankHead(rank=2)`. QASA's control is 40 vs 36.
4. **Readout richness with a padded control.** 2-local observables versus
   duplicated singles at matched classifier width. No precedent found.
5. **Dimension-scaling of the Lipschitz constant, measured.** Theory gives the
   bound; we measure it across matched arms at d = 4, 8, 16.
6. **The statistical-protocol demonstration.** An effect significant under
   seed-level resampling that vanishes under a nested bootstrap, on real few-shot
   medical data.
7. **Pre-registration with 11 dated amendments**, plus freezing and
   gradient-flow proofs with a negative control.

---

## 2. Title and framing

> **Where the Capacity Lives: A Dequantization- and Capacity-Controlled
> Evaluation of Variational Quantum Heads for Few-Shot Medical Image
> Classification**

**Framing sentence for the introduction:**

> Recent work has raised the possibility that gains attributed to parameterized
> quantum circuits may stem from the low-capacity bottleneck they impose rather
> than from quantum computation, comparing a PQC against an approximately
> parameter-matched classical low-rank map on synthetic regression tasks
> [Chen & Kuo]. We take that concern as our starting point and address it
> systematically on real medical imaging data: we quantify how trainable capacity
> is distributed across the architecture, construct exactly parameter-matched
> controls at three bottleneck dimensions, and — rather than matching the
> projection's size — remove its adaptivity entirely using both an optimal and a
> random frozen projection.

---

## 3. Results, in the order they should appear

### R1 — The quantum head computes classical trigonometry
Residual **1e-16** across six configurations; wrong-frequency control fails at
0.908. Establishes the function class before any comparison.

### R2 — Unitarity bounds the head, dimension-independently *(lead result)*

| arm | L(d=4) | L(d=8) | L(d=16) | \|out\|max |
|---|---|---|---|---|
| quantum_vqc | 1.515 | 1.393 | **1.203** | **0.921** |
| fourier_rff | 3.86 | 4.758 | **5.671** | 3.883 |
| matched_param | 3.74 | 4.501 | **5.240** | 8.731 |

Quantum output bounded below 1 at every dimension; its constant falls with d
while classical constants grow. Ratio to quantum: `fourier_rff` 2.55× → 3.42× →
4.72×.

### R3 — The learned bottleneck dominates *(the capacity result)*

| component | params | share |
|---|---|---|
| bottleneck `Linear(256,4)` | 1,028 | **97%** |
| head | 24 | 2% |
| classifier | 10 | 1% |

Freezing it costs up to +0.293 AUC (**d = +6.73**), and the scarcity crossover is
present under a learned projection (slope −0.0145) but **absent under both frozen
policies** (PCA −0.0119, random +0.0049).

### R4 — No robust parameter-efficiency advantage

| n/cls | 5 | 10 | 20 | 50 | 100 | full |
|---|---|---|---|---|---|---|
| pooled Δ | **+0.0142** | +0.0002 | −0.0050 | **−0.0080** | **−0.0074** | −0.0068 |

H-P1 and H-P2 both supported at d=4 — **but** Δ(5) collapses to +0.0041
[−0.0057, +0.0148] without BloodMNIST, reverses at d=8 (classical better at n=10
and n=50), and vanishes at d=16. The classical advantage **plateaus at ≈ −0.007**
rather than widening with unlimited data.

### R5 — Every proposed mechanism fails
Superposition (R1). Capacity restriction — 1,000-run classical sweep, 19/20 null.
Readout impoverishment — see R6, it helps but not where the advantage is.

### R6 — Richer readout helps at matched parameters
`quantum_rich` − `quantum_rich_padded`: +0.0139 [+0.0046, +0.0250] at n=10,
+0.0139 at n=20, +0.0075 at n=50. Identical circuit, identical 24 parameters,
identical classifier width.

### R7 — Failure under noise is calibration, not ranking
PathMNIST n=20, σ=0.20: `fourier_rff` AUC 0.966→0.613 but F1 0.719→**0.130**;
`quantum_vqc` AUC 0.925→0.620, F1 0.600→**0.087**. Depolarizing noise is
**provably AUC-invariant** for binary tasks (`⟨X_i⟩ → c⟨X_i⟩`, monotone in the
decision score), confirmed to four decimals; the effect appears only in ECE and
probability spread.

### R8 — Fairness ablations
**Depth:** L=4 beats L=2 on multi-class, but 3·L·d = 48 parameters breaks parity,
so L=2 is the only depth at which the comparison is matched — now justified by
data. **Angle scale:** π vs π/2 null for the quantum arm, worse for the classical
control; the pre-registered choice did not handicap the treatment.
**tanh:** removing it helps `mlp` (+0.0412), `low_rank` (+0.0463) and
`fourier_rff` — **the classical arms were carrying a constraint only the quantum
arm needs, and still won.**

### R9 — Structural claims proven
Frozen backbone bit-identical (0 params, 0 buffers, six arms) with a negative
control that drifts 45 buffers. Gradients reach layer3 from every head; layer3
displacement 0.53.

---

## 4. Paper structure

1. Introduction — the claim, the clinical motivation, the capacity question
2. Related work — §1 above, QASA positioned explicitly
3. Method — architecture, parity contract, dequantization, bottleneck policies,
   structural verification, pre-registration and statistics
4. Results — R1…R9 in the order above
5. Discussion — what would have to be true for an advantage; evaluation protocol
   in hybrid QML
6. Limitations
7. Conclusion

---

## 5. Remaining work

| | |
|---|---|
| `src/eval/generate_paper_plots.py` | rewrite — last unwritten code |
| `docs/analysis_plan.md` | Amendments 3a, 9, 10, 11 |
| `docs/RESULTS.md` | fold in full-data, depth, angle, tanh |
| `paper/main.tex` | rewrite |

No compute remains.

---

## 6. Honest assessment

**Strong:** controls beyond the subfield's norm; two structural proofs; exact
parity at three dimensions; pre-registration with a visible amendment log; a
paper that corrects its own authors' prior claim.

**Weak:** the headline is a negative result; one backbone, one dataset family;
28×28 upsampled; simulation only; the bottleneck insight has a preprint
precedent.

**Likely objections, and the answers:**

| Objection | Answer |
|---|---|
| "QASA already showed this" | Cited. They matched size on synthetic regression with 5 seeds; we quantify the 97% share, remove the projection two ways, at three dimensions, with 40 seeds and pre-registration |
| "You didn't tune the quantum arm" | Six-point per-arm grid on validation, optimum interior, full sweep in the appendix |
| "Only 4 qubits" | d = 4, 8, 16 with exact parity at each |
| "You only measured 4 observables" | R6 — measured all 2-local terms, with a padded control |
| "Simulation, not hardware" | Shot and depolarizing ablations; stated as feasibility |
| "Negative results are uninformative" | The capacity and protocol findings apply beyond this paper |

**Cannot be promised.** Acceptance depends on reviewers. What is defensible is
that the usual grounds for rejecting a null — weak baselines, underpowered tests,
uncontrolled confounds, undisclosed prior art — have each been closed.
