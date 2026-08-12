# Hypotheses

Every hypothesis the paper will state, what tests it, what evidence exists, and what each
possible outcome would mean. Statuses are current as of 12 August 2026.

---

## The one-sentence version

> At a fixed small parameter budget and with very few labelled images, does a variational
> quantum head extract more from a compressed feature vector than a classical head — and
> if it does, is the advantage quantum, or merely trigonometric?

Everything below serves that question.

---

## H0 — Latent compression is nearly free

**Claim.** Compressing a 256-d feature vector to d=4 costs almost no accuracy, provided the
encoder can adapt to the bottleneck.

**Test.** Experiment 6. Identical architecture and training, `d ∈ {4,8,16,32,64,256}`,
linear head, 4 datasets × {n=100, full} × 3 seeds = 144 cells.

**Result — COMPLETE.** Mean AUC gap (d=256 − d=4) = **+0.0018**. Per-cell gaps span
−0.018 to +0.014, all within seed standard deviations. On BreastMNIST, d=4 is *better*
than d=256.

Residual signal appears only on multi-class tasks: PathMNIST full 0.9709 (d=4) vs 0.9876
(d=16) — error 2.91% → 1.24%, roughly a doubling of residual error. Binary datasets show
nothing.

**Interpretation.** *Compression to d=4 costs little in absolute AUC and nothing on binary
tasks, but roughly doubles residual error on multi-class tasks.*

**What this kills.** The conference paper's "topological collapse", the "Bottleneck Gap",
and the edge-compression motivation. All contradicted by the project's own control.

**Why it is a contribution.** Much of the hybrid-QML literature implicitly assumes
aggressive compression justifies a quantum head. This is 144 controlled cells across four
modalities showing otherwise — including a correction of the authors' own earlier claim.

---

## H1 — Parameter efficiency **(PRIMARY)**

**Claim.** At an identical parameter budget and under data scarcity, the VQC extracts more
from the compressed vector than a classical head.

**Test.** `quantum_vqc` (24 head parameters) vs `matched_param` (24 head parameters), both
emitting 4 features to the classifier, paired over seeds.
4 datasets × n ∈ {5,10,20,50,100} × d=4 × {frozen, adaptive} × 10 seeds = 400 runs.

**Status — RUNNING (~6h).** This is the question the paper is built on, and it has not yet
been tested. The 1200-cell diagnostic omitted `matched_param`.

**Parity established.** Both arms: 24 head parameters, 4 features to the classifier,
weight decay 1e-4, LR 1e-3, clip 20.0, identical splits and seeds. Convergence audited —
best epoch 52.9 (quantum) vs 56.9/57.9 (classical); the VQC is not undertrained.

**Outcomes.**
- *VQC wins* → a parameter-efficiency result with unusually strong controls. H3 then
  determines whether the advantage is quantum or trigonometric.
- *Tie* → the quantum head is one implementation among several at this budget.
- *VQC loses* → a negative result, bounded by the §4 scope note in the math verification.

---

## H2 — The encoder absorbs the bottleneck

**Claim.** H0 holds only because the encoder reorganises around the constraint. Freeze it
and the compression penalty appears.

**Test.** Experiment 1, `frozen` (backbone fully frozen, 1,038 trainable parameters) vs
`adaptive` (layer3 unfrozen, 2,100,750). Augmentation **off on both sides** so freezing is
the only difference.

**Result — COMPLETE, 1200 cells.** Adaptation dominates:

| cell | frozen | adaptive |
|---|---|---|
| BloodMNIST n=5, linear | 0.640 | 0.825 |
| BloodMNIST n=20, linear | 0.820 | 0.933 |
| PathMNIST n=5, linear | 0.754 | 0.876 |

**Unplanned finding — a crossover.** At PathMNIST n=100 the direction **inverts**: frozen
linear 0.9632 beats adaptive 0.9351, and the same holds for the other heads. Fine-tuning
layer3 on 900 images across 9 classes overfits. Adaptation helps under scarcity and hurts
once data is sufficient — a scarcity-dependent crossover nobody set out to look for.

**Interpretation.** This is the reframed "Latent Reshaping" claim: properly tested, and a
property of *every* head rather than a special virtue of the quantum one.

---

## H3 — Dequantization: is any advantage quantum, or just trigonometric?

**Claim.** The VQC's output lies inside an explicitly constructible `3^d` classical
trigonometric span. A classical head drawing from the same span should therefore match it.

**Theoretical basis.** Derived in `MATH_VERIFICATION.md` §2–3 and consistent with Schuld,
Sweke & Meyer (2021). Confirmed numerically at machine precision
(`tests/test_fourier_equivalence.py`, residuals 6e-16 to 2e-15, with a wrong-frequency
negative control failing at 0.908).

**Result — COMPLETE but requiring careful framing.** Against `fourier_rff` (324 head
parameters), 1200 cells:

| dataset | Fourier better | no difference | VQC better |
|---|---|---|---|
| BloodMNIST (8 classes) | 8 | 2 | 0 |
| PathMNIST (9 classes) | 5 | 1 | 0 |
| BreastMNIST (binary) | 0 | 9 | 1 |
| PneumoniaMNIST (binary) | 0 | tied throughout | 0 |

Effect sizes on multi-class are large and far outside noise: PathMNIST n=10 frozen −0.166,
n=20 frozen −0.152, BloodMNIST n=20 frozen −0.101. The gap scales with class count:
2 classes → tie, 8 → ~−0.08, 9 → ~−0.15.

**Required framing.** This comparison is matched on **basis dimension**, not on parameter
count (324 vs 24). It answers *"does the quantum coefficient manifold `ℳ_{L,n}` reach as
much of the shared function class as a direct fit?"* — a legitimate and interesting
question, but **not** the parameter-efficiency question. H1 answers that. Both belong in
the paper, labelled distinctly.

**Novelty.** The dequantization control has not previously been applied to a hybrid
CNN+VQC medical imaging pipeline. This is the strongest card in the paper.

---

## H4 — Differential noise robustness

**Claim.** At matched parameters, the quantum and classical heads degrade differently under
analog sensor noise.

**Test.** Experiment 3, AWGN in physical pixel space, `σ ∈ [0, 0.20]`, RNG parity across
arms, reporting AUC **and** Macro-F1 **and** calibration (ECE, probability spread) at every
level.

**Status — NOT STARTED.** Must include `matched_param` and `fourier_rff`, not just the
original three arms: plain "quantum vs classical noise robustness on MedMNIST" is already
published, so only the *differential* comparison against matched controls is novel.

**Prior signal.** The conference "Zombie State" — AUC 0.6118 with probability standard
deviation 0.0057 — is the signature of a **calibration** failure, not a decision-boundary
failure. If AUC holds while F1 collapses, the finding is threshold drift, which is a
narrower and more accurate claim than the conference version made.

**Existing contradiction in the literature.** One published paper reports VQCs as more
noise-fragile than classical models on MedMNIST; another reports them as more robust.
Nobody has explained the discrepancy. Data scarcity is a plausible resolving variable, and
the conference results already hint at it.

---

## H5 — Spectral richness vs quantum-ness **(PROPOSED)**

**Claim.** Performance is governed by the richness of the accessible frequency spectrum,
not by the computation being quantum.

**Rationale.** Because each feature is encoded once, the spectrum is confined to
`{−1,0,+1}` per coordinate — the most restricted possible. Pérez-Salinas et al. (2020) and
Schuld et al. (2021) establish that data re-uploading widens this to `{−L..L}` and that
expressivity tracks spectral richness. Universality requires it.

**Test.** Add a re-uploading arm (encoding repeated 2–3 times). If performance rises with
spectrum width, the operative variable is spectral richness rather than quantum-ness.

**Why it matters.** It is the first question an informed reviewer will ask about a
single-encoding architecture, and without it the H1/H3 results are bounded by an
untested confound. Infrastructure already supports it; cost is roughly one day.

**Recommendation: add it.**

---

## Status summary

| # | Hypothesis | Evidence | Status |
|---|---|---|---|
| H0 | Compression is nearly free | 144 cells | **Complete** |
| H1 | Parameter efficiency | 400 runs | **Running** |
| H2 | Encoder absorbs the bottleneck | 1200 cells | **Complete** |
| H3 | Dequantization | 1200 cells + machine-precision proof | **Complete** |
| H4 | Differential noise robustness | — | Not started |
| H5 | Spectral richness | — | Proposed |

---

## Manuscript ordering

1. **Theory** — the `3^d` span, its derivation, the numerical verification, and its
   grounding in Schuld et al. 2021. This frames everything.
2. **H0 + H2** — compression is nearly free, and the encoder is why. Corrects a widespread
   assumption, including the authors' own earlier claim.
3. **H1** — the parameter-efficiency comparison. The empirical core.
4. **H3** — dequantization. Determines whether any H1 advantage is quantum.
5. **H4** — robustness, resolving the published contradiction.
6. **H5** — spectral richness, if run.

**Scope statement required in the discussion.** At 4–16 qubits on a state-vector simulator
this work cannot and does not demonstrate quantum advantage; the model is classically
simulable by construction — that is how it is being run. What it establishes is whether a
specific quantum-derived inductive bias is *useful* at a fixed budget. That is a real and
modest contribution, and stating the limit plainly is a strength in review, not a weakness.
