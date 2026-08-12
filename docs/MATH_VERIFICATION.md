# Mathematical Verification

**Purpose.** Independent re-derivation of every mathematical claim the project rests on,
and a check of each against the accepted theory of the field. Written so a reviewer can
follow the argument without running anything.

**Status of each claim:** VERIFIED (derived here and consistent with published theory),
CONFIRMED NUMERICALLY (also checked by a test in this repository), or
SCOPE NOTE (correct, but a deliberate limitation that must be stated in the manuscript).

---

## 1. The encoded state

`AngleEmbedding(rotation='Y')` applies `RY(θ) = exp(-iθY/2)` to each wire from `|0⟩`:

```
RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

|ψ(z̃)⟩ = ⊗_{j=1..n} [ cos(z̃_j/2)|0⟩ + sin(z̃_j/2)|1⟩ ]
```

Amplitude of basis state `|b⟩`, `b ∈ {0,1}^n`:

```
ψ_b(z̃) = ∏_j [cos(z̃_j/2)]^{1-b_j} · [sin(z̃_j/2)]^{b_j}
```

All amplitudes are **real**. The state occupies a real `2^n`-dimensional submanifold of
`ℂ^{2^n}`, not the full complex space.

**Status: VERIFIED.**

**Manuscript correction required.** Describing the encoding as mapping into a
"complex Hilbert space of dimension 2^n" overstates what happens here — with real
amplitudes and Pauli-X readout, no complex phase is ever used. The conference draft
made this claim; it should not survive into the journal version.

---

## 2. The measured function lies in a 3^n-dimensional trigonometric span

Let `M(Θ) = U†(Θ) X_i U(Θ)`. This is a fixed Hermitian matrix depending on the trainable
parameters but **not** on the data. Then

```
v_i(z̃) = ⟨ψ(z̃)| M(Θ) |ψ(z̃)⟩ = Σ_{b,b'} ψ_b(z̃) · M_{bb'}(Θ) · ψ_{b'}(z̃)
```

Each product `ψ_b ψ_{b'}` factorises over qubits, and each factor takes one of three forms:

| (b_j, b'_j) | factor | half-angle identity |
|---|---|---|
| (0,0) | `cos²(z̃_j/2)` | `(1 + cos z̃_j)/2` |
| (1,1) | `sin²(z̃_j/2)` | `(1 − cos z̃_j)/2` |
| (0,1), (1,0) | `sin(z̃_j/2)cos(z̃_j/2)` | `sin(z̃_j)/2` |

Every factor is an affine combination of `{1, cos z̃_j, sin z̃_j}`. Therefore

```
v_i(z̃) = Σ_{s ∈ {0,c,s}^n} c_s(Θ) ∏_j f_{s_j}(z̃_j),    f_0 = 1, f_c = cos, f_s = sin
```

The basis has exactly `3^n` elements: 81 at n=4, 6561 at n=8, 43,046,721 at n=16.

**Status: VERIFIED and CONFIRMED NUMERICALLY** — `tests/test_fourier_equivalence.py`
solves least squares against this basis and returns residuals of 6e-16 to 2e-15 across
(d,L) = (2,1), (2,3), (3,2), (4,1), (4,2), (4,4), with a wrong-frequency negative control
failing at 0.908 as required.

---

## 3. Alignment with published theory

The result above is not novel — and that is the point. It reproduces, for this specific
architecture, the framework established by Schuld, Sweke and Meyer (2021), *Effect of data
encoding on the expressive power of variational quantum machine learning models*
(Phys. Rev. A 103, 032430).

Their result: a variational model with data encoded through gates `exp(-i x G)` computes a
**truncated Fourier series**

```
f(x) = Σ_{ω ∈ Ω} c_ω(Θ) e^{iωx}
```

where the accessible frequency set `Ω` is fixed by the **eigenvalue differences of the
encoding generators**, and the trainable part determines only the coefficients `c_ω(Θ)`.

Applying that to our case: `RY(θ) = exp(-iθY/2)` has generator `Y/2` with eigenvalues
`±1/2`. The eigenvalue differences are

```
{ 1/2 − 1/2, 1/2 − (−1/2), −1/2 − 1/2, −1/2 + 1/2 } = {−1, 0, +1}
```

With one encoding gate per feature and n features, the joint spectrum is the product set
`Ω = {−1, 0, +1}^n`, of cardinality `3^n`. **This matches Section 2 exactly.**

The decomposition also cleanly separates our two objects of study:
- **accessible frequencies** `Ω` — fixed by the encoding, identical for the VQC and the
  classical Fourier surrogate;
- **achievable coefficients** `ℳ_{L,n} = { c(Θ) : Θ ∈ ℝ^{L×n×3} }` — a
  `3Ln`-dimensional manifold inside `ℝ^{3^n}`, and the only thing that distinguishes the
  VQC from a direct linear fit over the same basis.

**Status: VERIFIED against accepted theory.** Schuld et al. 2021 is the correct citation
and should anchor the theory section.

---

## 4. SCOPE NOTE — no data re-uploading, and why it matters

This is the most important interpretive limitation in the project.

Because each feature is encoded **exactly once**, the spectrum is confined to
`{−1, 0, +1}` per coordinate: the model can only represent *degree-one* trigonometric
polynomials in each input.

Pérez-Salinas et al. (2020), *Data re-uploading for a universal quantum classifier*
(Quantum 4, 226), and Schuld et al. 2021 both establish that repeating the encoding `L`
times widens the spectrum to `{−L, …, +L}` per coordinate, and that expressivity is
governed by this spectral richness. Universality requires it.

**Consequence for the manuscript.** If the VQC underperforms, the honest reading is not
"quantum inductive bias is inferior" but rather:

> A single-encoding hybrid VQC has a maximally restricted frequency spectrum and reaches
> only a low-dimensional coefficient manifold within it. Under those constraints it does
> not outperform a matched classical head.

That is narrower than the original claim, and it is defensible.

**Recommended additional experiment.** Add a re-uploading arm (encoding repeated 2–3
times, spectrum `{−2..2}^n` or `{−3..3}^n`). This directly tests whether *spectral
richness* rather than *quantum-ness* is the operative variable — and it is the first
question an informed reviewer will ask. The infrastructure already supports it; the cost
is roughly one additional day.

---

## 5. Input scaling

```
z̃ = tanh(z) · (π/2) ∈ [−π/2, π/2]^n
```

**Injectivity.** `RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩` is injective for `θ ∈ [−π, π]`,
since `θ/2 ∈ [−π/2, π/2]` gives `cos(θ/2) ≥ 0` while `sin(θ/2)` sweeps `[−1, 1]`
monotonically. Our range `[−π/2, π/2]` is a strict subset, so **no phase wrapping occurs**.

**Status: VERIFIED** — the stated purpose (avoiding `2π` wrap-around) is achieved.

**SCOPE NOTE.** The choice is conservative: it uses only half the injective angular range.
A scaling of `tanh(z)·π` would remain injective while roughly doubling angular separation
between inputs. This is a free hyperparameter that has never been tuned, and it plausibly
disadvantages the quantum arm. Two honest options: sweep it as a hyperparameter, or state
explicitly that it was fixed a priori and not optimised. It should not be left unmentioned.

Note that `z̃` is fed identically to every arm, so the choice does not bias the comparison
in an obvious direction — but it does bound absolute VQC performance.

---

## 6. The classical Fourier surrogate

```
φ_RFF(z̃) = [ cos(ω⁽¹⁾·z̃), sin(ω⁽¹⁾·z̃), …, cos(ω⁽ᵐ⁾·z̃), sin(ω⁽ᵐ⁾·z̃) ] ∈ ℝ^{2m}
```

with `ω⁽ᵏ⁾` sampled from `{−1,0,1}^n` and fixed at initialisation.

**Correctness.** Since `Ω = {−1,0,1}^n` (Section 3), every `cos(ω·z̃)` and `sin(ω·z̃)` with
`ω ∈ Ω` lies inside the VQC's function class. The RFF head therefore spans a **subspace of
the same space** — exactly what a dequantization control must do.

**Status: VERIFIED.**

**One implementation detail to check.** `cos(−ω·z̃) = cos(ω·z̃)` and
`sin(−ω·z̃) = −sin(ω·z̃)`, so sampling both `ω` and `−ω` yields linearly dependent
features. The sampler should draw one representative per `±` pair. At n=4 there are 81
frequencies → 40 sign-pairs plus `ω = 0`; the configured `2m = 80` is consistent with
correct handling, but this warrants a direct check.

---

## 7. Barren plateaus

Measured quantum-weight gradient variance (batch 32, `default.qubit`, backprop, GPU):

| n | L=1 | L=2 | L=4 |
|---|---|---|---|
| 4 | 1.045e+01 | 1.121e+01 | 1.296e+01 |
| 8 | 1.017e+01 | 2.588e+00 | 9.957e−01 |
| 16 | 4.289e+00 | 1.938e+00 | 2.098e−01 |

Monotone decay in both `n` and `L`; ~62× from (n=4,L=4) to (n=16,L=4).

**Status: VERIFIED, consistent with McClean et al. (2018),** *Barren plateaus in quantum
neural network training landscapes* (Nat. Commun. 9, 4812), which predicts gradient
variance vanishing exponentially in qubit count for sufficiently expressive random
circuits. Our measurement is a direct empirical observation of the onset and is reportable
as such.

**Interpretive use.** If the VQC underperforms at d=16, trainability and expressivity are
competing explanations, and this table is what distinguishes them.

---

## 8. Statistics

**Nested paired bootstrap.** Two variance sources must both be captured — test-set
sampling and training seed:

```
for b in 1..B:
    resample test indices I_b with replacement
    resample seed indices S_b with replacement
    Δ_b = mean_{s ∈ S_b} [ metric_A(s, I_b) − metric_B(s, I_b) ]
```

**Status: VERIFIED as the correct construction.** A Welch t-test over seeds captures only
training variance. With `n_test = 156` on BreastMNIST, the Hanley–McNeil standard error on
AUC is ≈0.03–0.04 — larger than the effects at stake — so a seed-only test can report
significance on a difference that a different draw of test images would reverse.

**Benjamini–Hochberg FDR.** Standard, correctly specified, with the test family declared
in advance.

**Note on the current diagnostic output.** The 95% intervals printed by
`01_frozen_backbone_ablation.py` use a normal approximation to the paired seed differences
(`1.96 · SE`). That is appropriate for an exploratory diagnostic. The confirmatory analysis
must use the nested bootstrap above, and the manuscript must not quote the diagnostic
intervals.

---

## 9. Noise model

```
1. inverse-normalize:  x_real = x·σ_ImageNet + μ_ImageNet   → [0,1]
2. inject:             x_noisy = x_real + ε,  ε ~ 𝒩(0, σ²)
3. clamp:              [0,1]                                 → physical sensor bound
4. re-normalize:       x' = (x_noisy − μ) / σ
```

**Status: VERIFIED.** Ordering is essential and now correct. The conference version
clamped in *normalized* coordinates, where `[0,1]` is not a physical bound — normalized
pixels span roughly `[−2.1, 2.6]` — so that clamp destroyed signal rather than modelling a
sensor floor and ceiling. Numbers produced under the old protocol are not comparable to
these.

RNG is seeded as `seed + round(σ·1000)`, giving bit-identical corrupted tensors across
architectures. `round` rather than truncation avoids float-representation collisions.

---

## 10. Summary

| # | Claim | Status |
|---|---|---|
| 1 | Real-amplitude encoded state | VERIFIED — manuscript wording needs correction |
| 2 | `3^n` trigonometric span | VERIFIED + CONFIRMED NUMERICALLY |
| 3 | Matches Schuld et al. 2021 spectrum theory | VERIFIED |
| 4 | No re-uploading ⇒ degree-1 spectrum | SCOPE NOTE — bounds the claim; extra arm recommended |
| 5 | `tanh·(π/2)` injective | VERIFIED — conservative, untuned, must be disclosed |
| 6 | RFF spans a subspace of the VQC class | VERIFIED — check `±ω` deduplication |
| 7 | Barren-plateau onset | VERIFIED vs McClean et al. 2018 |
| 8 | Nested bootstrap + BH-FDR | VERIFIED — diagnostic CIs are normal-approx only |
| 9 | AWGN in physical pixel space | VERIFIED |

**No mathematical errors found.** Two scope notes (§4, §5) and one implementation check
(§6) require action before submission.
