# RESULTS — Definitive

**11 September 2026.** ALL EXPERIMENTS COMPLETE — 12 sweeps, ~11,000 runs. Every number below is from a nested paired bootstrap
(B=2000, test indices *and* seeds resampled), BH-FDR corrected at m=23, with
`Integrity: max |recomputed − recorded| = 0.00e+00` on all post-float32 runs.

---

## R1 — The quantum head computes classical trigonometry

`AngleEmbedding(Y)`, single upload: the measured output lies **exactly** in the
`3^d` trigonometric span, for any parameters and any input.

- Residual **1e-16** across six (d, L) configurations
- Wrong-frequency negative control fails at **0.908**
- Reproduces Schuld/Sweke/Meyer (2021) for this architecture

The 2^d state exists; the d single-qubit expectations read out of it are
classically constructible.

---

## R2 — No parameter-efficiency advantage that survives scope changes

**d=4, 40 seeds, tuned per-arm LRs** (`quantum_vqc` − `matched_param_fullrank`):

| n/cls | Δ | 95% CI | verdict |
|---|---|---|---|
| 5 | **+0.0142** | [+0.0026, +0.0258] | quantum better |
| 10 | +0.0002 | [−0.0069, +0.0081] | — |
| 20 | −0.0050 | [−0.0111, +0.0016] | — |
| 50 | **−0.0080** | [−0.0126, −0.0032] | classical better |
| 100 | **−0.0074** | [−0.0109, −0.0038] | classical better |
| **full** | −0.0068 | [−0.0175, +0.0053] | no difference |

H-P1 supported. H-P2 slope **−0.00462** [−0.00844, −0.00127], supported.

**The classical advantage plateaus at about −0.007** rather than widening with
unlimited data. At full data the pooled contrast is not significant; the two
multi-class sets carry what signal there is (BloodMNIST −0.0419, d = −1.28;
PathMNIST −0.0141, d = −4.35) while both binary sets are flat.

**But it does not generalise.**

| Leave out | Δ(5) | 95% CI |
|---|---|---|
| nothing | +0.0142 | [+0.0018, +0.0256] |
| **BloodMNIST** | **+0.0041** | **[−0.0057, +0.0148]** ← gone |
| BreastMNIST | +0.0155 | [+0.0014, +0.0283] |
| PathMNIST | +0.0202 | [+0.0041, +0.0364] |
| PneumoniaMNIST | +0.0171 | [+0.0030, +0.0303] |

**d=8** (`quantum_vqc` − `low_rank`, 48 params each, exact parity):
Δ negative at every regime; significant against quantum at n=10 (−0.0114) and
n=50 (−0.0110); slope **+0.0015**, no crossover.

**d=16** (binary datasets): no difference at any regime.

**Conclusion.** A small advantage at d=4, n=5, carried by one of four datasets,
that reverses at d=8 and vanishes at d=16.

---

## R3 — Every proposed mechanism is refuted

| Mechanism | Test | Result |
|---|---|---|
| Superposition | analytic + numeric | **refuted**, R1 |
| Capacity restriction | 1,000-run classical sweep, ranks 8→72 params | **refuted** — 19/20 cells null, survivor dies under BH |
| Impoverished readout | 10 observables vs 4, padded control | see R4 — helps, but not where the advantage is |

The capacity sweep is the decisive one: varying restriction directly, in a
purely classical head, reproduces nothing.

---

## R4 — Richer readout helps, at matched parameters

`quantum_rich` − `quantum_rich_padded`. Identical circuit, identical 24
parameters, identical classifier width. Only difference: whether the 6 extra
columns carry real ⟨XᵢXⱼ⟩ or duplicated singles.

| n/cls | Δ | 95% CI | verdict |
|---|---|---|---|
| 5 | +0.0142 | [−0.0033, +0.0322] | — |
| **10** | **+0.0139** | **[+0.0046, +0.0250]** | rich better |
| **20** | **+0.0139** | **[+0.0043, +0.0235]** | rich better |
| **50** | **+0.0075** | **[+0.0003, +0.0155]** | rich better |
| 100 | +0.0073 | [−0.0012, +0.0147] | — |

**Measuring more of the state helps.** The padded control rules out classifier
capacity as the explanation. This is the only positive quantum-side result that
replicates across regimes.

---

## R5 — The learned bottleneck dominates everything

Trainable capacity, d=4, binary:

| component | params | share |
|---|---|---|
| bottleneck `Linear(256,4)` | 1,028 | **97%** |
| head | 24 | 2% |
| classifier | 10 | 1% |

Freezing the projection (`bn=learned` − `bn=pca`), 900 runs:

| dataset, n | arm | Δ | Cohen's d |
|---|---|---|---|
| Blood, 100 | linear | +0.0974 | **+6.73** |
| Blood, 20 | linear | +0.1714 | +3.51 |
| Breast, 10 | linear | +0.2931 | +3.06 |
| Blood, 10 | quantum | +0.1276 | +2.66 |

**The largest and most consistent effect in the project** — an order of
magnitude bigger than any head-level difference. The head, quantum or classical,
is not what the "frozen backbone" protocol measures.

The scarcity crossover is present under a learned projection (slope −0.0145) and
**absent under both frozen policies** (PCA −0.0119 no crossover; random +0.0049).

---

## R6 — Unitarity bounds the head, and the bound is dimension-independent

Architecture-level, 20 parameter draws:

| arm | L(d=4) | L(d=8) | L(d=16) | \|out\|max (d=16) |
|---|---|---|---|---|
| **quantum_vqc** | 1.515 | **1.393** | **1.203** | **0.921** |
| quantum_reupload | 1.63 | 1.489 | 1.280 | 0.910 |
| low_rank | 1.14 | 1.121 | 1.079 | 1.483 |
| matched_param_fullrank | 2.38 | 2.336 | 2.072 | 4.945 |
| fourier_rff | 3.86 | 4.758 | **5.671** | 3.883 |
| matched_param | 3.74 | 4.501 | **5.240** | 8.731 |

Quantum output stays **bounded below 1 at every dimension** — exactly what
`v_i = ⟨ψ|U†X_iU|ψ⟩` with unitary U predicts. Its Lipschitz constant is flat or
**falling** with d, while `fourier_rff` grows 3.86 → 5.67 and `matched_param`
reaches 8.73 unbounded.

Ratio to quantum_vqc grows with dimension: `fourier_rff` 2.55× → 3.42× → 4.72×.

**Theory predicted it; measurement confirms it at three dimensions.**

---

## R7 — Failure under noise is calibration, not ranking

Input noise, 1,400 runs. PathMNIST n=20, σ=0.20:

| arm | AUC | Macro-F1 | trained L |
|---|---|---|---|
| fourier_rff | 0.9658 → 0.6125 | 0.7188 → **0.1303** | 12.79 |
| quantum_vqc | 0.9251 → 0.6199 | 0.5998 → **0.0869** | 1.59 |

F1 collapses far harder than AUC across every arm and dataset: ranking survives,
the decision threshold does not. Pooled AUC contrast at σ=0.20 is significant
only at n=50 and is not robust to leave-one-out.

**Depolarizing noise, derived and confirmed.** Single-qubit depolarizing gives
`⟨X_i⟩ → c⟨X_i⟩` with `c = 1 − 4p/3`, identical for every i and every input. The
binary decision score `l₁ − l₀ = c(w₁−w₀)·v + (b₁−b₀)` is a monotone
transformation, so **AUC is exactly invariant** — measured identical to four
decimals at p = 0.000…0.050. Multi-class moves only via the softmax
(0.8382 → 0.8385). ECE rises 0.0865 → 0.0974; prob_std falls 0.1947 → 0.1893.

Shot noise does degrade: 0.7590 → 0.6976 at 64 shots; ≤1% loss at 1024.

---

## R8b — Fairness ablations, all three complete

**Depth (L ∈ {1,2,4}, 600 runs).** L=4 beats L=2 on BloodMNIST at 4 of 5 regimes
(to +0.0278, d = +1.35) and PathMNIST at 2 of 5; null on both binary sets.
**But 3·L·d = 48 parameters at L=4 against the control's 24** — depth helps by
adding parameters, so L=4 cannot enter the matched comparison. L=2 is the only
depth at which parity holds, now justified by data rather than by assumption.

**Angle scale (π vs π/2, 800 runs).** Null for `quantum_vqc` almost everywhere;
one cell favours π (Blood n=100, +0.0194). For `matched_param_fullrank` π is
*worse* in four cells. **The pre-registered π/2 did not handicap the quantum
arm**, and π would have hurt the classical control.

**tanh (2,000 runs).** Removing tanh helps `mlp` (+0.0412, d = +0.95),
`low_rank` (+0.0463, d = +1.18), `fourier_rff` and `matched_param_fullrank`
consistently across regimes; only `linear` prefers it, negligibly.

**This strengthens the main result.** The shared tanh is required only by the
quantum arm — RY is 2π-periodic, so unbounded z destroys injectivity. Imposing it
on every arm for fairness measurably costs the classical arms up to 0.046 AUC,
and they still won. Remove the handicap and the quantum arm looks worse, not
better.

---

## R8 — Structural claims proven

| Claim | Evidence |
|---|---|
| Frozen backbone unchanged | 0 params, 0 buffers, max delta 0.00e+00, six arms |
| Test not vacuous | negative control without `set_bn_eval()`: 45 buffers drift |
| Gradients reach encoder from every head | layer3 displacement 0.53, quantum included |
| Frozen blocks clean | no gradient in frozen regime |
| Parameter parity 24/24/24 | unit test |
| Predictions faithful | integrity 0.00e+00 across 4,500+ runs |

---

## The paper

**Not** "quantum wins" and **not** "quantum loses". The defensible claim:

> A parameter-matched variational quantum head shows a small advantage under
> extreme scarcity at one bottleneck dimension, carried by one of four datasets,
> which reverses at d=8 and vanishes at d=16. None of the mechanisms usually
> invoked explains it: the output is classically constructible, capacity
> restriction reproduces nothing, and the effect requires a learned bottleneck.
> Two properties *are* robust and mechanistic — unitarity bounds the head's
> output and Lipschitz constant independently of dimension, and richer
> measurement of the same state improves accuracy at matched parameters.
> Separately, the standard "frozen backbone" protocol does not isolate the head:
> the projection between backbone and head holds 97% of trainable capacity, and
> freezing it changes results by up to Cohen's d = 6.73.

**Two contributions apply beyond this paper:** the capacity-accounting flaw in
frozen-backbone protocols, and the demonstration that seed-level statistics
overstate effects on few-shot medical benchmarks.

---

## Remaining — no compute

| | |
|---|---|
| `generate_paper_plots.py` rewrite | last unwritten code |
| Amendments 3a, 9, 10, 11 | analysis_plan.md |
| `paper/main.tex` | rewrite |

Every experiment is complete. Integrity `0.00e+00` on every post-float32
namespace: 01_frozen_tuned (1,600), 12_bottleneck (900), 10_capacity (1,000),
14_readout (600), 03_robustness (1,400), 15_dim8 (800), 16_dim16 (200),
18_depth (600), 19_angle (800), 20_tanh (2,000), 17_fulldata (80).

## Prior art — read before writing

The bottleneck-dominance finding has a close precedent: Chen & Kuo,
arXiv:2504.05336, later revision, "Is the gain quantum, or just a compact
bottleneck?" — a rank-2 bilinear + tanh control, 40 params vs the PQC's 36, on
synthetic time-series regression with 5 seeds. It must be cited. See
`PAPER_OUTLINE.md` §1 for the full list of precedents and what remains ours.
