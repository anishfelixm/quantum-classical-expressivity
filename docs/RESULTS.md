# RESULTS — Final

**11 October 2026. Added 13 October: reporting tables (§6), the exploratory E8
latent-saturation reproduction (§3) and the pre-plan compression check (§3).**
Supersedes the
4 October version, which called the restriction mechanism "refuted" — the
powered follow-up (Amendment 17) does not support that word (§2).

Every number below comes from a nested bootstrap (test images and seeds
resampled within each cell, datasets weighted equally, B = 2000, +1-corrected
p-values) unless stated. Stored predictions reproduce recorded metrics to
0.00e+00 in every namespace used.

Three evidential tiers, never mixed:

- **Confirmatory** — the 23 tests of `analysis_plan.md` §4, one BH correction
  (m = 23), computed by `13_family_table.py`.
- **Pre-specified follow-up** — Amendments 16 and 17, each written before its
  run, each with its own correction. They change no confirmatory verdict.
- **Exploratory** — everything else, corrected within each analysis.

The analysis plan was fixed in version control before the confirmatory run.
Amendments are dated; 3a and 9–11 were written retrospectively and say so.
Describe this as a *pre-specified analysis plan*, not as a third-party
pre-registration.

---

## 1. Confirmatory — 23 tests, BH-FDR m = 23

**14 supported · 1 opposite to prediction · 1 difference without prediction · 7 not supported**

| Test | Estimate | 95% CI | p_adj | Verdict |
|---|---|---|---|---|
| **H-P1** VQC − control, Δ(5) | **+0.0142** | [+0.0025, +0.0256] | 0.026 | **supported** |
| **H-P2** slope on log₂ n | **−0.0046** | [−0.0070, −0.0022] | 0.002 | **supported** |
| H-S1 re-upload − VQC, n=5 | +0.0039 | [−0.0059, +0.0133] | 0.41 | not supported (pred. −) |
| H-S1 n=10 | +0.0090 | [+0.0029, +0.0154] | 0.006 | **opposite** (pred. −) |
| H-S1 n=20 | +0.0125 | [+0.0077, +0.0175] | 0.002 | difference (no pred.) |
| H-S1 n=50 | +0.0102 | [+0.0060, +0.0140] | 0.002 | **supported** |
| H-S1 n=100 | +0.0103 | [+0.0078, +0.0132] | 0.002 | **supported** |
| H-S2 VQC − Fourier, n=5 | −0.0095 | [−0.0193, +0.0003] | 0.084 | not supported |
| H-S2 n=10 | −0.0187 | [−0.0249, −0.0123] | 0.002 | **supported** |
| H-S2 n=20 | −0.0212 | [−0.0261, −0.0157] | 0.002 | **supported** |
| H-S2 n=50 | −0.0203 | [−0.0239, −0.0166] | 0.002 | **supported** |
| H-S2 n=100 | −0.0161 | [−0.0194, −0.0127] | 0.002 | **supported** |
| H-S3 VQC F1-vs-AUC gap, σ=0.05 | +0.356 | [+0.329, +0.382] | 0.002 | **supported** |
| H-S3 σ=0.10 | +0.401 | [+0.374, +0.425] | 0.002 | **supported** |
| H-S3 σ=0.15 | +0.382 | [+0.353, +0.408] | 0.002 | **supported** |
| H-S3 σ=0.20 | +0.365 | [+0.340, +0.387] | 0.002 | **supported** |
| H-S4 encoder absorbs head | −0.0048 | [−0.0188, +0.0014] | 0.12 | not supported |
| H-S5a restriction, Δ₀(5) | −0.0124 | [−0.0380, +0.0114] | 0.34 | not supported — underpowered, see §2 |
| H-S5b restriction, slope | +0.0029 | [−0.0017, +0.0076] | 0.26 | not supported — see §2 |
| H-S6 frozen PCA, Δ(5) | −0.0283 | [−0.1085, +0.0479] | 0.51 | not supported — underpowered, see §2 |
| H-S6 frozen random, Δ(5) | −0.0545 | [−0.1232, +0.0153] | 0.15 | not supported — underpowered, see §2 |
| H-S7a rich − padded readout | +0.0114 | [+0.0065, +0.0163] | 0.002 | **supported** |
| H-S7b rich − single readout | +0.0111 | [+0.0065, +0.0156] | 0.002 | **supported** |

**Anchors.** H-P1's observed Δ(5) reproduces `04` exactly (+0.0142). `04`'s
H-P2 interval, recomputed by its own procedure, reproduces exactly
([−0.00844, −0.00127]). The nested H-P2 interval is narrower because it treats
the four datasets as fixed — so it supports inference about *these* datasets,
and generalisation across datasets rests on the leave-one-out in §3.

### What the decision rule licenses

The rule, fixed before the data: *H-P1 and H-P2 supported → "scarcity-dependent
quantum advantage, attributed to function-class restriction acting as a
regulariser."* Both hold. The paper therefore claims the scarcity-dependent
advantage, and reports alongside it that the attribution was tested twice and
**not established** (H-S5 and its follow-up, §2). Wording: *an advantage of the
VQC head over a parameter-matched classical head* — the head computes a
classical function (§4), so "quantum advantage" would overstate it.

---

## 2. Pre-specified follow-ups — Amendments 16 and 17

Both confirmatory mechanism tests, H-S5 and H-S6, originally ran outside the
primary's protocol (default rate 1e-3; 10 and 5 seeds). Each was rerun under
the primary's protocol — 40 seeds, rate 1e-2, same cells — with the design and
prediction written before the run.

### H-S6 follow-up (Amendment 16a) — does the advantage survive a frozen projection?

| Bottleneck | n=5 (test) | n=10 | n=20 | n=50 | n=100 |
|---|---|---|---|---|---|
| learned (= H-P1) | **+0.0142** | +0.0002 | −0.0050 | −0.0080 | −0.0074 |
| frozen PCA | **−0.0301** | −0.0212 | −0.0164 | −0.0112 | −0.0090 |
| frozen random | **−0.0190** | −0.0320 | −0.0326 | −0.0287 | −0.0242 |

Test, BH m = 2: PCA −0.0301 [−0.0421, −0.0179], p_adj 0.002; random −0.0190
[−0.0321, −0.0063], p_adj 0.005. **Prediction confirmed** ("not positive"), and
exceeded: with the projection frozen, the VQC head is significantly *worse*
than the matched head at every n, under both policies.

The negative sign replicates the original H-S6 at a different learning rate
(1e-3: PCA −0.028, random −0.055), which answers the stated limitation that
1e-2 was not re-tuned for frozen bottlenecks.

**Conclusion.** The n=5 advantage exists only when a learned projection adapts
upstream. It is a property of the projection–VQC system, not of the quantum
head's function class in isolation.

### H-S5 follow-up (Amendment 17) — does restriction help under scarcity?

`low_rank` rank 0 (8 parameters) − rank 8 (72 parameters), d=4, learned
bottleneck:

| n | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ₀ | **+0.0097** | **+0.0118** | +0.0041 | +0.0019 | **+0.0059** |
| 95% CI | [+.0007, +.0187] | [+.0035, +.0205] | [−.0031, +.0111] | [−.0040, +.0076] | [+.0016, +.0102] |

| Test (BH m = 2) | Estimate | 95% CI | p_adj | Rule's label |
|---|---|---|---|---|
| H-S5a Δ₀(5) > 0 | +0.0097 | [+0.0007, +0.0187] | 0.074 | no restriction effect |
| H-S5b slope < 0 | −0.0016 | [−0.0037, +0.0005] | 0.143 | no restriction effect |

**Prediction ("no restriction effect") — not cleanly confirmed.** Neither test
survives correction. But Δ₀(5) moved from −0.0124 to +0.0097, its unadjusted
interval excludes zero (p = 0.037), and the smaller head leads at every n.

**What this licenses, and nothing stronger:**

1. **No scarcity-dependent restriction effect** — the slope is null in both runs.
   The prediction the analysis plan attached to the attribution (restriction
   helps *more* when data are scarce) is not supported.
2. **A possible scarcity-independent smaller-head benefit**, up to about
   0.019 AUC, not established after correction and not excluded.
3. **Restriction is neither established nor excluded** as a contributor to the
   +0.0142 advantage, which lies inside that range.
4. The contrast is classical-vs-classical. It shows how restriction behaves for
   a classical head under this protocol; it does not measure the VQC directly.

The 4 October version's "restriction refuted" was too strong even for the
original H-S5, whose interval never excluded an effect of this size.

### E7 — function-class control for re-uploading (Amendment 16b)

Re-upload VQC (24 parameters) − direct fit over {−2..2}^d (2,500 parameters):

| n | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ | **+0.0150** | **+0.0148** | +0.0029 | −0.0002 | **−0.0038** |
| 95% CI | [+.0022, +.0279] | [+.0038, +.0264] | [−.0041, +.0105] | [−.0067, +.0071] | [−.0063, −.0013] |

Prediction (Δ < 0 at n ≥ 10) **wrong** at n=10; held only at n=100. Not
parameter-matched: a 24-parameter model beating a 2,500-parameter fit of the
same classical function class at 5–10 shots, and losing at 100, is a
bias–variance pattern. Motivated by data, labelled exploratory in its design
note, reported with its pre-written prediction.

### How the two follow-ups fit together

The H-S6 follow-up says *where* the advantage is: in the learned projection
working with the VQC. The H-S5 follow-up says *what is not shown*: that
restriction is the reason. A projection that adapts to a small, bounded head
could favour that head for reasons other than restriction; these data do not
separate the two, and the paper says so.

---

## 3. Exploratory

Leave-one-out intervals are from the nested bootstrap (the method of every other
number here), computed by `generate_paper_plots.py`; the point estimates equal
those from `04`, whose dataset-level intervals were slightly narrower
([−0.0057, +0.0148] without BloodMNIST). The method was fixed before the nested
values were seen.

### Scope of the primary effect

| Analysis | Result |
|---|---|
| Leave-one-dataset-out, Δ(5), nested bootstrap | without BloodMNIST **+0.0041 [−0.0071, +0.0155]**; without Breast +0.0155 [+0.0022, +0.0293], Path +0.0202 [+0.0056, +0.0344], Pneumonia +0.0171 [+0.0030, +0.0310] — the effect is carried by BloodMNIST |
| Per dataset, Δ(5) | BloodMNIST **+0.0446 [+0.0117, +0.0770]**; Breast +0.0104, Path −0.0038, Pneumonia +0.0056, all intervals spanning 0 |
| Per dataset, n ≥ 20 | PathMNIST negative at n = 20, 50, 100 (−0.020, −0.020, −0.018; intervals exclude 0); BloodMNIST negative at 50 and 100 |
| Pooled Δ(n), per-n rows of H-P2 | +0.0142, +0.0002, −0.0050, **−0.0080 [−0.0124, −0.0032]**, **−0.0074 [−0.0110, −0.0038]** — the matched head is ahead at n ≥ 50 (descriptive) |
| d=8, VQC − low_rank (48 each, 10 seeds) | n=5 −0.0132 [−0.0283, +0.0044]; n=10 **−0.0114**; n=50 **−0.0110** — reverses |
| d=16, binary sets (96 each) | no difference at any n |
| Full data (5 seeds) | pooled −0.0068 [−0.0175, +0.0053]; plateau ≈ −0.007 |

### Re-uploading and ansatz

| Analysis | Result |
|---|---|
| Re-upload − control (not pre-specified) | **+0.0181, +0.0092, +0.0074**, +0.0022, **+0.0029**; CI excludes 0 at four of five n |
| Ansatz: Basic − Strong (24 each) | negative throughout; after within-analysis BH, significant at n=100 (−0.0180, p_adj 0.005). The primary used the better circuit |

### Capacity

| Quantity | Value |
|---|---|
| Trainable parameters, d=4, binary | projection 1,028 (**97%**), head 24, classifier 10 |
| Learned − frozen PCA (12_bottleneck, pooled over 3 arms) | +0.187 at n=5 → +0.082 at n=100; slope −0.027 [−0.043, −0.011] |
| Learned − frozen random | +0.180 → +0.202; slope +0.004 [−0.007, +0.016] |

### Noise and robustness

| Analysis | Result |
|---|---|
| Classical control's F1-vs-AUC gap | +0.303, +0.335, +0.339, +0.317 |
| Paired VQC − control gap | **+0.053, +0.065, +0.044, +0.048** — all CIs exclude 0 |
| Head sensitivity normalised by output range (L / \|out\|max) | VQC 1.55 / 1.53 / 1.31 (d=4/8/16) vs primary control 0.58 / 0.49 / 0.42 |

The F1 collapse under sensor noise is generic to small heads; the VQC adds a
modest, consistent excess. **It is not a calibration failure** (added 13 Oct,
`15_reporting_tables.py`, descriptive): the VQC's expected calibration error is
the lowest or within 0.005 of the lowest of all seven heads at every σ
(§6.4) — 0.226 vs 0.263 for the matched head at σ = 0.05, 0.342 vs 0.370 at
σ = 0.20. What collapses is the argmax decision while the ranking holds: a
decision-threshold effect. H-S3's verdict, which tests the F1-vs-AUC gap, is
unchanged; the registered wording "a calibration failure" is not supported and
the paper says "decision-threshold degradation". Bounded output did not buy robustness: the VQC's raw
Lipschitz constant is small because unitarity caps its output at 1, but the
classifier rescales it, and relative to its own range the VQC is about three
times as sensitive as the control.

### Hardware feasibility (`07`, default rate 1e-3, quantum only)

AUC retention at 1,024 shots: 0.989–1.005. Depolarising noise **applied once per
qubit at the end of the circuit** — the model `07` simulates for the
single-block VQC — leaves binary AUC exactly invariant: the readout contracts
uniformly, ⟨Xᵢ⟩ → (1 − 4p/3)⟨Xᵢ⟩, and the binary score c·(w₁ − w₀)·v + (b₁ − b₀)
is a monotone transform of the noiseless one. Multi-class AUC moves only through
the softmax (0.8382 → 0.8385 at p = 0.05).

**Scope, added 13 October.** The invariance is a property of end-of-circuit
noise, not of depolarising noise in general. An independent numpy simulation
(`tests/test_independent_theory.py`) with noise after *every* layer gives
input-dependent contraction factors (0.81–0.87 at p = 0.05) and reorders 24 of
1,770 test pairs. The paper names the noise model and states this. `07`'s
docstring writes the channel as ρ → (1 − p)ρ + p·I/2, which would contract by
(1 − p); PennyLane's `DepolarizingChannel` is the Pauli form
ρ → (1 − p)ρ + (p/3)(XρX + YρY + ZρZ), which contracts by (1 − 4p/3) — the
factor quoted. Numbers unaffected; wording only.

### Compression premise (`06_premise_check.py`; pre-plan diagnostic)

Added to this document 13 October 2026. The run is older than the analysis plan:
commit `538d8361`, completed by 12 August 2026 (`HYPOTHESES.md` records it as
complete on that date). It used a linear head, 3 seeds, and the training
protocol of that date, before the plan and its amendments (no per-arm learning
rate tuning). Log: `logs_premise.txt`.

The architecture and training are the same in every row; only the projection
width d changes. Mean AUC:

| Dataset | n | d=4 | d=256 | d=256 − d=4 |
|---|---|---|---|---|
| BloodMNIST | 100 | 0.9786 | 0.9886 | +0.0099 |
| BloodMNIST | full | 0.9901 | 0.9915 | +0.0015 |
| BreastMNIST | 100 | 0.8395 | 0.8328 | −0.0067 |
| BreastMNIST | full | 0.9025 | 0.8844 | −0.0181 |
| PathMNIST | 100 | 0.9385 | 0.9519 | +0.0134 |
| PathMNIST | full | 0.9709 | 0.9854 | +0.0144 |
| PneumoniaMNIST | 100 | 0.9437 | 0.9437 | +0.0000 |
| PneumoniaMNIST | full | 0.9733 | 0.9730 | −0.0002 |

Mean gap +0.0018. **Compressing to d=4 costs almost nothing in AUC** at n=100
and at full data. The only consistent cost is on PathMNIST (+0.013 to +0.014).
There, d=16 reaches 0.9876 at full data, so 1 − AUC falls from 2.9 % to 1.2 %.
This result contradicts the conference paper's "Bottleneck Gap" more directly
than anything else in the project: there is no large compression penalty for a
head to recover. It was not run at n ≤ 50. The manuscript reports it as
exploratory context and does not frame d=4 as a severe constraint.

### Fairness ablations (10 seeds)

| Axis | Result |
|---|---|
| Depth L ∈ {1,2,4} | on BloodMNIST both L=1 and L=4 beat L=2 at n ≥ 10; null elsewhere. L=2 is the only depth with exact parity (24 = 24) |
| Angle scale π vs π/2 | π helps the VQC in 3/20 cells (+0.012 to +0.019), hurts the control in 5/20 — π/2 did not handicap the VQC |
| tanh removed (classical arms) | helps `mlp` and `low_rank` on multi-class sets; the primary control mostly unaffected |

### Latent saturation — E8 (13 October 2026; descriptive, no test)

**Why it was run.** The function-class figure's run showed two of its four
latent coordinates pinned at the tanh boundary. E8 asks whether that is typical,
and whether it differs between the primary arms.

**What was run.** `src/17_latent_saturation.py`, namespace `28_latent_saturation`.
It retrained 160 primary confirmatory runs from their own records:
`quantum_vqc` and `matched_param_fullrank`, four datasets, n ∈ {5, 100}, the
first 10 confirmatory seeds. **All 160 reproduced their recorded test AUC
exactly** (largest difference 0.0), so the latents are those of the reported
models. A coordinate counts as *pinned* when at least half the test latents lie
within 5 % of the boundary (|z| > 0.95·π/2). *Active* = 4 − pinned. The two
predictions and these definitions were written in the script's docstring and
committed (`8817335`) before any run.

Mean active coordinates (of 4), 10 seeds per cell:

| Dataset | n=5 matched | n=5 VQC | VQC − matched | n=100 matched | n=100 VQC | VQC − matched |
|---|---|---|---|---|---|---|
| BloodMNIST | 3.8 | 2.8 | −1.0 | 3.1 | 2.1 | −1.0 |
| BreastMNIST | 3.9 | 2.8 | −1.1 | 3.0 | 2.9 | −0.1 |
| PathMNIST | 4.0 | 2.7 | −1.3 | 3.3 | 2.6 | −0.7 |
| PneumoniaMNIST | 3.9 | 2.9 | −1.0 | 3.1 | 2.9 | −0.2 |

**P1 — "saturation is not VQC-specific" (the arms differ by < 1 coordinate in
every cell): not supported.** The VQC keeps fewer coordinates active in all 8
cells. At n=5 the gap is 1.0 to 1.3 in every dataset. At n=100 it is 1.0 on
BloodMNIST and 0.1 to 0.7 elsewhere. Three cells sit at exactly −1.0, which
fails "< 1" only on the boundary. BreastMNIST n=5 (−1.1) and PathMNIST n=5
(−1.3) fail outright, so the verdict does not depend on where the boundary is
drawn.

**P2 — "both arms saturate more at n=5 than at n=100": not supported** (7 of 8
rows). The matched head saturates *more with more data*: 3.8 to 4.0 active at
n=5, 3.0 to 3.3 at n=100, in every dataset. The VQC is roughly flat (2.6 to
2.9; BloodMNIST n=100 2.1). The arm gap is largest at n=5 because the matched
head hardly saturates there, not because the VQC saturates more.

**Saturation does not track the advantage.** Per seed at n=5, VQC − matched AUC
against VQC − matched active coordinates gives Spearman ρ = +0.30, +0.24,
−0.12, −0.22 (Blood, Breast, Path, Pneumonia): small, of inconsistent sign, and
not tested. Ten seeds could not detect a weak association. Across datasets,
PathMNIST has the most VQC saturation at n=5 (2.7 active) and no VQC advantage
(0.834 vs 0.838, §6.1).

**Learning rate is not the explanation.** The projection is trained at the
arm's `lr_head`. Every primary record at n ∈ {5, 100} in `01_frozen_tuned`
(320 per arm, 40 seeds, including the 10 E8 retrained) used 0.01 for both arms,
as `artifacts/lr_selection.json` selected.
The arms differ only in the head itself: its function, the gradient it returns
to the projection, and its initialisation. That difference is the treatment,
not a confound.

**What E8 licenses.** Training the projection through the VQC head leaves about
one fewer latent coordinate in use at n=5 than training it through the matched
head, in every dataset (exploratory, one reproduction per run, no test). It
does not show that saturation causes the n=5 advantage, and the seed-level and
dataset-level patterns give no support to that reading.

---

## 4. Established structural results

| Claim | Evidence |
|---|---|
| Single-encoding VQC output is classical | in the 3^d trigonometric span, residual 1e-16; wrong-frequency control fails (0.908) |
| Re-uploading output is classical | in the 5^d span, residual < 1e-8 in the test suite, 2.21e-7 in `verify_v40`; the truncated {−1,0,1}^d basis fails (0.469) |
| Independent re-derivation | plain-numpy rebuild of the circuit, no PennyLane or project code: 3^d span 2×10⁻¹⁵, two-qubit readout 2×10⁻¹⁵, re-upload 5^d span 4×10⁻¹⁵ (3^d span fails, 0.27), end-of-circuit depolarising contraction exact to 2×10⁻¹⁶ (`tests/test_independent_theory.py`) |
| Parity | 24 = 24 = 24 at d=4; 48 and 96 at d=8, 16 via `low_rank` |
| Quantum gradient variance (`MATH_VERIFICATION.md` §7; batch 32) | falls with qubit count at every depth: L=4 12.96 → 0.996 → 0.210 (d = 4/8/16; ~62×). It falls with depth at d=8 and 16, **but not at d=4**, where it rises (10.45 → 11.21 → 12.96 for L = 1/2/4). That document's "monotone decay in both n and L" is wrong for d=4. The script and log behind this table must be identified before the paper cites it (`WORK_REMAINING.md`) |
| Frozen backbone is frozen | 0 parameters, 0 buffers changed; negative control drifts 45 buffers |
| Adaptive encoder adapts, for every arm | backbone gradient non-zero for VQC and control |
| Learning rates | every arm's optimum at the registered grid's edge; extension changed only `linear` — primary pair invariant |

---

## 5. The findings, in one list

1. A small advantage of the VQC head over a parameter-matched classical head at
   five labels per class, shrinking with data — **confirmatory**.
2. It is carried by one dataset, reverses at d=8, is null at d=16 and absent at
   full data — **exploratory scope**.
3. It exists only with a learned projection; frozen, the VQC head loses at every
   n — **pre-specified follow-up**.
4. Restriction as its cause: not established, not excluded; no scarcity
   dependence — **confirmatory + pre-specified follow-up**.
5. The VQC loses to a direct fit over its own function class at n ≥ 10 —
   **confirmatory**.
6. Re-uploading helps at n ≥ 10 (opposite to its pre-written prediction) —
   **confirmatory**; it beats a 2,500-parameter fit of its own class at 5–10
   shots and loses at 100 — **follow-up, not parameter-matched**.
7. Richer readout helps — **confirmatory**.
8. Under sensor noise F1 collapses while AUC holds, for every small head; the
   VQC's excess is modest — **confirmatory + exploratory**. It is a
   decision-threshold effect, not miscalibration: the VQC's ECE is among the
   lowest of seven heads — **descriptive**. Bounded output is not robustness —
   **exploratory**.
9. The projection trained through the VQC head leaves about one fewer of four
   latent coordinates unsaturated at n=5 than under the matched head, at an
   identical learning rate; this does not track the AUC advantage. Both
   pre-written predictions were wrong — **exploratory (E8)**.
10. Compressing the 256 features to d=4 costs almost no AUC at n=100 and at full
    data (mean gap +0.0018; PathMNIST the exception at about +0.014). The
    conference paper's "Bottleneck Gap" does not exist at those data sizes —
    **exploratory, pre-plan protocol, 3 seeds**.

---

## 6. Reporting tables — 13 October 2026

Computed by `src/15_reporting_tables.py` on the cluster from every run record
and stored prediction (integrity: recorded vs recomputed AUC, max 2.2×10⁻¹⁶
over 4,000 primary runs). Definitions were fixed in that file's docstring
before any of these numbers existed. All descriptive; none enters the
confirmatory family. Full tables: `paper/tables/*.csv`.

### 6.1 Absolute AUC, primary arms, n = 5 (mean over 40 seeds)

| Dataset | Matched head | VQC |
|---|---|---|
| BloodMNIST | 0.742 (SD 0.091) | 0.786 (SD 0.064) |
| BreastMNIST | 0.658 (SD 0.073) | 0.668 (SD 0.068) |
| PathMNIST | 0.838 (SD 0.045) | 0.834 (SD 0.028) |
| PneumoniaMNIST | 0.889 (SD 0.051) | 0.895 (SD 0.033) |

**Correction to the plan's limitation 4**, which says both arms sit at "AUC
0.60–0.68 at n = 5": that range came from the diagnostic runs. In the
confirmatory data only BreastMNIST is that low; the others are 0.74–0.90.
Recorded in the plan's results section; the original text is not edited.

### 6.2 Checkpoint chosen by validation F1 instead of AUC (Amendment 2)

Pooled Δ, seed-level 95% interval:

| | n = 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| VQC − matched, AUC-selected | +0.0142 | +0.0002 | −0.0050 | −0.0080 | −0.0074 |
| VQC − matched, F1-selected | +0.0168 [+.0036, +.0299] | +0.0045 | −0.0051 | −0.0091 | −0.0067 |

Same sign at every n for the primary contrast, H-S1 and H-S2. The conclusions do
not depend on the selection rule.

### 6.3 Effect sizes (§5) and practical significance

Paired d_z over seeds, pooled as each test pools. Tests in AUC units with
|estimate| < 0.01 are **negligible** under §5.

| Test | Estimate | d_z | Seeds | Verdict | §5 |
|---|---|---|---|---|---|
| H-P1 | +0.0142 | +0.36 | 40 | supported | — |
| H-P2 | −0.0046 | −0.56 | 40 | supported | slope |
| H-S1 n=5 / 10 / 20 / 50 / 100 | +.0039 / +.0090 / +.0125 / +.0102 / +.0103 | +0.17 / +0.50 / +0.88 / +0.95 / +1.38 | 40 | ns / **opposite** / difference / supported / supported | n=5, **n=10 negligible** |
| H-S2 n=5 … 100 | −.0095 / −.0187 / −.0212 / −.0203 / −.0161 | −0.34 / −1.38 / −1.54 / −2.28 / −1.89 | 40 | ns / supported ×4 | n=5 negligible |
| H-S3 σ = .05 / .10 / .15 / .20 | +.356 / +.401 / +.382 / +.365 | +10.7 / +9.1 / +7.2 / +8.7 | 10 | supported | — |
| H-S4 | −0.0048 | −0.93 * | 10 | ns | negligible |
| H-S5a / b | −0.0124 / +0.0029 | −0.45 / +0.55 | 10 | ns | — |
| H-S6 PCA / random | −0.0283 / −0.0545 | −0.22 / −0.73 | 5 | ns | — |
| H-S7a / b | +0.0114 / +0.0111 | +1.73 / +1.42 | 10 | supported | — |

\* H-S4's d_z uses the per-seed analogue |Δ_frozen,s| − |Δ_adaptive,s|; taking
the absolute value per seed adds seed noise to both terms, so it is not
comparable with the registered statistic (absolute value after averaging).
Reported with that warning, not interpreted.

**The one "opposite to prediction" result (H-S1, n = 10) is significant but
negligible** (+0.0090). The paper says so. H-P1's estimate exceeds 0.01, but
its interval reaches 0.0025: the advantage may be practically negligible.

### 6.4 Noise, all seven heads (means over 200 runs per head and σ)

| σ | G: VQC | matched | low-rank | Fourier | linear | 2-local | padded | ECE: VQC | matched | low-rank | Fourier | linear | 2-local | padded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | **.356** | .303 | .297 | .320 | .329 | .340 | .348 | **.226** | .263 | .252 | .330 | .313 | .268 | .244 |
| 0.10 | **.401** | .335 | .348 | .361 | .381 | .365 | .377 | **.289** | .325 | .304 | .403 | .396 | .334 | .295 |
| 0.15 | **.382** | .339 | .338 | .356 | .376 | .362 | .366 | .319 | .354 | .337 | .446 | .439 | .363 | **.318** |
| 0.20 | **.365** | .317 | .306 | .331 | .347 | .351 | .347 | .342 | .370 | .352 | .467 | .460 | .386 | **.337** |

The VQC has the largest F1-vs-AUC gap of all seven heads and the lowest or
near-lowest calibration error. The Fourier and linear heads are the worst
calibrated.

### 6.5 d = 8, d = 16, full data (descriptive; nested intervals in §3)

| | n = 5 | 10 | 20 | 50 | 100 | Seeds |
|---|---|---|---|---|---|---|
| d = 8, VQC − low-rank (48 each) | −0.0132 | −0.0114 | −0.0070 | −0.0110 | −0.0050 | 10 |
| d = 16, VQC − low-rank (96 each, binary sets) | +0.0032 | +0.0093 | −0.0052 | +0.0034 | +0.0037 | 5 |
| full data, VQC − matched | — | — | — | — | — | 5; pooled −0.0068 |

### 6.6 Provenance — the plan existed before every experiment

Git history of `docs/analysis_plan.md`: first committed **d6e37db, 17 Aug 2026**
(the plan says "written 14 August"; git can verify only from 17 August);
Amendments 4–8 by **ec5cae6, 27 Aug**; unchanged until bd833ba, 30 Sep.

| Experiment | Runs | First run (UTC; cluster clock, consistent with commit times) | Commit | Plan at that commit contained |
|---|---|---|---|---|
| H-S6 (`12_bottleneck`) | 900 | 30 Aug 09:56 | 8f48df8 | Amendments 1–8 (H-S6 = Amendment 5) |
| H-S5 (`10_capacity`) | 1,000 | 31 Aug 18:28 | 9dcda12 | 1–8 (H-S5 = Amendment 4) |
| **Primary** (`01_frozen_tuned`) | 1,600 of 4,000 | **31 Aug 19:15** | 9dcda12 | 1–8; H-P1, H-P2 and the decision rule (§2) |
| H-S7 (`14_readout`) | 600 | 2 Sep 19:07 | 9e637b3 | 1–8 (H-S7 = Amendment 6) |
| d = 8, d = 16 | 1,000 | 6–7 Sep | 141a711 | 1–8 |
| full data | 80 | 11 Sep | e07459a | 1–8 |
| H-S1, H-S2, H-S3 (v2), H-S4, ansatz | 4,200 | 30 Sep – 1 Oct | bd833ba | 1–14 (Amendments 13, 14) |
| H-S6 follow-up, E7 | 4,000 | 3 Oct 11:18 | 30e57d9 | 1–16 (Amendment 16) |
| H-S5 follow-up | 1,600 | 6 Oct 14:42:28 | 17506e8 | 1–17 (Amendment 17, committed 14:42:01) |

Every experiment ran on a commit whose analysis plan already contained its
hypothesis and, for the follow-ups, its written prediction. Two qualifications
belong in the paper: the primary crossover was first noticed in diagnostic data
before the plan (§0 of the plan says so), and Amendments 3a and 9–11 were
written retrospectively (they say so).

### 6.7 The function-class figure is the reported model

`src/16_reproduce_for_figure.py` retrained the run quantum_vqc / BloodMNIST /
n = 5 / seed 42 on the cluster GPU: test AUC 0.809446 recorded, 0.809446
reproduced, maximum per-image probability difference 0.0. The trained head's
output (PennyLane) equals an 81-term classical trigonometric sum to 2×10⁻¹⁵.

**For the caption.** This run has 2 of 4 coordinates pinned (z2, z4). The
BloodMNIST n=5 VQC average is 1.2 (E8, §3), so the run is more saturated than
typical. The figure shows the plane of the two coordinates with the largest
interquartile range (z1–z3) with the pinned pair fixed at their medians. That
rule was chosen after a first version plotted a saturated plane. It is a
presentation choice, and the caption states it.

---

## 7. Disclosures

Validation uses 2n labels per class (training n) · BatchNorm in eval mode ·
`10_capacity`, `12_bottleneck`, `07` at the 1e-3 default (H-S5 and H-S6 rerun
at 1e-2 in the follow-ups) · `12_bottleneck` 5 seeds, `10_capacity` 10 seeds ·
datasets treated as fixed in the nested bootstrap · 28×28 images upsampled to
224 · one backbone (ResNet-18 to layer3) · simulation only · follow-up rate
tuned under a learned bottleneck · Pauli-X readout.
