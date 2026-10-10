# RESULTS — Final

**11 October 2026. All experiments complete; cluster returned.** Supersedes the
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
modest, consistent excess. Bounded output did not buy robustness: the VQC's raw
Lipschitz constant is small because unitarity caps its output at 1, but the
classifier rescales it, and relative to its own range the VQC is about three
times as sensitive as the control.

### Hardware feasibility (`07`, default rate 1e-3, quantum only)

AUC retention at 1,024 shots: 0.989–1.005. Depolarising noise leaves binary AUC
**exactly** invariant — the readout contracts uniformly, ⟨Xᵢ⟩ → (1 − 4p/3)⟨Xᵢ⟩,
a monotone transform of the decision score — and moves multi-class AUC only
through the softmax (0.8382 → 0.8385 at p = 0.05).

### Fairness ablations (10 seeds)

| Axis | Result |
|---|---|
| Depth L ∈ {1,2,4} | on BloodMNIST both L=1 and L=4 beat L=2 at n ≥ 10; null elsewhere. L=2 is the only depth with exact parity (24 = 24) |
| Angle scale π vs π/2 | π helps the VQC in 3/20 cells (+0.012 to +0.019), hurts the control in 5/20 — π/2 did not handicap the VQC |
| tanh removed (classical arms) | helps `mlp` and `low_rank` on multi-class sets; the primary control mostly unaffected |

---

## 4. Established structural results

| Claim | Evidence |
|---|---|
| Single-encoding VQC output is classical | in the 3^d trigonometric span, residual 1e-16; wrong-frequency control fails (0.908) |
| Re-uploading output is classical | in the 5^d span, residual < 1e-8 in the test suite, 2.21e-7 in `verify_v40`; the truncated {−1,0,1}^d basis fails (0.469) |
| Parity | 24 = 24 = 24 at d=4; 48 and 96 at d=8, 16 via `low_rank` |
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
   VQC's excess is modest — **confirmatory + exploratory**. Bounded output is not
   robustness — **exploratory**.

---

## 6. Disclosures

Validation uses 2n labels per class (training n) · BatchNorm in eval mode ·
`10_capacity`, `12_bottleneck`, `07` at the 1e-3 default (H-S5 and H-S6 rerun
at 1e-2 in the follow-ups) · `12_bottleneck` 5 seeds, `10_capacity` 10 seeds ·
datasets treated as fixed in the nested bootstrap · 28×28 images upsampled to
224 · one backbone (ResNet-18 to layer3) · simulation only · follow-up rate
tuned under a learned bottleneck · Pauli-X readout.
