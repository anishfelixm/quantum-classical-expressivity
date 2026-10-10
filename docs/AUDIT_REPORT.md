# Audit Report

**First written 3 October 2026. Revised 11 October 2026** after both
pre-specified follow-ups reported. The revision closes §2.3, adds §2.6 (a
correction to this report's own wording), and updates §4 and §6.

A line-by-line review of the code, the mathematics, the statistics and the
claims, against all results and the current literature.

**Bottom line.** The code and the mathematics are correct. The confirmatory
results are trustworthy — reproduced independently to four decimals. Five
claims in the earlier documents were wrong or overstated; all five are
corrected below and in `RESULTS.md` / `PAPER_OUTLINE.md` v3.1. No experiment
remains to run.

---

## 1. What was verified, and how

| Component | Check | Result |
|---|---|---|
| Training loop (`loop.py`) | selection on val AUC with strict `>`; one optimiser path; identical weight decay, clipping (20.0, above every observed norm), scheduler and class weighting for every arm; gradient norms measured before clipping | correct |
| Parameter parity (`heads.py`) | `matched_param_fullrank` = 16 + 4 + 4 = **24**; `low_rank` = 2dr + 2d = **24** (r=2), **48** (r=5); VQC = 3Ld = **24** | exact |
| Dequantization | ρ(z) = ⊗½(I + sin z_j X + cos z_j Z) ⇒ outputs span products of {1, cos z_j, sin z_j} = 3^d functions; product-to-sum maps this onto {cos ω·z, sin ω·z : ω ∈ {−1,0,1}^d}, which `fourier_rff` implements as 40 canonical frequencies → 80 features + bias = 81 | correct |
| Re-uploading spectrum | output in the {−2..2}^d span (test suite < 1e-8; `verify_v40` 2.21e-7); truncated basis fails (0.469) | correct — re-uploading is also classically constructible |
| Data pipeline | stratified sampling; full official test split; loader refactor bit-identical (max diff 0.0); native-resolution noise | verified on the pod |
| Metrics | fast AUC / macro-F1 match scikit-learn to 1e-9, including ties and duplicated indices | verified on the pod |
| Stored predictions | recomputed vs recorded AUC | **0.00e+00** on every namespace |
| Gradient flow | frozen backbone: no gradient; adaptive: non-zero for both arms | all `OK` |
| Family table | independent reproduction of `04`'s H-P1 (+0.0142) and H-P2 interval [−0.00844, −0.00127] | **exact** |
| Exploratory engine | re-upload − control reproduced the hand-computed sum of two table rows | **exact** |
| Learning rates | all four arms hit the registered grid's upper boundary; extension changed only `linear` | primary pair invariant to Amendment 3a |

---

## 2. Corrections to the paper's claims

### 2.1 The planned lead result was a scale artifact — **reframed**

Earlier drafts led with *"unitarity bounds the head, dimension-independently"*,
presented as robust and mechanistic.

The VQC's raw Lipschitz constant is small because unitarity caps its output at
1 — but the shared classifier that follows simply rescales that. Normalised by
each head's own output range (from `logs_lipschitz.txt`):

| Head | L / \|out\|max, d=4 | d=8 | d=16 |
|---|---|---|---|
| quantum_vqc | 1.55 | 1.53 | 1.31 |
| quantum_reupload | 1.84 | 1.63 | 1.41 |
| **matched_param_fullrank** | **0.58** | **0.49** | **0.42** |
| low_rank | 0.76 | 0.75 | 0.73 |
| fourier_rff | 1.18 | 1.25 | 1.46 |

Relative to its own scale, **the VQC is among the most sensitive heads, about
three times the primary control.** This agrees with the noise experiments:
under sensor noise the VQC's macro-F1 fell slightly *more* than the control's
(paired difference +0.04 to +0.07, every σ).

**What remains true:** the VQC's output is bounded in [−1, 1] by construction,
and its raw constant does not grow over d ∈ {4, 8, 16} at initialisation. That
is a property of the function class, not evidence of robustness.

### 2.2 "No robust parameter-efficiency advantage" — **contradicted the decision rule**

H-P1 and H-P2 are both supported after correction across all 23 tests. The
pre-specified rule then requires the claim *"scarcity-dependent advantage"*.
The paper states it, with three qualifications from the same evidence:

1. **Scope:** the nested interval holds the four datasets fixed. Leave-one-out
   (exploratory) shows the n=5 effect is carried by BloodMNIST.
2. **Attribution:** the rule attributes the advantage to restriction acting as a
   regulariser. That attribution was tested twice and is **not established**
   (§2.6).
3. **Terminology:** "VQC-head advantage over a parameter-matched classical
   head", not "quantum advantage" — the head computes a classical function.

### 2.3 "The advantage requires a learned bottleneck" — **now established**

*3 Oct:* H-S6's own experiment could not detect the advantage even under its
learned bottleneck; the claim was undetermined.

*11 Oct:* the follow-up (Amendment 16a; 40 seeds, rate 1e-2) gives Δ(5) =
−0.0301 [−0.0421, −0.0179] under frozen PCA and −0.0190 [−0.0321, −0.0063]
under a frozen random projection, p_adj 0.002 and 0.005, negative at every n.
The claim is supported, as a pre-specified follow-up outside the confirmatory
family.

### 2.4 "Zombie state" as a VQC failure mode — **mostly generic**

The F1-versus-AUC gap is +0.30 to +0.34 for the classical control and +0.36 to
+0.40 for the VQC. The phenomenon belongs to small heads on noisy inputs; the
VQC adds a modest, consistent excess. Also: this is *decision-threshold*
degradation with ranking preserved, which is not strictly calibration — report
ECE separately and use the precise term.

### 2.5 Smaller corrections

| Earlier text | Correction |
|---|---|
| "Every proposed mechanism fails" | Superposition: excluded by dequantization. Restriction: not established, not excluded (§2.6). Learned projection: required (§2.3). Spectrum: re-uploading helps (H-S1) |
| H-S1 prediction "re-uploading hurts under scarcity" | Wrong: it helps at every n ≥ 10, opposite at n=10. Report as a failed prediction, consistent with Cassé et al. |
| `LowRankHead` docstring: "the paper's central claim is a regularization effect" | Outdated. Replace with: "the restriction account was tested (H-S5 and Amendment 17) and not established" |
| Ansatz "no difference" prediction | Wrong: StronglyEntangling is better. The primary used the better circuit |
| "Interior learning-rate optimum" | Wrong: every arm's optimum sat at the registered grid's edge; the extension changed only `linear` |

### 2.6 "Restriction refuted" — **this report's own overstatement, withdrawn**

The 3 October version of this report, and `RESULTS.md` / `PAPER_OUTLINE.md`
v3.0, said restriction was "refuted" and the advantage "not explained by
capacity restriction". The original H-S5 could not support that: Δ₀(5) =
−0.0124 with interval [−0.0380, +0.0114] excludes an effect larger than the
primary advantage but not a smaller one, and it ran at 1e-3 with 10 seeds.

The powered follow-up (Amendment 17; 40 seeds, 1e-2) gives Δ₀(5) = +0.0097
[+0.0007, +0.0187], p = 0.037, p_adj 0.074, and slope −0.0016 [−0.0037,
+0.0005], p_adj 0.143. The smaller head leads at every n. Correct wording:

- no **scarcity-dependent** restriction effect, in either run;
- a **scarcity-independent** smaller-head benefit of up to ~0.019 AUC is not
  established and not excluded;
- restriction is therefore **neither established nor excluded** as a
  contributor to H-P1's +0.0142.

The contrast is between two classical heads; it does not test the VQC directly.

---

## 3. Disclosures the Methods and Limitations must carry

None of these favours one arm over another. All of them are things a careful
reviewer would find.

| Item | Detail |
|---|---|
| Validation labels | `VAL_MULTIPLIER = 2`: model selection sees 2n labels per class, training sees n. Identical for every arm, but it qualifies the "few-shot" framing |
| BatchNorm | kept in eval mode, including in adaptive layers |
| Untuned experiments | `10_capacity`, `12_bottleneck` and `07_hardware_noise` ran at the 1e-3 default; `12_bottleneck` with 5 seeds, `10_capacity` with 10. The two mechanism tests were rerun at 1e-2 with 40 seeds (Amendments 16, 17) |
| Statistical scope | nested bootstrap treats seeds and test images as random, datasets as fixed |
| Image resolution | 28×28 upsampled to 224; MedMNIST+ now offers native 224×224 |
| Single backbone | ResNet-18 truncated at layer3 — inherited from the conference paper; state why |
| Simulation only | shot and depolarising noise as feasibility; depolarising AUC-invariance proven for binary tasks |
| Follow-up learning rate | tuned under a learned bottleneck; reused, not re-tuned, for frozen ones |
| Readout basis | Pauli-X rather than the common Pauli-Z; immaterial because the variational layer applies arbitrary rotations |

---

## 4. The results, as they now stand

### Confirmatory — 23 tests, BH-FDR m = 23

**14 supported · 1 opposite to prediction · 1 difference without prediction · 7 not supported**

| | Result |
|---|---|
| H-P1, H-P2 | **supported** — Δ(5) = +0.0142 (p_adj 0.026); slope −0.0046 (p_adj 0.002) |
| H-S1 (re-uploading) | n=5 not supported; **n=10 opposite**; n=20 difference; **n=50, 100 supported** |
| H-S2 (VQC vs own-span Fourier fit) | n=5 not supported (p_adj 0.084); **n=10–100 supported** |
| H-S3 (F1 falls more than AUC) | **supported at all four σ** |
| H-S4 (encoder absorbs head) | not supported |
| H-S5a/b (restriction) | not supported — underpowered; see follow-up |
| H-S6 (survives frozen bottleneck) | not supported — underpowered; see follow-up |
| H-S7a/b (richer readout) | **supported** |

### Pre-specified follow-ups — outside the family, BH m = 2 each

| | Result |
|---|---|
| H-S6 follow-up | PCA −0.0301 (p_adj 0.002), random −0.0190 (p_adj 0.005); negative at every n — **prediction confirmed** |
| H-S5 follow-up | Δ₀(5) +0.0097 (p_adj 0.074); slope −0.0016 (p_adj 0.143) — **no restriction effect by the rule; prediction not cleanly confirmed** |
| E7 re-upload − Fourier {−2..2}^d | +0.015, +0.015, +0.003, −0.000, −0.004 — **prediction wrong at n=10**; not parameter-matched |

### Exploratory

| Analysis | Result |
|---|---|
| Re-uploading VQC vs control (not pre-specified) | **+0.018, +0.009, +0.007, +0.002, +0.003** — CI excludes 0 at four of five n |
| Ansatz: Basic − Strong | negative throughout; significant at n=100 |
| Noise, control's gap / paired VQC − control | +0.30 to +0.34 / **+0.04 to +0.07** |
| Learned − frozen bottleneck (pooled) | ~**+0.2 AUC**; vs PCA shrinks with n, vs random does not |
| d=8 / d=16 / full data | reversal / null / plateau at ≈ −0.007 |
| Depth, angle, tanh | L=2 worst on BloodMNIST; π/2 not handicapping; tanh costs some classical heads |

---

## 5. The literature, tracked

| Work | What it did | Relation to this paper |
|---|---|---|
| Schuld, Sweke & Meyer (2021) | quantum models as partial Fourier series | foundation of the dequantization result |
| Pérez-Salinas et al., *Quantum* 4, 226 (2020) | data re-uploading | the H-S1 arm |
| Sweke et al., *Quantum* 9, 1640 (2025); Schreiber, Eisert & Meyer | RFF dequantization; classical surrogates | the `fourier_rff` arms instantiate this |
| Mari et al., *Quantum* 4, 340 (2020) | dressed quantum circuits | the architecture evaluated here is this template |
| Bowles, Ahmed & Schuld (2024) | 160-dataset benchmark; classical baselines generally win | the general finding this paper tests in the hybrid medical setting |
| Berberich et al., *PRR* 6, 043326 (2024) | Lipschitz bounds for quantum models | §2.1: bounded output ≠ classifier robustness |
| arXiv:2404.16154 | QML vs classical robustness, classical "Fourier network" analogue of a re-uploading model | precedent for E7's design |
| Cassé et al., arXiv:2412.12397 | re-uploading vs mono-encoded VQC at matched parameters; gains from wider harmonic support | **H-S1 agrees** |
| Chen & Kuo (QASA), arXiv:2504.05336, later revision | approximately matched classical bottleneck, synthetic regression | closest precedent for the capacity concern |
| BVM 2025 (end-to-end encoders for QCNNs) | learned encoder vs PCA before a QCNN, PneumoniaMNIST | precedent for learned-vs-PCA |
| Martín-Pérez et al., *CMES* 148 (Aug 2026) | peer-reviewed parameter-matched benchmark; same dressed template, tanh·π/2 | leaves asymmetric weight decay, SPSA-vs-backprop, shared LR, unaccounted projection |
| *Sci. Rep.* s41598-026-70175-4 (Sept 2026) | 24-parameter VQC vs 1,026–1,539-parameter linear heads, frozen ResNet-18 | the reporting practice the 97% capacity result addresses |
| arXiv:2605.23324 | blood-cell classification; 40-parameter VQC vs 110-parameter matched model | approximate matching; closest dataset |
| *Mathematics* 14(16) 2865 (2026) | parameter-matched frozen-DenseNet121 head ablation | convergent practice |
| *Quantum Reports* 8(2) 54 (2026) | survey calling for matched baselines and ablations | the gap this paper fills |
| *Sci. Rep.* 16, 9017 (2026) | MedMNIST on 127-qubit hardware | complementary; no classical network |

**Positioning.** Matched classical baselines are becoming standard in 2026, so
"we matched parameters" is not new. What remains distinctive: exact parity at
three dimensions; measured capacity distribution (97%) and the ~0.2 AUC worth of
the learned projection; powered frozen-projection controls; function-class
controls for both encodings; nested statistics with a 23-test pre-specified
family; and showing that a bounded-output robustness argument fails once scale
is accounted for.

**Before submission:** re-verify every entry above against the published record
(authors, volume, pages). Several were collected from search snippets.

---

## 6. Remaining steps

All experiments are done and backed up (results, feature cache, checkpoints,
repo snapshot, environment; SHA-256 verified; second copy on Google Drive).

1. Commit these document corrections.
2. `generate_paper_plots.py` — figures from the bootstrap caches, no GPU needed.
3. Manuscript from the conference `main.tex`, per `PAPER_OUTLINE.md` v3.1.
4. Citation check; cover letter.
5. Update `LowRankHead`'s docstring (§2.5) — a docstring-only change; rerun the
   24-parameter parity check afterwards anyway.
