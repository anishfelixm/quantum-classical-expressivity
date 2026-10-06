# The Paper — Outline v3.0 (final)

**4 October 2026.** All experiments complete. Supersedes v2.0, whose lead result
(Lipschitz) did not survive the audit and whose "no robust advantage" contradicts
the confirmatory decision rule. Numbers: `RESULTS.md`. Corrections: `AUDIT_REPORT.md`.

---

## 1. Title and core message

> **Where the Advantage Lives: A Capacity-Controlled, Pre-Specified Evaluation
> of Variational Quantum Classification Heads for Few-Shot Medical Imaging**

**Core message, one sentence.** A VQC head shows a small, confirmatory,
scarcity-dependent advantage over a parameter-matched classical head — but the
advantage exists only when a learned projection adapts upstream of it; isolate
the head and it loses at every data level.

## 2. Abstract (draft)

Hybrid models that attach a variational quantum circuit (VQC) to a frozen
pretrained backbone are widely reported to be parameter-efficient, typically by
comparing a few-dozen-parameter circuit against a much larger classical head. We
evaluate this setting under controls designed to attribute any difference to the
quantum head itself: exact parameter parity at three bottleneck dimensions,
classical function-class controls for both single and re-uploading encodings,
learning rates tuned per arm, and a pre-specified family of 23 tests analysed by
a nested bootstrap over test images and seeds with one false-discovery
correction. On four MedMNIST datasets with ResNet-18 features, the VQC head
outperforms its parameter-matched classical counterpart at five labels per class
(ΔAUC = +0.014, 95% CI [+0.003, +0.026]), an advantage that shrinks as data grow.
The effect is carried by one dataset, reverses at a larger bottleneck, and is
not explained by capacity restriction. A pre-specified follow-up identifies where
it lives: the trainable projection between backbone and head holds 97% of the
trainable parameters, and when it is frozen the VQC head is worse than the
classical head at every data level (ΔAUC = −0.030 and −0.019 at five labels per
class). The VQC also falls short of a direct fit over its own classical function
class, and its bounded output confers no robustness to sensor noise once the
downstream classifier is accounted for. We provide the analysis plan, code and
per-sample predictions.

## 3. Positioning — the first paragraph of Related Work decides acceptance

**What is already known, and must be cited as such:**

- Classical baselines generally outperform VQC models, and hybrid quantum
  components act like the classical parts they replace — Bowles, Ahmed & Schuld
  (2024), 160 datasets. *Cite in the introduction, as the general finding this
  paper tests in the setting where positive claims are most common.*
- VQC outputs are classical Fourier series — Schuld, Sweke & Meyer (2021);
  RFF dequantization — Sweke et al. (2025); classical surrogates — Schreiber,
  Eisert & Meyer.
- Re-uploading beats single encoding at matched parameters — Cassé et al.
  (arXiv:2412.12397). *Our H-S1 agrees; our pre-registered prediction did not.*
- Parameter-matched classical heads — Martín-Pérez et al. (CMES 2026);
  *Mathematics* 14(16) 2865 (2026); arXiv:2605.23324 (blood cells); QASA
  (arXiv:2504.05336, approximate, synthetic regression).
- Learned encoder vs PCA before a quantum circuit — BVM 2025 (PneumoniaMNIST).
- Lipschitz bounds for quantum models — Berberich et al., PRR 6, 043326 (2024).
- Reporting practice our capacity result addresses — *Sci. Rep.*
  s41598-026-70175-4 (2026): a 24-parameter VQC compared with 1,026–1,539-
  parameter linear heads behind a trainable compression layer.
- The call for matched baselines and attribution — *Quantum Reports* 8(2) 54 (2026).

**What is distinct — the six contributions, in this order:**

1. **Where the advantage lives.** Capacity accounting (97% in the projection;
   ~0.2 AUC worth of learned projection) plus a powered, pre-specified test
   showing the VQC head loses everywhere once the projection is frozen.
2. **A confirmatory, family-corrected positive result with measured limits** —
   small, scarcity-dependent, carried by one dataset, reversing at d=8, with its
   pre-registered mechanism (restriction) refuted.
3. **Function-class controls for both encodings** inside a real medical
   transfer-learning pipeline.
4. **Bounded output ≠ robustness** — scale-normalised sensitivity and paired
   noise results.
5. **A measured catalogue of confounds** — shared learning rate, clipping that
   binds on one arm, F1-based selection, unaccounted projection capacity,
   seed-level statistics.
6. **A pre-specified, version-controlled analysis plan** with dated amendments,
   released with code and per-sample predictions.

## 4. Structure (IEEE Access, two-column)

1. **Introduction** — the parameter-efficiency claim in hybrid medical QML;
   Bowles et al. as the general finding; the attribution question; contributions.
2. **Related work** — §3 above.
3. **Methods** — architecture (dressed circuit, Mari et al.); arms and exact
   parity; dequantization and the Fourier controls; bottleneck policies;
   training protocol and per-arm learning rates; analysis plan, nested
   bootstrap, family correction; structural verification.
4. **Results**
   - 4.1 Confirmatory table (all 23, one figure + one table)
   - 4.2 Where the advantage lives — capacity accounting + follow-up (**lead figure**)
   - 4.3 Scope — leave-one-out, d=8/16, full data
   - 4.4 Function-class controls — H-S2, E7, re-uploading
   - 4.5 Readout richness (H-S7)
   - 4.6 Noise and robustness — H-S3, the paired companion, normalised sensitivity
5. **Discussion** — what would have to be true for a quantum-head advantage;
   implications for evaluating hybrid models; the checklist.
6. **Limitations** — the disclosures list in `RESULTS.md` §5.
7. **Conclusion.**

**Supplement:** ansatz, depth, angle scale, tanh, hardware noise, learning-rate
sweep, encoder adaptation (H-S4), every per-cell table, the amendment log.

## 5. Figures (main text)

1. Pipeline diagram with trainable-parameter counts per component.
2. The confirmatory forest plot — 23 estimates with CIs, coloured by verdict.
3. **Δ(n) under learned, frozen-PCA and frozen-random bottlenecks** (lead figure).
4. Leave-one-out and d = 4/8/16.
5. Function-class controls: VQC vs Fourier (single encoding) and re-upload vs
   Fourier {−2..2}^d.
6. Noise: F1-vs-AUC gap for VQC and control, with the paired difference.

## 6. Claims discipline

**Never write:** "quantum advantage" · "pre-registered" without qualification ·
"robust" or "graceful degradation" for the VQC · "the quantum head is more
parameter-efficient" · anything implying generalisation beyond these four
datasets.

**Always pair:** the H-P1 advantage with its leave-one-out and d=8 result · the
H-P attribution with H-S5's failure · exploratory results with the word
"exploratory" · the E7 result with "not parameter-matched".

## 7. Likely objections and answers

| Objection | Answer |
|---|---|
| "Already known (Bowles et al.)" | They benchmarked generic models; we isolate the head in the hybrid medical setting where positive claims concentrate, find a confirmatory advantage, and locate it in the projection |
| "Effect is tiny" | Agreed and stated; the contribution is attribution, not the effect |
| "LR not re-tuned for frozen bottlenecks" | Same sign at 1e-3 and 1e-2 |
| "You didn't tune the VQC / chose a weak circuit" | Six-point per-arm grid, interior optimum; the alternative ansatz is worse |
| "Only 4 qubits" | Parity at d = 4, 8, 16 |
| "Simulation only" | Shot and depolarising analysis; stated as feasibility |
| "28×28 images" | Limitation; native-resolution replication proposed as future work |
| "Conference overlap" | Disclosed; the journal version tests and refutes two conference claims |

## 8. Remaining work

1. Update `STATE.md`, `WORK_REMAINING.md`, `MASTER_RESEARCH_DOCUMENT.md`.
2. `generate_paper_plots.py` — the six figures above, from shards.
3. Manuscript, in the IEEE Access template.
4. Cover letter: IEEE field, conference disclosure, data and code availability.
