# Work Remaining — v3

**Updated:** 11 October 2026.

## 1. Experiments — done

**All of them.** The 23-test confirmatory family, both pre-specified follow-ups
(Amendments 16 and 17), and every exploratory and fairness analysis listed in
`RESULTS.md`. Results are backed up and checksummed. The cluster is returned.

| Block | Status |
|---|---|
| Validity gate (freezing, gradient flow, parity, dequantization) | ✅ |
| LR selection, per arm | ✅ |
| Confirmatory sweep, 40 seeds | ✅ H-P1, H-P2 supported |
| H-S1 … H-S7 | ✅ 14 / 1 / 1 / 7 (supported / opposite / difference / not) |
| H-S6 follow-up | ✅ prediction confirmed |
| H-S5 follow-up | ✅ no restriction effect by the rule; prediction not cleanly confirmed |
| E1 ansatz, E7 re-upload control | ✅ |
| d=8, d=16, full data, noise, hardware noise, Lipschitz, depth, angle, tanh | ✅ |
| Backup | ✅ local + Google Drive, SHA-256 OK |

## 2. Writing — remaining

| # | Task | Needs |
|---|---|---|
| 1 | Commit corrected docs (v43) | 10 min |
| 2 | Extract results archive locally | 15 min |
| 3 | `generate_paper_plots.py`, seven figures | code from Claude, run locally, no GPU |
| 4 | Manuscript from conference `main.tex` | the bulk of the remaining work |
| 5 | Citation verification | every entry in `AUDIT_REPORT.md` §5 |
| 6 | Cover letter | conference overlap, data and code availability |
| 7 | `LowRankHead` docstring (one sentence) + parity check | 5 min |

## 3. The conference paper — what must not carry over

Checked against the conference `main.tex` (title *Expressivity and Robustness of
Hybrid Quantum Neural Networks for Constrained Medical Image Classification*).
None of the following may appear as a finding in the journal version:

| Conference claim | What the journal's controls show |
|---|---|
| VQC maps z into ℂ¹⁶ and gains expressivity from the Hilbert space; "Bottleneck Gap" | encoded amplitudes are real; outputs lie in a 3^d classical trigonometric span (residual 1e-16) |
| Classical heads suffer "topological collapse" at d=4 | the conference compared against Linear and MLP only, with no parameter matching; against a matched head the VQC is ahead only at n=5, by +0.014 |
| "Latent Reshaping": quantum gradients reshape the backbone favourably | H-S4 (encoder absorbs head) not supported; no VQC-specific reshaping is shown |
| "Precision Paradox", "Glass Cannon", "phase misalignment" | F1 collapse under noise with AUC preserved happens to the classical control too (gap +0.30 to +0.34); the VQC's excess is +0.04 to +0.07 |
| "Data abundance as a topological regulariser"; VQC superior under noise at full data | on clean full-data test sets the VQC shows no advantage (−0.0068 [−0.0175, +0.0053]); the noise analyses find the F1 collapse in every small head, so no VQC-specific fragility for data to "regularise" away |
| "Zombie State", exclusive to the hybrid, caused by tanh saturation | tanh·π/2 is applied to every arm in the journal design; the effect is not exclusive |
| Parameter-shift gradients | training uses backprop through the simulator (adjoint agrees to 3e-7) |
| Abbas et al. effective dimension as superior capacity | not tested; do not cite as support |

Protocol changes to state once, in a "changes from the conference version"
paragraph (needed for the overlap disclosure): quantum rate 5e-3 vs classical
1e-3 → per-arm tuned rates (1e-2); 50 epochs without early stopping → AUC-based
checkpoint selection on validation; 3 seeds → 40; fractional scarcity (10 % /
1 %) → absolute n ∈ {5, 10, 20, 50, 100} per class; two datasets → four; Linear
and MLP baselines → parameter-matched and function-class controls; PR-curve threshold locking for F1 → state the journal's F1 threshold
rule explicitly in Methods.

## 4. What a reviewer will most likely attack

1. **Effect size** — +0.014 AUC, carried by BloodMNIST. Answer: stated up front;
   the contribution is attribution.
2. **"So what causes it?"** — not determined. Learned projection required;
   restriction not established and not excluded. Say so plainly.
3. **28×28 images, one backbone, simulation only** — limitations, stated.
4. **Mechanism tests initially underpowered** — rerun at 40 seeds, pre-specified.
5. **Novelty against Bowles et al. and the 2026 matched-baseline papers** —
   positioning in `PAPER_OUTLINE.md` §3.

Acceptance cannot be promised. The controls earn a serious hearing at a
soundness-focused venue; the outcome depends on the manuscript and the reviewers.

## 5. Genuine limitations

1. No real quantum hardware; shot and depolarising simulation only.
2. Small qubit counts (4, 8, 16); classically simulable by construction.
3. MedMNIST 28×28, upsampled.
4. One backbone; four datasets, treated as fixed in the bootstrap.
5. Mechanism of the H-P1 advantage not identified beyond "needs a learned
   projection".
