# Work Remaining — v8

**Updated:** 11 November 2026. Replaces v7 (13 October).

## 1. Experiments — done and frozen

All of them. No run is planned. Any later run needs a new amendment, written
before it runs.

| Block | Status |
|---|---|
| Confirmatory family (23 tests), follow-ups (Amendments 16, 17) | ✅ |
| Budget, clipping, re-execution (Amendments 20, 21, 25) | ✅ 20,300 runs identical; clipping removed wherever it acted |
| Trained-head sensitivity, hardware feasibility, gradient variance (22–24) | ✅ |
| Circuit wiring (26) and its CPU re-execution (27) | ✅ |
| Squash on the matched head (28) | ✅ both predictions held |
| Inherited learning rates (29) | ✅ 3 of 11 configurations below their best rate |
| The three CPU checks without clipping (30) | ✅ every prediction held, after the AMD redo of block C |
| Each arm at its own rate; the circuit without the squash (31) | ✅ P1, P2, P3 failed; P3 holds without clipping |
| The sine–cosine model on raw z (32) | ✅ every prediction held |
| Re-execution | ✅ 35,740 runs identical (GPU 20,220, CPU 15,520); 2,480 unclipped re-runs step-identical |

## 2. Writing — remaining

| # | Task | Status |
|---|---|---|
| 1 | Manuscript edits in `PAPER_PLAN_2026-11-11.md` (kept with the manuscript), in the order of its Part 0 | ⬜ |
| 2 | Decide the AI-use statement (`PAPER_PLAN` Part 4.10) | ⬜ |
| 3 | Optional appendix table of all 32 amendments (date, one line each) | ⬜ |
| 4 | First-run dates for the five new provenance rows | ⬜ |
| 5 | New references: Duong et al. 2026, Cruz et al. 2026, Gillani et al. 2026 (check the Duong DOI) | ⬜ |
| 6 | Read arXiv:2611.10544 before submission | ⬜ |
| 7 | Zenodo deposit and DOI, including namespaces 31–39 and their `C_`/`CU_` folders | ⬜ |
| 8 | Biographies and photographs (Anish Felix Mathias, Susham P S); Shettar's department → Computer Science and Engineering | ⬜ |
| 9 | Literature re-check in the submission week | ⬜ |
| 10 | Cover letter and declarations | ⬜ |

## 3. The conference paper — what must not carry over

Checked against the conference `main.tex` (title *Expressivity and Robustness of
Hybrid Quantum Neural Networks for Constrained Medical Image Classification*).
None of the following may appear as a finding in the journal version:

| Conference claim | What the journal's controls show |
|---|---|
| VQC maps z into ℂ¹⁶ and gains expressivity from the Hilbert space; "Bottleneck Gap" | encoded amplitudes are real; outputs lie in a 3^d classical trigonometric span (residual 1e-16); so whatever the head adds is not Hilbert-space expressivity. Compression to d=4 does cost AUC with frozen features on BloodMNIST (every n) and PathMNIST (small n) (Amendment 18, `RESULTS.md` §3); the earlier end-to-end check's "+0.0018" does not transfer. No test measures whether a head "recovers" that loss |
| Classical heads suffer "topological collapse" at d=4 | the conference compared against Linear and MLP only, with no parameter matching; against a matched head the VQC is ahead only at n=5, by +0.014 |
| "Latent Reshaping": quantum gradients reshape the backbone favourably | H-S4 (encoder absorbs head) not supported; no VQC-specific reshaping is shown |
| "Precision Paradox", "Glass Cannon", "phase misalignment" | F1 collapse under noise with AUC preserved happens to the classical control too (gap +0.30 to +0.34); the VQC's excess is +0.04 to +0.07 |
| "Data abundance as a topological regulariser"; VQC superior under noise at full data | on clean full-data test sets the VQC shows no advantage (−0.0068 [−0.0175, +0.0053]); the noise analyses find the F1 collapse in every small head, so no VQC-specific fragility for data to "regularise" away |
| "Zombie State", exclusive to the hybrid, caused by tanh saturation | tanh·π/2 is applied to every arm in the journal design; the effect is not exclusive |
| Parameter-shift gradients | training uses backprop through the simulator (adjoint agrees to 3e-7) |
| Abbas et al. effective dimension as superior capacity | not tested; do not cite as support |

Protocol changes to state once, in a "changes from the conference version"
paragraph (the README only; the draft was never submitted, so the paper itself need not discuss it): quantum rate 5e-3 vs classical
1e-3 → per-arm tuned rates (1e-2); 50 epochs without early stopping → AUC-based
checkpoint selection on validation; 3 seeds → 40; fractional scarcity (10 % /
1 %) → absolute n ∈ {5, 10, 20, 50, 100} per class; two datasets → four; Linear
and MLP baselines → parameter-matched and function-class controls; PR-curve threshold locking for F1 → state the journal's F1 threshold
rule explicitly in Methods.

## 4. What a reviewer will most likely attack

1. **32 amendments.** Answer: every later check had its prediction written
   first; 8 of 19 failed and are reported (`tab:checks`); the confirmatory
   family never changed.
2. **Effect size.** +0.014 AUC, carried by BloodMNIST. Stated up front.
3. **"So what causes it?"** It needs the learned projection. The squash does not
   explain it: with both heads on raw z the lead is +0.0198. Restriction is
   consistent with the data, but only from exploratory checks, and the classical
   pattern depends on the k = 8 rate. Say "consistent with", never
   "established".
4. **Different devices.** Every comparison pairs arms trained on the same device
   type. On CPU the primary contrast (+0.0155) and the function-class pattern
   repeat.
5. **Duong et al. 2026 used a Fourier surrogate too.** Cite them and agree with
   them. The new part is a confirmed positive result traced to its source.
6. **One learning rate pooled over n.** A stated limitation.
7. **28×28 images, one backbone, simulation only.** Stated limitations.

Acceptance cannot be promised. The controls earn a serious hearing at a
soundness-focused venue; the outcome depends on the manuscript and the reviewers.

## 5. Genuine limitations

1. No real quantum hardware; shot and depolarising simulation only.
2. Small qubit counts (4, 8, 16); classically simulable by construction.
3. MedMNIST 28×28, upsampled.
4. One backbone; four datasets, treated as fixed in the bootstrap.
5. Why the VQC leads the matched head is not established. Regularisation by a
   small head trained with its projection is consistent with the exploratory
   checks, but it is not shown.
6. The squash and tuned-rate checks are at d = 4 only, on CPU; low-rank,
   identity and re-upload heads were not run on raw z.
7. One learning rate per arm, pooled over n ∈ {5, 20, 100}; rates at d = 16
   were not checked.
