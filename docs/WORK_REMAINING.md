# Work Remaining — v7

**Updated:** 13 October 2026.

## 1. Experiments — done

**All of them.** The 23-test confirmatory family, both pre-specified follow-ups
(Amendments 16 and 17), every exploratory and fairness analysis listed in
`RESULTS.md`, and the E8 latent-saturation reproduction (13 October). The
10 October results are backed up and checksummed; the two archives added since
still need copying (`STATE.md`). The pod is still available; nothing is
planned on it.

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
| E8 latent saturation (exploratory, 160 exact reproductions) | ✅ both predictions wrong; ~1 fewer active coordinate under the VQC at n=5; does not track the advantage |
| Backup | ✅ 10 Oct local + Google Drive, SHA-256 OK; ⬜ add the 13 Oct archives |

## 2. Writing — remaining

| # | Task | Status |
|---|---|---|
| 1 | Corrected docs (v43), results extracted locally | ✅ |
| 2 | Reporting tables (`15`), figure reproduction (`16`), independent theory check | ✅ |
| 3 | Figures 1–7 and S1 from data | ✅ |
| 4 | `LowRankHead` docstring + parity check (24/24/24, 19 passed) | ✅ |
| 5 | E8 written up (`RESULTS.md` §3, finding 9) | ✅ |
| 5a | Pre-plan compression check into `RESULTS.md` (§3, finding 10); gradient-variance row with its correction (§4) | ✅ v50 |
| 5b | Docstring and doc corrections: `07` channel formula and scope, the `09` and `config.py` "calibration failure" wording, `MATH_VERIFICATION.md` addendum, `HYPOTHESES.md` marked historical | ✅ v50, via `scripts/v50_text_fixes.py` |
| 5c | Pod logs from the 10 Oct "FINAL" commit (5 logs + `logs_gap_status.txt`) saved and committed; pod branch bundled | ✅ v50 |
| 5d | Full test suite once in the `qml_v2` environment on the pod | ✅ 66 passed |
| 6 | Manuscript from conference `main.tex` on the IEEE Access template | ⬜ waits on template and author details; the bulk of the remaining work |
| 7 | Citation verification | ⬜ every entry in `AUDIT_REPORT.md` §5, plus the 2026 papers (2608.07822, 2605.09118, 2605.19417, 2609.15267) |
| 8 | Cover letter and declarations | ⬜ data and code availability, no conflicts, no funding (cluster access acknowledged), AI-use disclosure. No overlap disclosure: the conference draft was never submitted or published (confirmed 14 Oct) |
| 9 | Code release: README rewritten to the journal's findings; conference figures and logs moved to `legacy/conference/` with a note; README claims checked mechanically (`scripts/v51_release_tidy.py`). Left: repository visibility (public now, or at acceptance) and the paper/figures conference PDFs, removed with the manuscript rewrite | ✅ v51, except visibility |
| 10 | Data release: per-sample predictions, bootstrap caches, run records and E8 latents in a citable archive (e.g. Zenodo DOI), named in the data-availability statement | ⬜ at submission |
| 11 | Barren-plateau table (`MATH_VERIFICATION.md` §7): no script in the repo produced it. Either recompute it from the `quantum_grad_var` that `loop.py` logs in every quantum run's history (backup shards), or leave it out of the paper | ⬜ |
| 12 | Literature re-check in the week of submission | ⬜ |

Captions to write with the manuscript: figure 2 must say its run is more
saturated than typical and how its plane was chosen (`RESULTS.md` §6.7); every
caption states the evidential tier of what it shows.

## 3. The conference paper — what must not carry over

Checked against the conference `main.tex` (title *Expressivity and Robustness of
Hybrid Quantum Neural Networks for Constrained Medical Image Classification*).
None of the following may appear as a finding in the journal version:

| Conference claim | What the journal's controls show |
|---|---|
| VQC maps z into ℂ¹⁶ and gains expressivity from the Hilbert space; "Bottleneck Gap" | encoded amplitudes are real; outputs lie in a 3^d classical trigonometric span (residual 1e-16); compression to d=4 costs +0.0018 mean AUC against d=256 at n=100 and full data (premise check, `RESULTS.md` §3), so there is no gap for a head to close |
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

1. **Effect size** — +0.014 AUC, carried by BloodMNIST. Answer: stated up front;
   the contribution is attribution.
2. **"So what causes it?"** — not determined. Learned projection required;
   restriction not established and not excluded. Say so plainly. If asked
   whether the projection degenerates under the VQC: E8 shows it saturates about
   one more coordinate at n=5 at the same learning rate, and this does not track
   the advantage (exploratory).
3. **28×28 images, one backbone, simulation only** — limitations, stated.
4. **Mechanism tests initially underpowered** — rerun at 40 seeds, pre-specified.
5. **"Why compress to d=4 at all, if it costs nothing?"** — the conference
   design, kept so the journal tests the conference's claims on their own
   terms; d=8 and d=16 are reported too. Say so, and do not motivate d=4 as an
   edge-compression necessity.
6. **Novelty against Bowles et al. and the 2026 matched-baseline papers** —
   positioning in `PAPER_OUTLINE.md` §3.

Acceptance cannot be promised. The controls earn a serious hearing at a
soundness-focused venue; the outcome depends on the manuscript and the reviewers.

## 5. Genuine limitations

1. No real quantum hardware; shot and depolarising simulation only.
2. Small qubit counts (4, 8, 16); classically simulable by construction.
3. MedMNIST 28×28, upsampled.
4. One backbone; four datasets, treated as fixed in the bootstrap.
5. Mechanism of the H-P1 advantage not identified beyond "needs a learned
   projection". E8 describes one difference in how the projection trains, not a
   mechanism.
