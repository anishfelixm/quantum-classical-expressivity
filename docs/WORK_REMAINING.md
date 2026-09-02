# Work Remaining — v2

**Updated:** 29 August 2026 · **Submission target:** ~26 September

---

## 1. Have all experiments run?

**No — 6 of 14 are done.** But the ones that carry the paper are.

| # | Experiment | Runs | Status |
|---|---|---|---|
| 1 | Dequantization proof | — | ✅ residual 1e-16 |
| 2 | Flow verification (freezing + gradients) | 13 | ✅ both proofs pass |
| 3 | Premise check (Q0) | 144 | ✅ |
| 4 | Diagnostic sweep | 1,803 | ✅ |
| 5 | LR selection | 1,440 | ✅ optimum interior |
| 6 | Capacity sweep, H-S5 | 1,000 | ✅ float32, refuted |
| 7 | Bottleneck ablation, H-S6 | 900 | ✅ float32 |
| 8 | **Confirmatory sweep** | 1,600 | ✅ **H-P2 supported** |
| 9 | H-S7 readout richness | 0 | ⬜ 6 h |
| 10 | Q5 software noise | 0 | ⬜ 12 h |
| 11 | Q7 hardware noise | 0 | ⬜ 8 h — never executed, `--quick` first |
| 12 | d=8 and d=16 | 0 | ⬜ 45 h |
| 13 | Full-data reference row | 0 | ⬜ 8 h |
| 14 | Depth / angle-scale / tanh ablations | 0 | ⬜ 24 h |
| 15 | Lipschitz | 0 | ⬜ minutes |

**≈103 GPU-hours remain. 28 days available.**

---

## 2. Has everything been tested?

**Code paths that have executed on real data: yes.** 61 unit tests pass, and
every script above ran end to end.

**Code paths that have never executed:**

- `07_hardware_noise.py` — **never run once.** Highest remaining risk.
  `torch.as_tensor(circuit(...))` on `default.mixed` is untested.
- `src/eval/generate_paper_plots.py` — conference-era, crashes on import.
- `03_robustness_evaluation.py` — the arm list was extended to 7 arms
  (including the rich-readout pair) and has not run since.
- `08_lipschitz.py` — same, plus the new per-dimension normalisation.

Every bug found in the last three days was in a path running for the first time
on a new shape of data. Expect the same from items 9–15.

---

## 3. The result, as it now stands

### Primary hypothesis

**H-P2 SUPPORTED.** Slope −0.00462 per doubling, CI [−0.00844, −0.00127],
40 seeds, per-arm tuned learning rates, nested bootstrap, BH-FDR m=23.

| shots/class | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| mean Δ | **+0.0142** | +0.0002 | −0.0050 | −0.0080 | −0.0074 |

**H-P1 — not yet computed.** The plan specifies Δ(5) *pooled across datasets*;
only per-dataset rows exist. One BH-surviving positive cell: BloodMNIST n=5,
Δ = +0.0446, p_adj = 0.026.

**This reverses the earlier reading.** The 10-seed untuned diagnostic showed no
crossover; 40 seeds with tuned LRs resolves one. That is what a confirmatory
sweep is for, and both readings must appear in the manuscript — the diagnostic
as diagnostic, the confirmatory as confirmatory.

### Secondary

| | Result |
|---|---|
| H-S5 restriction is the mechanism | **refuted** — 19/20 null, survivor dies under correction |
| H-S6 head vs bottleneck | crossover present under learned projection, absent under PCA and random |
| Bottleneck dominance | **d up to +6.73** — the largest effect in the project |
| Q2 dequantization | classical direct fit beats the VQC on multi-class, to −0.17 |

### The story the data tells

1. A small scarcity-dependent advantage exists and is statistically real.
2. It is **not** superposition — the output lies in a classical trigonometric span.
3. It is **not** capacity restriction — a classical capacity sweep shows nothing.
4. It **requires a learned bottleneck** — it vanishes under frozen projections.
5. The projection itself dominates everything, for every head.

An advantage that exists, is real, and is explained by none of the mechanisms
usually invoked for it — that is a stronger paper than either a clean win or a
clean null.

---

## 4. Order of work

**Now — the primary hypothesis is incomplete without these:**

1. Pooled H-P1 in `04` (code, not compute)
2. H-S7 readout, 6 h — answers "was the state used?"
3. Q5 noise, 12 h — the second pillar of the original hypothesis

**Then — robustness:**

4. d=8/16, 45 h — kills "is this a d=4 artifact?"
5. Full-data row, 8 h — completes the scarcity axis
6. Depth, angle-scale, tanh, 24 h — closes fairness objections
7. Hardware noise, 8 h — feasibility section
8. Lipschitz, minutes

**Last:** rewrite `generate_paper_plots.py`, then draft.

---

## 5. Documents needing updates

- `analysis_plan.md` — Amendment 3a (LR grid extension), Amendment 9 (float16 →
  float32 storage, with measured gaps 4.7e-03 → 1.04e-04 and which experiments
  were re-run), Amendment 10 (renormalisation made conditional).
- `MASTER_RESEARCH_DOCUMENT.md` — v4.0. The thesis changed again: H-P2 is
  supported, so the paper is no longer a pure null.
- `PAPER_OUTLINE.md` — §2 findings need the confirmatory numbers.
- `.gitignore` — add `logs_*.txt`; they are currently tracked.

---

## 6. Genuine limitations

1. No real quantum hardware.
2. Qubit count — state-vector simulation is exponential; depolarizing noise is
   4^d, worse.
3. MedMNIST is 28×28 by construction; native-resolution imaging is a different
   study.
4. No quantum advantage is *provable* here — the model is classically simulable
   by construction. The paper claims characterization.

Everything else on this list is being done.
