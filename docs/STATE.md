# Project State

**Updated:** 26 August 2026. Keep this current — it is the handoff document.

Read `docs/MASTER_RESEARCH_DOCUMENT.md` first for the full picture, then
`docs/analysis_plan.md` (binding pre-registration) and
`docs/MATH_VERIFICATION.md` (derivations). This file is the volatile one: what is
running, what is blocked, what is next.

---

## The question

At a fixed small parameter budget and with very few labelled images, does a
variational quantum head extract more from a compressed feature vector than a
classical head — and if it does, is the advantage quantum, or merely
trigonometric?

## The thesis (as currently supported)

The quantum head's advantage, where it exists, is a **regularization effect from
its restricted function class** — a 24-dimensional manifold inside an
81-dimensional trigonometric span — not a quantum computational advantage. It is
confined to the extreme-scarcity regime and reverses once data is sufficient.

**This is a characterization paper, not a "quantum wins" paper.**

---

## Answered

| Q | Question | Answer | Runs |
|---|---|---|---|
| Q0 | Does compression to d=4 cost anything? | No — 0.002 AUC when the encoder adapts | 144 |
| Q1 | Efficiency at 24 vs 24 parameters? | Tie overall (31/40 cells) | 400 |
| Q2 | Can the VQC exploit its own function class? | No — a direct fit over the same basis wins on multi-class | 1,200 |
| Q3 | Does the encoder absorb the bottleneck? | Yes, strongly; shrinks head differences 3–5× | 1,200 |

All four predate Amendment 2 (AUC selection) and are **diagnostic only**.

## Open

| Q | Question | Status | Cost |
|---|---|---|---|
| Q4 | Is narrow frequency support the cause? | relaunched frozen-only 17 Aug; **read the summary and fill in §"The finding"** | done? |
| — | Validity gate (`11`, `test_parity`, `test_freezing`) | **code ready, not run — blocks everything** | ~20 min |
| Q6 | Capacity sweep — the mechanism test (H-S5) | code ready | ~3 h |
| Q7 | Bottleneck ablation (H-S6) | code ready | ~8 h |
| — | LR selection (Amendment 3) | code ready | ~7 h |
| — | Confirmatory sweep, 40 seeds | blocked on the gate + LR | ~22 h |
| Q5 | Input-noise robustness, all five regimes | code ready | ~12 h |
| Q8 | Hardware noise (shot, depolarizing) | **never executed — run `--quick` first** | ~8 h |
| — | Full-data reference row | not started | ~7 h |
| — | d=8 / d=16 via `low_rank` | unblocked, not started | ~20 h |
| — | `generate_paper_plots.py` rewrite | conference-era, crashes on import | — |

**Immediate blocker: the validity gate has never been run.** Until
`11_flow_verification.py` passes, the two claims the design brief demands be
proven — that the frozen backbone stayed frozen, and that gradients reach the
encoder from every head including the quantum one — are assumed rather than
evidenced.

## The finding to confirm

Frozen encoder, Δ = AUC(quantum) − AUC(matched classical), averaged over 4 datasets:

| shots/class | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Δ | +0.039 | +0.023 | −0.025 | −0.023 | −0.020 |
| datasets favouring quantum | 4/4 | 3/4 | 1/4 | 0/4 | 0/4 |

Monotone crossover between n=10 and n=20. **Post hoc**, measured against the
rank-limited control and under F1 selection. Pre-registered in
`docs/analysis_plan.md`; to be confirmed at 40 seeds under the current protocol.

**Q4 result: ______** (fill in from
`python src/01_frozen_backbone_ablation.py --summary-only`. Prediction recorded
12 Aug, before the data existed: a wider spectrum should *hurt* at n=5–10 and
*help* at n=50–100.)

---

## Verified facts (do not re-derive)

- VQC output lies exactly in the 3^d trigonometric span; residual 1e-16
  (`tests/test_fourier_equivalence.py`). Matches Schuld/Sweke/Meyer 2021.
- Re-uploading widens the spectrum to (2R+1)^d, verified numerically.
- `freeze_policy`: `"all"` = frozen, `"layer3_only"` = adaptive. There is no `"frozen"`.
- Gradients propagate with `diff_method="backprop"`; adjoint agrees to 3e-7, 260× slower.
- Head parameters at d=4: `quantum_vqc` 24, `quantum_reupload` 24,
  `matched_param` 24, `matched_param_fullrank` 24, `low_rank(rank=2)` 24,
  `fourier_rff` 324.
- **`low_rank` at rank 2 gives 6d parameters at ANY d** — 24 / 48 / 96 at
  d = 4 / 8 / 16. The dense full-rank head manages parity only at d=4.
- `fourier_rff` = 80 independent features after the canonical-frequency fix.
- Gradient clipping at 20.0 never binds (largest observed p95 = 9.62).
- Barren plateau: gradient variance falls ~62× from d=4 to d=16.
- **Capacity split at d=4, learned bottleneck:** bottleneck 1,028 (97%), head 24
  (2%), classifier 10 (1%). Under a frozen bottleneck the head holds ~70%.

### Runtime, measured

| Setting | Per run |
|---|---|
| Classical head, frozen, cached features | seconds |
| `quantum_vqc`, frozen | ~2.5 min |
| `quantum_reupload`, frozen | 50 s (BreastMNIST) → ~13 min (PathMNIST) |
| Any arm, adaptive encoder | up to 46 min |

**Cost scales strongly with dataset size.** BreastMNIST estimates have
under-predicted PathMNIST by 5–20× twice. Always project from the largest dataset
in the sweep.

## Known defects, unfixed

- `MatchedParamHead` is rank-limited (width 3 regardless of d). Diagnostic only;
  **invalid at d > 4**.
- `fourier_rff` results before the canonical-frequency fix used a 68-dimensional
  basis and must be regenerated.
- Everything before Amendment 2 used F1 selection and is not comparable with
  later runs. Superseded Q4 shards are in
  `artifacts/shards/_superseded_f1selection/`.
- `src/eval/generate_paper_plots.py` is conference-era and crashes on import.
  **The last file needing a rewrite.**
- `07_hardware_noise.py` has never executed. Run `--quick` before committing hours.
- ETA display underestimates badly — it averages in instantly-returning cached cells.

---

## Environment

```
python 3.10 · torch 2.4.1+cu118 · pennylane 0.42.3 · numpy 1.26.4 · medmnist 3.0.2
conda env: qml_v2       (fallback: qml_journal)
GPU: A100-SXM4-40GB MIG 3g.20gb, driver 470 → CUDA capped at 11.4
simulator: default.qubit + backprop, all dimensions
```

Edit and commit on Windows; pull and run on Linux.
Artifacts live outside the repo via the `artifacts/` symlink.

**Never `git pull` on Linux while a sweep is running** — the git SHA is read at
shard-write time, so pulling mid-run splits one experiment across two commits.

**Commit and push before launching anything long.** Four days of Q4 compute were
lost once to an uncommitted editor undo.

---

## Next commands

```bash
# 1. the gate — nothing downstream is valid until this passes
python -m pytest tests/ -q
python src/11_flow_verification.py

# 2. prediction round-trip. If "SEED-LEVEL FALLBACK IN USE" appears, STOP.
python src/01_frozen_backbone_ablation.py \
  --datasets breastmnist --regimes 20 --dims 4 \
  --arms quantum_vqc matched_param_fullrank --seeds 42 123 --force
python src/04_statistical_analysis.py --experiment 01_frozen

# 3. sanity checks on the never-run scripts
python src/10_capacity_sweep.py --quick
python src/12_bottleneck_ablation.py --quick
python src/07_hardware_noise.py --quick
```

## Timeline — tight, no slack

| | |
|---|---|
| Validity gate + sanity checks | 26 Aug |
| LR selection, capacity sweep | 27 Aug |
| Confirmatory sweep | 28–29 Aug |
| Q5 noise, bottleneck ablation | 30–31 Aug |
| Statistics, figures | 1–3 Sep |
| Draft | 4–8 Sep |
| Review, submit | ~10 Sep |

**If it slips, cut in this order:** depth sweep, tanh ablation, angle-scale
sweep, latent probe, hardware noise, d=8/16. The primary claim needs only the
confirmatory sweep, Q2, Q5, and the validity gate.

Target venue: IEEE Access. APC covered by the university.
