# Work Remaining

**Created:** 27 August 2026 · **Submission target:** ~26 September 2026
**Rule:** an experiment omitted to save time is not a limitation. Only things
that are genuinely impossible go in Limitations (§5).

Tick items as they complete. Every item lists the command, the cost, and the
acceptance criterion — "done" means the criterion is met, not that the command
exited.

---

## 0. State as of 27 Aug

| | |
|---|---|
| Code | complete except `generate_paper_plots.py` |
| Tests | 61 passing |
| Structural proofs | ✅ freezing bit-identical (with negative control), gradient flow from every arm |
| Diagnostic runs | 2,944 + 1,000 capacity |
| Confirmatory runs | **0** |

### Hypotheses settled so far

| | Claim | Verdict |
|---|---|---|
| Original | Superposition gives access to more states | **Refuted** — output lies in the 3^d classical trigonometric span, residual 1e-16 |
| Replacement | Restriction acts as a regulariser | **Refuted** — capacity sweep, 1,000 runs, all slopes **positive** (+0.001 to +0.003), predicted negative |
| **Current** | The advantage is an artifact of the **learned bottleneck** | pilot supports it; **full run is item 1** |

**Consequence.** Both explanatory hypotheses are dead. The bottleneck ablation is
no longer a control — it is the paper's central claim, and it is promoted to
first priority.

---

## 1. Experiments — in priority order

### ☐ 1. Bottleneck ablation, full — **THE CENTRAL RESULT** · 8 GPU-h

```bash
nohup python -u src/12_bottleneck_ablation.py > logs_bottleneck.txt 2>&1 &
python src/12_bottleneck_ablation.py --summary-only
```

Pilot (2 datasets × 3 seeds) showed Δ at n=5 going **+0.014 → −0.007 (PCA) →
−0.181 (random)**. Now uses pool-fitted PCA (Amendment 7b), so "frozen" is no
longer confounded with "estimated from 10 images".

**Accept when:** all four datasets × 5 regimes × 10 seeds × 3 policies complete,
and the sign of Δ(5) under each policy is stated with a bootstrap CI.

---

### ☐ 2. Confirmatory sweep, 40 seeds, tuned LRs · 22 GPU-h

```bash
nohup python -u src/01_frozen_backbone_ablation.py --confirmatory \
  --use-tuned-lr --experiment 01_frozen_tuned > logs_confirm.txt 2>&1 &
python src/04_statistical_analysis.py --experiment 01_frozen_tuned --family-size 23
```

**Accept when:** no `SEED-LEVEL FALLBACK IN USE` banner, and H-P1/H-P2 have
BH-adjusted p-values.

---

### ☐ 3. H-S7 readout richness, full · 6 GPU-h

```bash
nohup python -u src/01_frozen_backbone_ablation.py \
  --arms quantum_vqc quantum_rich quantum_rich_padded \
  --dims 4 --experiment 14_readout --use-tuned-lr > logs_readout.txt 2>&1 &
```

Directly answers *"was the state ever used?"* — 10 observables versus 4 from an
identical circuit. **rich − padded** is the informative contrast (matched
classifier, matched parameters); **rich − vqc** is the headline.

**Accept when:** both contrasts have CIs, and the classifier delta (10 → 22
parameters) is stated in the table.

---

### ☐ 4. d = 8 and d = 16 · 45 GPU-h

```bash
# d=8, full 10 seeds
nohup python -u src/01_frozen_backbone_ablation.py \
  --dims 8 --arms quantum_vqc low_rank fourier_rff linear \
  --experiment 15_dim8 --use-tuned-lr > logs_d8.txt 2>&1 &

# d=16, binary datasets only (backprop is 0.357 s/step at d=16 vs 0.026 at d=4)
nohup python -u src/01_frozen_backbone_ablation.py \
  --dims 16 --datasets breastmnist pneumoniamnist \
  --arms quantum_vqc low_rank fourier_rff linear \
  --experiment 16_dim16 --use-tuned-lr > logs_d16.txt 2>&1 &
```

Uses `low_rank(rank=2)`, which gives exactly 6d parameters at **any** d — 24/48/96.
`matched_param_fullrank` cannot do this above d=4.

**Accept when:** the primary contrast is reported at all three dimensions and the
d=16 reduced scope is stated.

---

### ☐ 5. Full-data reference row · 8 GPU-h

```bash
nohup python -u src/01_frozen_backbone_ablation.py \
  --regimes full --dims 4 --seeds 42 123 2026 777 888 \
  --experiment 17_fulldata --use-tuned-lr > logs_full.txt 2>&1 &
```

Your brief asked for *"very low till full data range"*. The scarcity grid stops
at 100/class; this is the missing upper end. PathMNIST capped at 10,000
(`config.FULL_DATA_CAP`).

**Accept when:** every arm has a full-data row and the scarcity curve extends to it.

---

### ☐ 6. Q5 software noise, all five regimes · 12 GPU-h

```bash
nohup python -u src/03_robustness_evaluation.py --use-tuned-lr \
  > logs_noise.txt 2>&1 &
python src/04_statistical_analysis.py --experiment 03_robustness \
  --condition 0.20 --family-size 23
```

Seven arms now, including the rich-readout pair. Tests H-S3 and the
noise × scarcity interaction — the second pillar of the original hypothesis, and
the one with real mathematics behind it (unitarity bounds the output in [−1,1]).

**Accept when:** retention-by-scarcity table is populated at all five regimes.

---

### ☐ 7. Q7 hardware noise · 8 GPU-h — **run `--quick` first, never executed**

```bash
python src/07_hardware_noise.py --quick        # MUST pass before the full run
nohup python -u src/07_hardware_noise.py > logs_hwnoise.txt 2>&1 &
```

Shot noise (1024, 256, 64) and depolarizing (0.001–0.05) on `default.mixed`.
Quantum-only, so it is a feasibility section and **not** a quantum-vs-classical
comparison.

**Accept when:** both curves exist and the section is labelled feasibility.

---

### ☐ 8. Lipschitz constants · minutes

```bash
python src/08_lipschitz.py --dims 4 8 16
```

Now includes `low_rank` and both rich arms, and reports **L/√(out_dim)** — heads
emit different numbers of observables, and an L2 norm over more components is
mechanically larger even at equal per-component sensitivity. Raw and normalised
are both printed.

**Accept when:** the normalised column is used for any cross-arm claim.

---

### ☐ 9. Depth sweep L ∈ {1, 2, 4} · 12 GPU-h

```bash
for L in 1 4; do
  nohup python -u src/01_frozen_backbone_ablation.py --n-layers $L \
    --arms quantum_vqc --dims 4 --experiment 18_depth$L >> logs_depth.txt 2>&1
done
```

Circuit depth sets the manifold dimension (3·L·d = 12/24/48) and has never been
tuned. An untuned free parameter that could favour one arm is a fairness
objection.

**Accept when:** depth-vs-AUC is reported and the L=2 choice is justified by data.

---

### ☐ 10. Angle-scale sweep {π/2, π} · 8 GPU-h

```bash
nohup python -u src/01_frozen_backbone_ablation.py --angle-scale 3.14159 \
  --arms quantum_vqc matched_param_fullrank --dims 4 \
  --experiment 19_anglepi > logs_angle.txt 2>&1 &
```

RY is injective on [−π, π], so π/2 uses **half** the available range and bounds
what the quantum arm can resolve. Leaving it untuned at a value that plausibly
handicaps the treatment arm is exactly what a reviewer calls unfair.

**Accept when:** both scales are reported and the default is justified.

---

### ☐ 11. tanh ablation, classical arms only · 4 GPU-h

```bash
nohup python -u src/01_frozen_backbone_ablation.py --no-tanh \
  --arms linear mlp matched_param_fullrank low_rank fourier_rff \
  --dims 4 --experiment 20_notanh > logs_notanh.txt 2>&1 &
```

Quantum arms are dropped automatically — RY is 2π-periodic, so unbounded z maps
distinct latents onto identical states. Tests whether classical arms were
handicapped by a squashing they do not need.

**Accept when:** stated whether tanh helped or hurt the classical arms.

---

### ☐ 12. Premise check without X-ray flip · 3 GPU-h

```bash
python src/06_premise_check.py --datasets pneumoniamnist --force
python src/06_premise_check.py --summary-only
```

Amendment 8: mirroring a chest radiograph produces situs inversus. Augmentation
was active only in the premise check, so only these cells are affected.

**Accept when:** the d=256 − d=4 gap is re-reported for PneumoniaMNIST.

---

## 2. Code

### ☐ 13. `src/eval/generate_paper_plots.py` — full rewrite
Conference-era, crashes on import. **Last file needing work.** Write after the
runs so the figures match real results.

### ☑ Arm lists extended in `03` and `08`
### ☑ Lipschitz normalised by output width
### ☑ LR summary reads columns from disk, warns on boundary optima

---

## 3. Documents

- ☐ `analysis_plan.md` — **Amendment 3a**: LR grid extended to {3e-4 … 1e-1}
  because the optimum sat on the boundary; now interior. Record selected values:
  linear 3e-2, matched_param_fullrank 1e-2, fourier_rff 1e-2, quantum_vqc 1e-2.
  Note that linear and quantum_vqc prefer 3e-2 at n=5 — the global choice is a
  stated compromise.
- ☐ `analysis_plan.md` — record **H-S5 refuted** with the slope table.
- ☐ `MASTER_RESEARCH_DOCUMENT.md` — v4.0: thesis is now the bottleneck artifact;
  both explanatory hypotheses refuted.
- ☐ `STATE.md` — refresh after each sweep.

---

## 4. Budget

| Item | GPU-h |
|---|---|
| 1 bottleneck · 2 confirmatory · 3 readout | 36 |
| 4 dimensions | 45 |
| 5 full-data · 6 noise · 7 hardware | 28 |
| 9 depth · 10 angle · 11 tanh · 12 premise | 27 |
| **Total** | **~136 ≈ 6 days** |

Roughly 30 days remain. Run items 1–3 first: they carry the paper.

---

## 5. Genuine limitations — infeasible, not skipped

1. **No access to real quantum hardware.** Everything is simulated. Item 7 is
   the closest available approximation.
2. **Qubit count.** State-vector simulation is exponential; beyond ~20 qubits is
   not reachable on one GPU, and the density-matrix simulation used for
   depolarizing noise is 4^d rather than 2^d.
3. **Native-resolution medical imaging.** MedMNIST is 28×28 by construction.
   A native-resolution dataset is a different study.
4. **No quantum advantage is provable here.** The model is classically simulable
   by construction — that is how it is run. The paper claims characterization,
   not advantage.

Everything else on this list is being done.
