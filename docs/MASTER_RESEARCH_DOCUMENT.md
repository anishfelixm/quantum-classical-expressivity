# Master Research Document

**Project:** Quantum–Classical Expressivity Under Extreme Latent Compression
**Repository:** `github.com/anishfelixm/quantum-classical-expressivity`, branch `feature/journal-expansion`
**Version 6.0 — 11 November 2026** (v5.2 of 15 October plus Amendments 20–32,
all run and recorded). Supersedes v3.0 (26 August), whose thesis ("the
advantage is an artifact"; "no parameter-efficiency advantage") was written
before the confirmatory run and is contradicted by it.
**Target venue:** IEEE Access. **Status:** experiments finished and frozen;
manuscript being revised to `PAPER_PLAN_2026-11-11.md` (kept with the
manuscript, outside the repository).

See `docs/CLAIMS_LEDGER.md` for the status of every manuscript claim.

> Read this first. Then `docs/RESULTS.md` (every number, by evidential tier),
> `docs/analysis_plan.md` (binding; amendments; follow-up results at the end),
> `docs/PAPER_OUTLINE.md` (v3.1) and `docs/STATE.md`.

---

## 1. The finding

> **A VQC head shows a small, confirmatory, scarcity-dependent advantage over a
> parameter-matched classical head, but only when a learned projection adapts
> upstream of it. Isolate the head and it loses at every data level. The
> circuit never significantly beats a classical model of its own function
> class. The shared squash does not explain the advantage: with both heads on
> raw z the lead is larger. In exploratory checks, smaller learners beat larger
> ones over the same function class at small n, which is consistent with
> regularisation but does not establish it.**

| Claim | Evidence | Tier |
|---|---|---|
| Advantage at n=5: Δ(5) = +0.0142 [+0.0025, +0.0256], p_adj 0.026 | H-P1 | confirmatory |
| It shrinks with data: slope −0.0046 per doubling, p_adj 0.002 | H-P2 | confirmatory |
| Carried by BloodMNIST (without it +0.0041 [−0.0071, +0.0155]); reverses at d=8; null at d=16; not detected with full data | leave-one-out, d=8/16, full | exploratory |
| Needs the learned projection: frozen PCA −0.0301, frozen random −0.0190 at n=5, negative at every n | H-S6 follow-up (Am. 16) | pre-specified follow-up |
| Restriction: no scarcity dependence in either run; a constant smaller-head benefit up to ~0.019 not established, not excluded | H-S5, H-S5 follow-up (Am. 17) | confirmatory + follow-up |
| Restriction at selected rates (k = 8 at 3e-3): +0.0131 [+0.0039, +0.0217] at n=5, −0.0052 at n=50, slope −0.0039 [−0.0057, −0.0020]; the pattern depends on the k = 8 rate | Am. 31 | exploratory |
| Re-upload VQC (24) vs fit of its 5^d class (2,500): leads at n=5 and 10 in all six versions (GPU/CPU, clipped/unclipped, inherited/selected rate); negative at n=100 | Am. 16b, 21, 31 | exploratory |
| Squash cost: matched +0.0153 at n=5; VQC +0.0130 to +0.0197 at every n; `fourier_rff` none detected (+0.0013 to +0.0052) | Am. 28, 31, 32 | exploratory |
| Both heads on raw z, each at its tuned rate (VQC 3e-2, matched 1e-2): +0.0198 [+0.0104, +0.0307] at n=5; squash removed from the matched head only: +0.0002 [−0.0099, +0.0104] | Am. 31, 28 | exploratory |
| The VQC loses to a direct fit over its own function class at n ≥ 10 | H-S2 | confirmatory |
| Same on CPU (−0.0082 ns at n=5; −0.0167 to −0.0200 at n ≥ 10) and on raw z (+0.0063 [−0.0044, +0.0176] at n=5; negative from n=20) | Am. 32 | exploratory |
| Its output is classical: 3^d trigonometric span, residual 1e-16 | dequantization test | proof + numerics |
| Bounded output is not robustness; F1 collapse under noise is generic | H-S3, Lipschitz, paired gap | confirmatory + exploratory |
| With frozen features, compression to d=4 costs AUC on BloodMNIST at every n (+0.018 to +0.055) and on PathMNIST at small n (+0.023 at n=5); nothing detectable on the binary sets. Both predictions failed. (The pre-plan end-to-end check, layer3 trainable, found +0.0018 mean at n=100/full; it does not transfer.) | Amendment 18, 10 seeds | exploratory |
| Under the VQC head the projection leaves ~1 fewer of 4 latent coordinates unsaturated at n=5 (same LR, 0.01, both arms); does not track the advantage; both pre-written predictions wrong | E8, 160 exact reproductions | exploratory |

**Wording.** "An advantage of the VQC head over a parameter-matched classical
head." Never "quantum advantage". Never "restriction refuted" or "restriction
explains". Never "the squash explains the advantage". "Consistent with
regularisation", labelled exploratory.

**Title caution.** The manuscript title is about where the advantage comes
from, not about compression. Amendment 18 shows d=4 is a real constraint with
frozen features on BloodMNIST, which is also the dataset that carries the
advantage (exploratory observation, untested). Do not turn that coincidence
into a claim that the circuit "recovers" compression loss: no test measures
that, and the function the circuit computes is classical either way.

**What the paper is.** A characterization study with stronger controls than the
subfield's norm. Its contribution is attribution — where a reported advantage
lives and what does not explain it — not the size of the effect.

---

## 2. Architecture

```
image (28x28, upsampled 224x224)
  -> ResNet-18 truncated after layer3, ImageNet-pretrained
  -> global pool -> h (256-d)
  -> bottleneck Linear(256, d)   [learned | frozen PCA | frozen random]
  -> z_tilde = tanh(z) * pi/2    applied to EVERY arm (raw z in Am. 28, 31, 32)
  -> HEAD                        the only thing that varies
  -> Linear(out, C) classifier   shared by every arm
```

Capacity at d=4, binary, learned bottleneck: projection 1,028 (**97%**),
head 24, classifier 10. Under a frozen projection the head holds ~70%.

| `freeze_policy` | Meaning |
|---|---|
| `"all"` | backbone frozen (features cached) |
| `"layer3_only"` | layer3 adapts |

There is no `"frozen"` value; using it raises.

---

## 3. The arms

| Arm | Role | Head params (d=4) |
|---|---|---|
| `quantum_vqc` | treatment: RY AngleEmbedding + StronglyEntanglingLayers ×2, ⟨Xᵢ⟩ | 24 |
| `matched_param_fullrank` | primary capacity control (full rank at d=4) | 24 |
| `low_rank` | capacity control at any d: 2dr + 2d | 24 (r=2); 48 / 96 at d=8 / 16 |
| `fourier_rff` | function-class control: 40 canonical freqs of {−1,0,1}^4 → 80 features + bias = 81 = 3^4 | 324 |
| `quantum_reupload` | R=2, spectrum {−2..2}^d | 24 |
| `fourier_rff_r2` | function-class control for re-upload: 312 freqs, 624 features | 2,500 |
| `quantum_rich`, `quantum_rich_padded` | 2-local readout, 10 outputs (H-S7) | — |
| `quantum_basic` | BasicEntanglerLayers ×6 (ansatz check) | 24 |
| `linear`, `mlp` | floors / references | 0 |
| `matched_param` | rank-limited — **diagnostic only** | 24 |

Parity identities: `MatchedParamFullRank` d² + 2d = 6d ⟺ d = 4 only;
`LowRank` 2dr + 2d = 6d ⟺ r = 2 at any d.

---

## 4. Methodology — the non-negotiables

- **Scarcity is absolute:** n ∈ {5, 10, 20, 50, 100} labels per class;
  stratified sampling; validation `min(2n, available)` per class.
- **Datasets:** BreastMNIST, PneumoniaMNIST, BloodMNIST, PathMNIST.
- **Seeds:** 40 confirmatory seeds; LR selection on disjoint seeds.
- **Checkpoint selection on validation AUC** (Amendment 2).
- **Per-arm learning rates** (Amendment 3, 3a); untuned arms inherit by
  `config.LR_INHERITANCE`.
- **Regularisation parity:** identical weight decay, gradient clipping at
  norm 20, scheduler and class weighting. The clipping is identical in
  setting but **not in effect**. The records of 15 October show the
  epoch-mean norm above 20 in Fourier heads (`fourier_rff_r2` 1,053 of
  70,673 epochs; d=8 Fourier 2,466 of 17,200) and never in a circuit arm.
  Amendment 21 re-executes every reported run step by step, and repeats
  without clipping where it acted.
- **Training budget:** 100 epochs. 57–63% of primary runs reach the cap.
  The cap does not change n=5 results (Amendment 19); Amendment 20 checks
  every n and arm.
- **Every run saves per-sample predictions;** recomputed metrics match
  recorded ones to 0.00e+00.
- **Statistics:** nested paired bootstrap over test images and seeds, B = 2000,
  datasets weighted equally and treated as fixed; +1-corrected p; BH-FDR over the
  declared family of **23** (`config.DECLARED_FAMILY_SIZE`), m = max(declared,
  computed). Verdict = p_adj ≤ 0.05 *and* the predicted sign.
- **Three tiers:** confirmatory (23 tests) · pre-specified follow-up
  (Amendments 16, 17; own BH) · exploratory (BH within each analysis).

---

## 5. Verified facts — do not re-derive

- Single-encoding VQC output in the 3^d span, residual 1e-16; wrong-frequency
  control fails (0.908). Re-upload output in the {−2..2}^d span (< 1e-8 test
  suite; 2.21e-7 `verify_v40`); truncated basis fails (0.469).
- Encoded amplitudes are **real**. The earlier, unsubmitted draft's "ℂ¹⁶" claim must not reach the manuscript.
- `diff_method="backprop"`; adjoint agrees to 3e-7.
- Frozen backbone: 0 parameters, 0 buffers changed; negative control without
  `set_bn_eval()` drifts 45 buffers. Adaptive: backbone gradient non-zero for
  every arm.
- Depolarising noise **applied at the end of the circuit** (what `07` simulates)
  leaves binary AUC exactly invariant: ⟨Xᵢ⟩ → (1 − 4p/3)⟨Xᵢ⟩. Noise after every
  layer does not: contraction 0.81–0.87 at p = 0.05, 7 of 1,770 test pairs
  reorder for the test's readout vector (median 10 over 1,000 random readouts,
  5th–95th percentile 1–25; `scripts/print_independent_theory.py`). The figure
  "24" in earlier versions was wrong. Name the noise model in the paper.
- Head sensitivity normalised by output range, at initialisation (`08`, 20
  draws): VQC 1.55 / 1.53 / 1.31 (d = 4 / 8 / 16). `matched_param_fullrank`:
  0.58 / 0.49 / 0.42; it is parameter-matched only at d = 4. `low_rank`, the
  parameter-matched control at every d: 0.76 / 0.75 / 0.73. Source:
  `logs_lipschitz.txt`, L_max / |out|max.
- `canonical_frequencies(d, max_freq=1)` bit-identical to the pre-refactor basis.

### Runtime (for any future rerun)

| Setting | Per run |
|---|---|
| Classical head, frozen, cached features | seconds |
| `quantum_vqc`, frozen | ~2.5 min |
| `quantum_reupload`, frozen | 50 s (Breast) → ~13 min (Path) |
| Any arm, adaptive encoder | up to 46 min |

BreastMNIST estimates have under-predicted PathMNIST by 5–20×. **Always project
from the largest dataset.**

---

## 6. Corrections made along the way

Each of these was believed at some point and is now withdrawn:

| Withdrawn | Replaced by |
|---|---|
| Earlier, unsubmitted draft: Hilbert-space expressivity, Latent Reshaping, Precision Paradox, Glass Cannon, Zombie State, data abundance as regulariser | see `WORK_REMAINING.md` §3 |
| v3.0 (Aug): "the advantage is an artifact; no parameter-efficiency advantage" | H-P1, H-P2 supported |
| Outline v2.0: Lipschitz bound as lead result | scale artifact (§5) |
| 4 Oct: "restriction refuted" | not established, not excluded (Amendment 17) |
| "The advantage requires a learned bottleneck — undetermined" | established by the H-S6 follow-up |
| "Interior LR optimum" | optimum at the registered grid's edge; primary pair invariant |
| "Compression to d=4 is nearly free" (pre-plan end-to-end check) | false with frozen features on BloodMNIST and on PathMNIST at small n (Amendment 18) |
| "Per-layer noise reorders 24 of 1,770 pairs" | 7 of 1,770 for the test's readout; median 10 over 1,000 readouts (15 Oct) |
| "Clipping never binds" | binds on Fourier heads, never on circuits (15 Oct records); Amendment 21 |
| "Clipping never acts on circuits" | it acted in 59 circuit runs (Amendment 21 P2) |
| "Without the squash the VQC's lead is not detected" (Amendment 28) | holds only when the squash is removed from the matched head alone; with both on raw z the lead is +0.0198 (Amendment 31) |
| "Inherited rates are the ones tuning would pick" | false for `low_rank` k = 2, k = 8 and `fourier_rff_r2` (Amendment 29); affected comparisons re-run (Amendment 31) |
| "Full data: advantage absent" | "not detected" (Amendment 20 P4) |
| Clip threshold "2× the p95 of 9.62 (`deep_funnel`), max 15.87" | set before run records existed (git `7be3dcf`, 10 Aug 19:30); replaced by measured step norms (Amendment 21) |
| Pre-tuning gradient norms "0.48–0.76 / 0.97–2.55" | Amendment 3's figures; being recomputed from the diagnostic records (`verify_claims.py` C) before any use |

Confounds found and fixed: rank-limited "matched" control; clipping at 1.0
binding on classical arms only (threshold raised to 20 on 10 August); F1-based checkpoint selection; duplicated Fourier
frequencies; shared learning rate across arms with ~5× different gradient
scales; noise injected after upsampling (v1, disclosed); BH anti-conservative
when computed family > declared; unstratified pooling across angle scales;
mechanism tests run off the primary's protocol; clipping at 20 still binding on
Fourier heads (Amendment 21); the 100-epoch cap (Amendment 20); the CNOT wiring
of the re-upload circuit (Amendment 26); the squash chosen for the circuit but
applied to every head (Amendments 28, 31, 32); inherited learning rates
(Amendments 29, 31). Sixteen in the manuscript's confound table.

---

## 7. Environment and data custody

```
python 3.10 · torch 2.4.1+cu118 · pennylane 0.42.3 · numpy 1.26.4 · medmnist 3.0.2
conda env: qml_v2   (exported: conda_env_qml_v2.yml, pip_freeze.txt)
GPU: A100-SXM4-40GB MIG 3g.20gb (RVCE Kubeflow pod; available on
     15 October for Amendments 20–21)
simulator: default.qubit + backprop
CPU (Amendments 25–32): Google Colab Pro, High-RAM, AMD EPYC 7B12 only
     (an Intel Xeon session diverged at step 1; quarantined)
```

Runs since 21 October live on Google Drive, `QML-backup-run/cpu_reruns/`.

Backup dated 2026-10-10 at `C:\Users\Anish\quantum-classical-expressivity-remote-backup`
and on Google Drive: results (shards, predictions, bootstrap caches, tables,
logs), feature cache, checkpoints, data cache, latents, repo snapshot, git
state. SHA-256 verified. Added after that backup: `reproduced_2026-10-13b.tgz`
(function-class figure data) and `e8_results.tgz` (E8 shards, latents,
predictions, logs), both in the pod's `backup_2026-10-10/` and to be copied into
the local backup and Google Drive.

---

## 8. Publication assessment — honest

**In favour.** IEEE Access reviews for soundness and welcomes negative results.
Exact parity at three dimensions, function-class controls for both encodings,
capacity accounting, powered frozen-projection controls, a 23-test family under
one correction, the squash tested on every main head at tuned rates, 35,740
runs re-executed bit for bit, per-sample predictions released — together these are stronger
than the subfield's norm. The earlier draft was never submitted or
published, so nothing is retracted; the paper reports a study whose controls
were built to test claims of that kind.

**Against.** The effect is small and carried by one dataset. The cause is not
identified; the regularisation reading is exploratory. Simulation only, 28×28
images, one backbone. 32 amendments, which the scorecard of 19 pre-written
predictions (8 failed) must answer. The broad point that classical baselines
usually match VQCs is already published (Bowles et al. 2024), and Duong et al.
(Diagnostics, Oct 2026) already used a Fourier surrogate of a circuit head on
brain MRI and found the circuit worse. Novelty rests on tracing a confirmed
positive result to its source in the hybrid medical setting.

**Cannot be promised.** The controls earn a serious hearing; acceptance depends
on the manuscript and the reviewers.

---

## 9. Session conventions

- **Paste the file when in doubt.** The repo is the source of truth, not the chat.
- After touching `heads.py` or `registry.py`: `quantum_vqc`,
  `matched_param_fullrank`, `low_rank(2)` must all report 24 at d=4.
- After touching prediction I/O: two cells, then `04`. If
  `SEED-LEVEL FALLBACK IN USE` appears, stop.
- Commit and push before anything long. Never `git pull` on the run machine
  mid-sweep.
- No analysis changes after seeing results; amendments are dated with reasons.
