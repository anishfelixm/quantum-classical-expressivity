# Claims ledger: every statement in the manuscript and where it is established

Written 15 October 2026 from the model manuscript (v3). Every factual or numeric
statement is listed, with its source and status. Nothing goes into the submitted
text unless its status here is VERIFIED or it is changed to the corrected value.

**Status labels**

- **VERIFIED** — checked against the code, a committed log or the run records;
  the source is named.
- **CORRECTED** — the draft's value is wrong; the right value and its source
  are given.
- **POD-C / POD-E / POD-F** — checked by `scripts/verify_claims.py` on the pod,
  in the named section, because it needs the data or the records.
- **AMEND n** — needs runs, specified in Amendment n of `analysis_plan.md`
  before they run.

Logs named here are in the repository root. "Records" means the shard records.
`check_paper_facts` refers to `logs_paper_facts.txt` (pod, 15 October).

## Abstract and introduction

| Claim | Source | Status |
|---|---|---|
| four-qubit head, frozen ResNet-18, four MedMNIST sets, 5–100 per class | `config.py`, `01` | VERIFIED |
| rates tuned per head on held-out seeds | `09_lr_selection.py` (seeds 90001–90005, asserted disjoint) | VERIFIED |
| main comparisons with 40 seeds; 23 tests, nested bootstrap, one correction | `config.CONFIRMATORY_SEEDS`; `13_family_table.py` | VERIFIED |
| +0.014 (0.003 to 0.026) at n = 5; shrank with data | `logs_family.txt`, `family_table.json` | VERIFIED; budget dependence: AMEND 20; clipping: AMEND 21 |
| rests on one dataset; reversed at a larger bottleneck; absent with full data | RESULTS §3 (nested leave-one-out, d = 8 table, full data) | VERIFIED. Full data: 71 of 80 runs hit the epoch cap, so AMEND 20 (full-data block) |
| worse at every n with the projection frozen | Amendment 16a record | VERIFIED |
| restriction neither established nor excluded | Amendment 17 record | VERIFIED |
| loses to its function class; noise-related loss shared by every small head | family table H-S2; §6.4 | VERIFIED; Fourier clipping: AMEND 21 |
| thirteen confounds (Table) | count of rows: 13 | VERIFIED. Add a 14th row: clipping at 20 still bound on Fourier heads (AMEND 21) |
| 97% of trainable parameters in the projection | 1,028 / 1,062 (`logs_e7.txt` capacity line, `registry.py`) | VERIFIED |

## Methods

| Claim | Source | Status |
|---|---|---|
| split sizes (Table I) | MedMNIST v2 official splits | POD-B (counts read from the data cache) |
| size ratio > 150 | 89,996 / 546 = 165 | VERIFIED |
| exactly n per class; validation min(2n, available) per class | `medmnist_loader._stratified`, `config.VAL_MULTIPLIER = 2` | VERIFIED (code); POD-B (recorded per-class counts of all 4,000 primary runs) |
| test sets 156 … 7,180, unchanged | loader uses the full test split | VERIFIED (code); POD-B (stored predictions) |
| full data: complete splits; PathMNIST subsampled to 10,000 | `_proportional`: per-class `round(count × 10,000 / 89,996)`; recorded n_train 10,002 | **CORRECTED: 10,002 (10,000 allocated by class share, rounded per class)**; POD-B prints the per-class counts |
| [0,1] scaling, bilinear upsampling to 224, 3-channel replication, ImageNet normalisation | `medmnist_loader._prepare/_to_input`; `config.NORM_MEAN/STD` | VERIFIED |
| no augmentation in any reported comparison | `config.AUGMENT_FROZEN = False`; keys `aug=0` in every reported namespace | VERIFIED |
| ResNet-18 to layer3, global pooling, h ∈ R^256 | `models/backbone.py`, `registry.FEATURE_DIM` | VERIFIED |
| frozen, BN in eval mode; bit-exact check; negative control changes 45 buffers | `logs_flow_verification.txt` | VERIFIED |
| projection Linear(256, d), 1,028 parameters at d = 4 | `registry.py` | VERIFIED |
| PCA: centred top-d components of the training-split features; PathMNIST via its 10,002 subsample; no labels | `registry.fit_bottleneck`; `01.get_pool_features` (regime "full", POOL_SEED) | VERIFIED; say 10,002 |
| random projection N(0, 1/256), seeded by the run seed | `registry._configure_bottleneck`; `build_arm` passes the run seed | VERIFIED |
| tanh·π/2 for every arm | `registry` | VERIFIED |
| R_Y encoding; StronglyEntanglingLayers ×2, ranges 1 and 2; ⟨X_i⟩ readout | `models/quantum_vqc.py` | VERIFIED |
| 24 angles, N(0, 0.1²) initialisation | `quantum_vqc.py` (`init_std`); parity test | VERIFIED |
| exact state-vector simulation, backpropagation | `default.qubit`, `diff_method="backprop"` | VERIFIED |
| adjoint agrees "to within 3×10⁻⁷" | `logs_verify_environment.txt`: relative differences 2.9×10⁻⁷ to 3.9×10⁻⁷ | **CORRECTED: "to a relative 4×10⁻⁷"** |
| output in the 3^d span (derivation) | `MATH_VERIFICATION.md`; tests | VERIFIED |
| PennyLane residuals "6×10⁻¹⁶ to 2×10⁻¹⁵ for six (d, L)" | `logs_pytest.txt`: 4.4×10⁻¹⁶ to 3.8×10⁻¹⁵ | **CORRECTED: 4×10⁻¹⁶ to 4×10⁻¹⁵** |
| wrong-frequency basis fails | `logs_pytest.txt`: 0.908 | VERIFIED |
| NumPy rebuild "2×10⁻¹⁵ single and two-qubit, 4×10⁻¹⁵ re-upload; 3^d fails for re-upload" | `logs_independent_theory.txt`: 3.4×10⁻¹⁵, 1.7×10⁻¹⁵, 3.4×10⁻¹⁵; 3^d span 0.274 | **CORRECTED to the logged values** |
| Fig. 2: run reproduces AUC 0.809446; fit residual ≤ 2×10⁻¹⁵ | RESULTS §6.7 (`16_reproduce_for_figure.py` output) | VERIFIED from RESULTS; the residual value comes from the plotting script, so print it from `figure_data.json` before submission |
| end-of-circuit depolarising: uniform contraction 1 − 4p/3, binary AUC exactly unchanged | derivation; `logs_independent_theory.txt` (3.3×10⁻¹⁶); `logs_hwnoise.txt` (binary AUC identical at every p) | VERIFIED |
| per-layer noise: contraction 0.81–0.87 at p = 0.05; "reorders 24 of 1,770 pairs" | `logs_independent_theory.txt`: 0.813–0.871; 7 of 1,770 for the test's readout, median 10 over 1,000 readouts | **CORRECTED: 7 (median 10)**; on trained models: AMEND 23(b) |
| arm definitions and parameter counts (Table) | `heads.py`, `classical_fourier.py`, `logs_e7.txt` (Fourier r2 2,500) | VERIFIED |
| parity at d = 4 asserted for all three | `tests/test_parity.py` (66 passed, `logs_pytest.txt`) | VERIFIED |
| low-rank k = 0 (8 parameters), k = 8 (72) | 2dk + 2d | VERIFIED |
| Adam; weight decay 1e-4 on all parameters; batch 32 | `train/loop.py`, `config.py` | VERIFIED |
| early stopping: first epoch with the highest validation AUC; patience 30; scheduler halves after 15 | `loop.py`; `check_paper_facts` matches every run | VERIFIED |
| "minimum is 50 at n ≤ 10 on every dataset and 20 on the larger multi-class sets" | floor = min(max(20, 200 // B), 50) gives 50, 50, 40, 20, 20 (BloodMNIST); 50, 50, 33, 20, 20 (PathMNIST); 50, 50, 50, 50, 28 (binary sets) | **CORRECTED: state the table**; POD-C prints it |
| class weights total / (C · count), clipped [0.1, 10]; no effect on balanced sets | `loop.class_weights`; balanced n-shot sets give weight 1 | VERIFIED |
| Kaiming-normal linear layers, zero biases; low-rank U, V ~ N(0, 0.01/d); scales one | `heads.init_weights`; `LowRankHead` | VERIFIED |
| clip at 20, "twice the largest p95 (9.62, deep encoder; max 15.87)" | the value was set before any run record existed (git `7be3dcf`, 10 Aug 19:30; shard system `3e67091`, 23:55) | **REMOVE the basis**; replace with the measured step norms of every reported run (AMEND 21); POD-D searches anyway |
| "mean pre-clip norms at 1e-3: 0.48–0.76 (VQC), 0.97–2.55 (identity and Fourier)" | Amendment 3 text; the diagnostic runs carry per-epoch norms | POD-C (recomputed per dataset and n); quote only what POD-C prints |
| clip 1.0 replaced on 10 August before any reported run | git `7be3dcf`; every reported run is later | VERIFIED |
| "clipping acts only as a guard" | records: epoch means above 20 in Fourier heads | **FALSE as written** → AMEND 21 |
| budget check: PneumoniaMNIST, 3 seeds, selected epochs 8–108 | `13_stepbudget` shards (no script or log) | **REPLACE** by Amendment 19 (done) and Amendment 20 |
| rate grid, tuning at n ∈ {5, 20, 100}, all four sets, five seeds | `logs_lr.txt`; `09` | VERIFIED |
| every arm's best rate at the grid's upper edge at every tuned n | `logs_lr.txt` per-regime breakdown | VERIFIED |
| extension to 3e-2 and 1e-1 changed only the identity head | extended-grid validation AUCs (`check_paper_facts`) | VERIFIED |
| tuning table values (0.8785, 0.8845, 0.8979, 0.9048) | `logs_lr.txt`; `check_paper_facts` | VERIFIED; clipping in tuning: AMEND 21 (P6) |
| inherited rates; table of rates per experiment | recorded `lr_head` / `lr_quantum` per namespace (`check_paper_facts`) | VERIFIED (the "[to confirm]" row is 1e-2) |
| plan written 14 Aug, committed 17 Aug; amendments committed 27 Aug | git log; Records of 13 Oct | VERIFIED |
| provenance table (runs, first run, plan contents) | `paper/tables/provenance.csv` (`15_reporting_tables.py`) | VERIFIED |
| A100 MIG 3g.20gb; full-data runs on CPU | `logs_full.txt` header `device=cpu`; GPU | POD-A prints the GPU and partition |
| Python 3.10, PyTorch 2.4.1, PennyLane 0.42.3, NumPy 1.26.4, medmnist 3.0.2 | `requirements.lock.txt` | POD-A prints the installed versions |
| 800 re-upload runs 9 h 23 min; 800 Fourier runs 34 min | `logs_gap_closure.txt` | VERIFIED |
| integrity 2.2×10⁻¹⁶ over 4,000 runs; 66 tests | `logs_reporting_tables.txt`; `logs_pytest.txt` | VERIFIED |

## Results

| Claim | Source | Status |
|---|---|---|
| metric definitions; ECE with 15 bins | `train/metrics.py` (`n_bins=15`) | VERIFIED |
| G(σ) as a difference, fixed by amendment before the data | Amendment 13 | VERIFIED |
| Lipschitz procedure (20 draws, 512 inputs, 32 directions, ε = 1e-3, divided by max output) | `08_lipschitz.py`; `logs_lipschitz.txt` | VERIFIED; trained heads: AMEND 22 |
| bootstrap: B = 2000, +1 correction, smallest p ≈ 0.001, 0.002 adjusted | `13_family_table.py` | VERIFIED |
| SE of one AUC on BreastMNIST about 0.03–0.04 | — | POD-E (bootstrap SE over the test images of every BreastMNIST VQC run) |
| negligibility threshold 0.01, fixed in advance | plan §5 | VERIFIED |
| family table, all 23 rows | `family_table.json`, `logs_family.txt` | VERIFIED; budget (AMEND 20) and clipping (AMEND 21) sensitivity pending |
| absolute AUC table; F1-selection table | `paper/tables/absolute_auc.csv`, RESULTS §6.2 | VERIFIED |
| per-dataset Δ(5), leave-one-out values | RESULTS §3 | VERIFIED |
| d = 8 / 16 table, full data −0.0068 [−0.0175, +0.0053] | RESULTS §3 | VERIFIED; full-data budget: AMEND 20 |
| frozen-projection follow-up table | Amendment 16a record | VERIFIED |
| learned − PCA 0.187 → 0.082; learned − random 0.180–0.229 | `logs_stats_12_bottleneck.txt` | VERIFIED; untuned rate (1e-3), stated |
| restriction follow-up table | Amendment 17 record | VERIFIED; low-rank k = 8 epoch-mean norm 19.9 → AMEND 21 |
| E7 table | Amendment 16b record | VERIFIED; Fourier r2 clipping → AMEND 21 |
| readout +0.0114 / +0.0111 | family table | VERIFIED |
| noise table, G of other heads 0.297–0.381, paired excess, ECE values | `paper/tables/noise_metrics.csv`; RESULTS §3 | VERIFIED |
| noise realisation per run seed and σ | `noise.seed_for_sigma(seed, σ)` | VERIFIED |
| sensitivity 1.55 vs 0.58 / 0.76; 1.53, 1.31 vs 0.75, 0.73 | `logs_lipschitz.txt` (L_max / \|out\|max) | VERIFIED; trained: AMEND 22 |
| hardware: rate 1e-3; ≤ 500 test images; 1,024 shots retained 0.989–1.005; BloodMNIST 0.8382 → 0.8385 | `logs_hwnoise.txt` | VERIFIED as ratios of cell means; single runs vary more (POD-E); tuned models: AMEND 23 |
| compression check (pre-plan) table | `logs_premise.txt` | VERIFIED as context; superseded by Amendment 18 |
| E8: 160 exact reproductions; active coordinates; rank correlations | `logs_e8_saturation.txt`, `logs_e8_summary.txt` | VERIFIED |
| re-upload vs matched; ansatz −0.0180 (p_adj 0.005); depth; angle 3/20, 5/20; tanh | RESULTS §3 | VERIFIED |
| encoder adaptation not supported | family table | VERIFIED; trainable layer3 matched-head epoch norm 18.3 → AMEND 21 |

## Discussion, limitations, availability

| Claim | Status |
|---|---|
| Every exploratory label, seed count and rate disclosure | VERIFIED against the plan and the records |
| "trained normalised constants do not exist" | **REPLACE** with AMEND 22's measurement |
| "every number can be recomputed"; predictions and records released | needs the Zenodo deposit, including the new weights and step norms |
| AI-assistance statement | as agreed with the authors |
