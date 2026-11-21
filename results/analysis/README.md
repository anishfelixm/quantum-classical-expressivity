# Analysis outputs of Amendments 26–32

Small JSON outputs of the exploratory checks run on Google Colab CPU
(31 October – 10 November 2026). Until 21 November 2026 they were on Google
Drive only. They are copied here unchanged.

| File | Written by | Amendment |
|---|---|---|
| `entangling_range.json`, `entangling_range_unclipped.json` | `scripts/analyse_entangling_range.py` | 26, 30 |
| `entangling_device_check.json` | `scripts/analyse_entangling_range.py` (device check) | 26 |
| `device_lock.json` | `scripts/entangling_colab.ipynb` (device check) | 26 |
| `squash_check.json`, `squash_check_unclipped.json` | `scripts/analyse_squash.py` | 28, 30 |
| `lr_inherited.json`, `lr_inherited_unclipped.json` | `scripts/analyse_lr_inherited.py` | 29, 30 |
| `nosquash_rates.json`, `nosquash_rates_unclipped.json` | `scripts/analyse_amendment31.py` (rate selection) | 31 |
| `amendment31.json`, `amendment31_unclipped.json` | `scripts/analyse_amendment31.py` | 31 |
| `fourier_nosquash_rate.json`, `fourier_nosquash_rate_unclipped.json` | `scripts/analyse_amendment32.py` (rate selection) | 32 |
| `amendment32.json`, `amendment32_unclipped.json` | `scripts/analyse_amendment32.py` | 32 |

`*_unclipped.json` holds the same analysis computed with clipping removed.
The printed console output of each analysis is in `logs/colab_cpu_reruns/stats_*.txt`
and `logs/colab_exp31/stats_entangling.txt`.

Figures 4, 6 and 7 of the paper are drawn from four of these files:

```bash
python scripts/fig_amendments.py --art results/analysis --out paper/figures
```

Before drawing, the script checks ten values against the recorded outputs. It
stops if any file is missing or any value differs.
