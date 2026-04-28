# Expressivity and Robustness of Hybrid Quantum Neural Networks for Constrained Medical Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-yellow)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Authors:** Anish Felix Mathias, Susham P S  
**Code Repository for the associated peer-reviewed submission.**

---

## 📝 Abstract

Deploying deep learning models in edge environments frequently mandates severe dimensionality reduction. However, compressing spatial feature maps into ultra-low-dimensional latent vectors causes classical classifiers to suffer from topological collapse, severely degrading performance. In this study, we investigate whether Hybrid Quantum-Classical Neural Networks (HQCNNs) can overcome this limitation by mapping highly compressed vectors into a complex quantum Hilbert space. We enforce a strict 4-dimensional information bottleneck on a ResNet-18 backbone and evaluate a 4-qubit Variational Quantum Circuit (VQC) against equivalent classical linear and multi-layer perceptron baselines. Through end-to-end optimization on medical imaging benchmarks (BreastMNIST and PneumoniaMNIST), we identify a "Latent Reshaping" effect: backpropagated quantum gradients actively adapt classical convolutional filters to output complex, quantum-friendly geometries. The VQC consistently bypassed classical optimization plateaus, achieving superior AUC-ROC in severe data-scarcity regimes (1% and 10%) while maintaining statistical parity under data abundance. However, robustness evaluations under additive Gaussian noise reveal a fundamental "Precision Paradox." In data-scarce regimes, the high-frequency phase interference granting the VQC its expressivity renders it exceptionally fragile, inducing catastrophic phase misalignment (σ > 0.03) while classical models degrade gracefully. Crucially, we empirically demonstrate that data abundance acts as a topological regularizer; VQCs optimized on complete datasets neutralize this fragility, maintaining robust decision boundaries that outperform classical architectures under extreme sensor noise. These findings define the critical expressivity-robustness trade-offs governing empirical quantum advantage in constrained neural architectures.

---

## 📊 Core Discoveries

### 1. Latent Reshaping and the Bottleneck Gap
We forced both models to classify high-resolution medical images using only **4 latent dimensions** ($d=4$). 
* **Classical Collapse:** The ultra-low Euclidean projection causes the classical linear and MLP models to suffer from topological collapse, hitting a hard optimization ceiling.
* **Quantum Advantage:** By evaluating complex expectation values across entangled states in a 16-dimensional complex Hilbert space ($\mathbb{C}^{16}$), the VQC actively reshapes the classical convolutional filters into a quantum-friendly geometry, bypassing classical plateaus in severe data-scarcity regimes.

![Expressivity Dynamics](paper/figures/fig02_expressivity_dynamics.png)

### 2. The Precision Paradox & Topological Regularization
We stress-tested the fully optimized models by injecting simulated analog sensor degradation (Additive Gaussian Noise).
* **The Glass Cannon:** Under data scarcity, the precise high-frequency phase angles required for the VQC's superior accuracy render it highly fragile. It suffers catastrophic **Phase Misalignment** at $\sigma > 0.03$, while the classical linear model acts as a stabilizing low-pass filter.
* **Data Abundance as a Regularizer:** Optimizing the VQC on a complete data regime neutralizes this fragility. The abundance of data forces the quantum gradients into flatter minima, allowing the VQC to maintain robust decision boundaries that outlast classical architectures under extreme noise.

![Precision Paradox](paper/figures/fig03_precision_paradox.png)

---

## 📂 Repository Architecture

This repository is designed for strict academic reproducibility. All experiments use locked deterministic seeds to guarantee parity across classical and quantum evaluations.

```text
quantum-classical-expressivity/
├── data/
│   └── README.md                  # Data loading instructions (MedMNIST v2)
├── paper/
│   ├── figures/                   # Publication-ready plots
│   └── main.tex                   # Final LaTeX source code
├── results/                       # Verified multi-seed JSON logs and .pt weights
└── src/
    ├── models/
    │   ├── classical_resnet.py    # Classical Baselines (Linear & MLP Heads)
    │   └── quantum_vqc.py         # Quantum Hybrid (4-Qubit VQC via PennyLane)
    ├── eval/
    │   └── generate_paper_plots.py # Reproduces all paper figures from JSON logs
    │
    ├── 01_frozen_backbone_ablation.py  # Establishes static-latent expressivity limits
    ├── 02_end_to_end_finetuning.py     # Unfreezes Layer 3 for Latent Reshaping
    └── 03_robustness_evaluation.py     # Injects locked-seed sensor degradation
```

---

## Citation

If you find this code or research helpful in your own work, please cite our paper:

```bibtex
@article{mathias2026expressivity,
  title={Expressivity and Robustness of Hybrid Quantum Neural Networks for Constrained Medical Image Classification},
  author={Mathias, Anish Felix and P S, Susham},
  year={2026},
  journal={arXiv preprint},
  note={Code available at: \url{[https://github.com/anishfelixm/quantum-classical-expressivity](https://github.com/anishfelixm/quantum-classical-expressivity)}}
}
```