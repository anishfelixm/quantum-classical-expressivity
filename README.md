# Expressivity and Robustness of Hybrid Quantum-Classical Models in Medical Image Classification under Severe Information Constraints

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-yellow)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Authors:** Anish Felix Mathias, Susham P S  
**Code Repository for the associated IEEE Conference submission.**

---

## 📝 Abstract

The deployment of deep learning models in edge computing environments—particularly in privacy-sensitive domains like medical imaging—often mandates severe dimensionality reduction. However, compressing high-dimensional spatial features into ultra-low-dimensional latent spaces causes classical linear layers to suffer from topological collapse, severely limiting classification performance. In this paper, we investigate whether Hybrid Quantum-Classical Neural Networks can circumvent this limitation by mapping highly compressed latent representations into an exponentially large quantum Hilbert space. We enforce a strict 4-dimensional information bottleneck on a fine-tuned ResNet-18 backbone, comparing the expressivity of a 4-Qubit Variational Quantum Circuit (VQC) against an equivalent classical linear layer. Evaluated on BreastMNIST and PneumoniaMNIST, the quantum hybrid demonstrates a decisive expressivity advantage, breaking the classical optimization plateau. On PneumoniaMNIST, the VQC achieved a near state-of-the-art AUC-ROC of 0.9697 with only 10% of training data, outperforming the equivalent classical baseline of 0.8641, though this extreme data efficiency proved highly sensitive to the gradient variance of ambiguous textures (Hybrid Instability). Furthermore, our robustness analysis reveals a fundamental "Precision Paradox": the high-frequency phase interference that grants VQCs superior expressivity renders them highly susceptible to phase decoherence under additive Gaussian noise ($\sigma > 0.03$), whereas the classical model exhibits stochastic resonance and graceful degradation. This study formally defines the expressivity-robustness trade-off governing quantum advantage in constrained neural architectures.

---

## 📊 Core Discoveries

### 1. The Bottleneck Gap (Expressivity)
We force both models to classify high-resolution medical images using only **4 latent dimensions** ($d=4$). 
* **Classical Collapse:** The linear bottleneck causes the classical model to suffer from topological collapse, hitting a hard optimization ceiling.
* **Quantum Advantage:** The VQC mathematically maps the 4 compressed features into a 16-dimensional complex Hilbert space, successfully resolving the overlapping hyperplanes and continuing to minimize the loss function.

![Fine-Tuning Dynamics](paper/figures/finetune_dynamics.png)
*(Figure 1: The Quantum model (Orange) breaks the optimization plateau that limits the classical model (Blue) under the strict $d=4$ bottleneck constraint.)*

### 2. The Precision Paradox (Robustness)
We stress-test the peak-expressivity models by injecting simulated analog sensor degradation (Additive Gaussian Noise).
* **Classical Ruggedness:** Degrades gracefully. Because it utilizes a ReLU activation (hard zero-cutoff), the classical model demonstrates **Stochastic Resonance**, actively improving its performance at minimal noise levels ($\sigma=0.01$).
* **Quantum Phase Decoherence:** The VQC acts as a "Glass Cannon." The precise high-frequency phase angles required for its superior accuracy are rapidly destroyed by spatial perturbations, causing catastrophic failure at $\sigma > 0.03$.

![Robustness Decay](paper/figures/robustness_glass_cannon.png)
*(Figure 2: The Expressivity-Robustness Trade-off. The quantum model achieves superior baseline accuracy but crashes sharply due to phase decoherence, falling below the classical model at the $\sigma \approx 0.02$ crossover point.)*

---

## 📂 Repository Architecture

This repository is designed for strict academic reproducibility. The pipeline is separated into sequential, decoupled execution scripts.

```text
quantum-classical-expressivity/
├── data/
│   └── README.md                  # Data loading instructions (MedMNIST v2)
├── paper/
│   ├── figures/                   # Auto-generated publication-ready plots
│   └── main.tex                   # LaTeX source code
├── results/                       # Auto-generated JSON logs and .pt weights
└── src/
    ├── models/
    │   ├── classical_resnet.py    # Classical Baseline (ResNet + Linear Head)
    │   └── quantum_vqc.py         # Quantum Hybrid (ResNet + 4-Qubit VQC)
    ├── eval/
    │   └── generate_paper_plots.py # Plotting script
    │
    ├── 01_frozen_backbone_ablation.py  # Establishes zero-variance baselines
    ├── 02_end_to_end_finetuning.py     # Generates optimal weights and tests expressivity
    └── 03_robustness_evaluation.py     # Injects noise into the optimal weights
```

---

## Citation

@inproceedings{mathias2026expressivity,
  title={Expressivity and Robustness of Hybrid Quantum-Classical Models in Medical Image Classification under Severe Information Constraints},
  author={Mathias, Anish Felix and P S, Susham},
  year={2026},
  note={Code available at: [https://github.com/anishfelixm/quantum-classical-expressivity](https://github.com/anishfelixm/quantum-classical-expressivity)}
}