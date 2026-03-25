# Datasets: MedMNIST v2

This directory serves as the local cache for the medical imaging datasets used in the study: *Expressivity and Robustness of Hybrid Quantum-Classical Models in Medical Image Classification under Severe Information Constraints*.

## Datasets Used

We benchmark our architectures on two distinct modalities from the MedMNIST v2 collection, standardized to 28x28 pixel resolutions:

1. **BreastMNIST:** 780 breast ultrasound images (Binary Classification: Benign vs. Malignant). Used to evaluate performance on complex, low-contrast textures.
2. **PneumoniaMNIST:** 5,856 pediatric chest X-ray images (Binary Classification: Normal vs. Pneumonia). Used to evaluate performance on distinct topological structures.

**Data Scarcity Regimes:** As detailed in the paper, experiments are conducted on both the full datasets (100%) and severely constrained subsets (10%) to simulate rare-disease clinical settings. The 10% sampling logic is handled dynamically within our data loader scripts to ensure reproducibility.

## Automatic Download (Recommended)

You **do not** need to manually download the datasets. The data loading script (`src/data/medmnist_loader.py`) utilizes the official `medmnist` Python package. 

When you run any of the experiment scripts (e.g., `01_frozen_backbone_ablation.py`), the script will automatically:
1. Detect if the `.npz` files are missing from this `data/` directory.
2. Download the official standardized splits directly from the MedMNIST servers.
3. Cache them here for all future runs.

## Manual Download

If you prefer to download the raw data manually, or if you are working in an offline environment, you can download the `breastmnist.npz` and `pneumoniamnist.npz` files directly from the [Official MedMNIST Zenodo Repository](https://zenodo.org/record/6496656) and place them in this folder.

## Citation

If you use this data in your extended research, please ensure you cite the original MedMNIST authors alongside our paper:

> Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.