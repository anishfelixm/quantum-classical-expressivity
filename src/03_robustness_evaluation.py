"""
Phase 3: Robustness Evaluation under Sensor Degradation (The Zombie State Test).
"""
import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet, ClassicalDeepBottleneckResNet
from models.quantum_vqc import QuantumHybridResNet

DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
FRACTIONS_TO_TEST = [0.01, 1.0] 
BOTTLENECK_DIM = 4  
BATCH_SIZE = 32
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]
SEEDS = [42, 123, 2026, 777, 888]
RESULTS_FILE_NAME = "03_robustness_e2e_logs.json"


def add_gaussian_noise(images: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Applies the mathematically rigorous 4-Step Sensor Degradation.
    Assumes baseline dataloader normalization of mean=[0.5], std=[0.5].
    """
    if sigma == 0.0:
        return images
        
    # Standard MedMNIST Transform Stats (Adjust if using ImageNet stats)
    mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
    std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
    
    # 1. Inverse-Normalize to physical pixel space [0.0, 1.0]
    images_real = (images * std) + mean
    
    # 2. Inject Gaussian Noise
    noise = torch.randn_like(images_real) * sigma
    
    # 3. Hardware Sensor Clamp (biological reality constraint)
    images_noisy = torch.clamp(images_real + noise, min=0.0, max=1.0)
    
    # 4. Re-Normalize for ResNet digestion
    images_final = (images_noisy - mean) / std
    
    return images_final


def evaluate_robustness_curve(model, test_loader, device, seed, num_classes):
    model.eval()
    robustness_curve = {}
    
    for sigma in NOISE_LEVELS:
        torch.manual_seed(seed + int(sigma * 1000)) # Strict RNG locking per sigma
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.squeeze().long().to(device)
                
                # Apply rigorously bounded noise
                x_noisy = add_gaussian_noise(x, sigma, device)
                logits = model(x_noisy)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        robustness_curve[f"{sigma:.2f}"] = {
            "acc": float(acc),
            "bal_acc": float(bal_acc),
            "f1": float(macro_f1)
        }
        
    return robustness_curve


def load_and_test_seed(dataset, fraction, seed, device):
    _, _, test_loader = get_medmnist_loaders(
        dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=fraction, seed=seed, data_root="/home/jovyan/qml_exp_2026/data_cache"
    )
    all_raw_labels = [int(y[0]) for _, y in test_loader.dataset]
    num_classes = len(np.unique(all_raw_labels))
    
    lin_model = ClassicalLinearResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    mlp_model = ClassicalMLPResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    ae_model = ClassicalDeepBottleneckResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    q_model = QuantumHybridResNet(num_classes=num_classes, n_qubits=BOTTLENECK_DIM, n_layers=2).to(device)
    
    paths = {
        "Classical Linear": f"results/best_e2e_Classical_Linear_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Classical MLP": f"results/best_e2e_Classical_MLP_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Classical Deep AE": f"results/best_e2e_Classical_Deep_Bottleneck_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Quantum VQC": f"results/best_e2e_Quantum_VQC_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt"
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weights for {name} on {dataset} frac {fraction} seed {seed}.")
            
    lin_model.load_state_dict(torch.load(paths["Classical Linear"], map_location=device, weights_only=True))
    mlp_model.load_state_dict(torch.load(paths["Classical MLP"], map_location=device, weights_only=True))
    ae_model.load_state_dict(torch.load(paths["Classical Deep AE"], map_location=device, weights_only=True))
    q_model.load_state_dict(torch.load(paths["Quantum VQC"], map_location=device, weights_only=True))
    
    print(f"\n   --- RUNNING SEED: {seed} ---")
    lin_curve = evaluate_robustness_curve(lin_model, test_loader, device, seed, num_classes)
    mlp_curve = evaluate_robustness_curve(mlp_model, test_loader, device, seed, num_classes)
    ae_curve = evaluate_robustness_curve(ae_model, test_loader, device, seed, num_classes)
    q_curve = evaluate_robustness_curve(q_model, test_loader, device, seed, num_classes)
    
    return lin_curve, mlp_curve, ae_curve, q_curve


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, RESULTS_FILE_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}\n")
    
    results = {"experiment": "Robustness Decay", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        for frac in FRACTIONS_TO_TEST:
            print(f"\n=== {dataset.upper()} | ROBUSTNESS: {frac*100}% DATA ===")
            metrics = ["acc", "bal_acc", "f1"]
            
            # Arrays to store RAW values across seeds for Welch's t-test
            agg_lin = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_mlp = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_ae  = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_q   = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            
            for seed in SEEDS:
                try:
                    lin_curve, mlp_curve, ae_curve, q_curve = load_and_test_seed(dataset, frac, seed, device)
                    for sigma in NOISE_LEVELS:
                        s_key = f"{sigma:.2f}"
                        for m in metrics:
                            agg_lin[s_key][m].append(lin_curve[s_key][m])
                            agg_mlp[s_key][m].append(mlp_curve[s_key][m])
                            agg_ae[s_key][m].append(ae_curve[s_key][m])
                            agg_q[s_key][m].append(q_curve[s_key][m])
                except FileNotFoundError as e:
                    print(f"   [WARNING] Skipping seed {seed}: {e}")
                    continue

            # Calculate means and stds, and securely attach raw arrays for all metrics
            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear": {}, "classical_mlp": {}, 
                "classical_deep_ae": {}, "quantum_vqc": {}
            }
            
            for sigma in NOISE_LEVELS:
                s_key = f"{sigma:.2f}"
                
                # Helper function to generate full metric dicts cleanly
                def get_stats(agg_dict, current_s_key):
                    return {
                        "mean_f1": float(np.mean(agg_dict[current_s_key]["f1"])),
                        "std_f1": float(np.std(agg_dict[current_s_key]["f1"])),
                        "raw_f1": agg_dict[current_s_key]["f1"],
                        "mean_acc": float(np.mean(agg_dict[current_s_key]["acc"])),
                        "mean_bal_acc": float(np.mean(agg_dict[current_s_key]["bal_acc"]))
                    }

                results["datasets"][dataset]["fractions"][str(frac)]["classical_linear"][s_key] = get_stats(agg_lin, s_key)
                results["datasets"][dataset]["fractions"][str(frac)]["classical_mlp"][s_key] = get_stats(agg_mlp, s_key)
                results["datasets"][dataset]["fractions"][str(frac)]["classical_deep_ae"][s_key] = get_stats(agg_ae, s_key)
                results["datasets"][dataset]["fractions"][str(frac)]["quantum_vqc"][s_key] = get_stats(agg_q, s_key)
            
            print(f"\n   [AVERAGED MACRO-F1 DECAY AT SIGMA=0.10]")
            print(f"   Linear F1:      {np.mean(agg_lin['0.10']['f1']):.4f}")
            print(f"   Deep AE F1:     {np.mean(agg_ae['0.10']['f1']):.4f}")
            print(f"   Quantum F1:     {np.mean(agg_q['0.10']['f1']):.4f}")
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()