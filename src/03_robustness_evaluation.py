"""
Phase 3: Robustness Evaluation under Sensor Degradation (The Zombie State Test).

Evaluates the decision boundary stability of End-to-End Hybrid Quantum-Classical 
models against injected Gaussian noise. This script strictly enforces multi-class 
Argmax metrics to measure true topological degradation without threshold shifting.
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

# --- CONFIGURATION ---
# Target the datasets and fractions we want to stress-test.
# Typically, testing extreme scarcity (0.01) vs full data (1.0) highlights the contrast perfectly.
DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
FRACTIONS_TO_TEST = [0.01, 1.0] 
BOTTLENECK_DIM = 4  # Standardized evaluation on the 4-qubit/4-dim boundary
BATCH_SIZE = 32
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]
SEEDS = [42, 123, 2026, 777, 888]
RESULTS_FILE_NAME = "03_robustness_e2e_logs.json"


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Applies Sensor Degradation noise and strictly clamps to physical bounds."""
    if sigma == 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    # Ensure pixels do not exceed biological/sensor validity [0, 1]
    return torch.clamp(images + noise, min=0.0, max=1.0)


def evaluate_robustness_curve(model, test_loader, device, seed, num_classes):
    """
    Evaluates model decay across noise levels using strictly Argmax classification.
    """
    model.eval()
    robustness_curve = {}
    
    for sigma in NOISE_LEVELS:
        # STRICT METHODOLOGICAL PARITY: 
        # Reset RNG uniquely for this sigma, but identically across architectures.
        # This guarantees all models face the exact same noise tensors per test sample.
        torch.manual_seed(seed + int(sigma * 1000))
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.squeeze().long().to(device)
                
                x_noisy = add_gaussian_noise(x, sigma)
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
    
    # Extract num_classes dynamically
    all_raw_labels = [int(y[0]) for _, y in test_loader.dataset]
    num_classes = len(np.unique(all_raw_labels))
    
    # Initialize models
    lin_model = ClassicalLinearResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    mlp_model = ClassicalMLPResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    ae_model = ClassicalDeepBottleneckResNet(num_classes=num_classes, bottleneck_dim=BOTTLENECK_DIM).to(device)
    q_model = QuantumHybridResNet(num_classes=num_classes, n_qubits=BOTTLENECK_DIM, n_layers=2).to(device)
    
    # Construct strictly matching file paths based on Experiment 2 output
    paths = {
        "Classical Linear": f"results/best_e2e_Classical_Linear_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Classical MLP": f"results/best_e2e_Classical_MLP_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Classical Deep AE": f"results/best_e2e_Classical_Deep_Bottleneck_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt",
        "Quantum VQC": f"results/best_e2e_Quantum_VQC_{dataset}_frac{fraction}_b{BOTTLENECK_DIM}_seed{seed}.pt"
    }
    
    # Verify weights exist
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Phase 2 weights for {name} on {dataset} frac {fraction} seed {seed}. Path checked: {path}")
            
    # Load state dicts
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
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "Robustness Decay (End-to-End Models)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        for frac in FRACTIONS_TO_TEST:
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | ROBUSTNESS: {frac*100}% DATA REGIME")
            print(f"=====================================================")
            
            metrics = ["acc", "bal_acc", "f1"]
            agg_lin = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_mlp = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_ae  = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_q   = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            
            successful_seeds = 0
            
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
                    successful_seeds += 1
                            
                except FileNotFoundError as e:
                    print(f"   [WARNING] Skipping seed {seed}: {e}")
                    continue

            if successful_seeds == 0:
                print(f"   [!] No completed Phase 2 weights found for {dataset} at frac {frac}. Skipping evaluation.")
                continue

            # Average the curves
            avg_lin_curve, avg_mlp_curve, avg_ae_curve, avg_q_curve = {}, {}, {}, {}
            
            for sigma in NOISE_LEVELS:
                s_key = f"{sigma:.2f}"
                avg_lin_curve[s_key] = {}
                avg_mlp_curve[s_key] = {}
                avg_ae_curve[s_key] = {}
                avg_q_curve[s_key] = {}
                
                for m in metrics:
                    avg_lin_curve[s_key][f"mean_{m}"] = float(np.mean(agg_lin[s_key][m]))
                    avg_lin_curve[s_key][f"std_{m}"] = float(np.std(agg_lin[s_key][m]))
                    
                    avg_mlp_curve[s_key][f"mean_{m}"] = float(np.mean(agg_mlp[s_key][m]))
                    avg_mlp_curve[s_key][f"std_{m}"] = float(np.std(agg_mlp[s_key][m]))

                    avg_ae_curve[s_key][f"mean_{m}"] = float(np.mean(agg_ae[s_key][m]))
                    avg_ae_curve[s_key][f"std_{m}"] = float(np.std(agg_ae[s_key][m]))
                    
                    avg_q_curve[s_key][f"mean_{m}"] = float(np.mean(agg_q[s_key][m]))
                    avg_q_curve[s_key][f"std_{m}"] = float(np.std(agg_q[s_key][m]))

            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear_avg": avg_lin_curve,
                "classical_mlp_avg": avg_mlp_curve,
                "classical_deep_ae_avg": avg_ae_curve,
                "quantum_avg": avg_q_curve
            }
            
            # Print diagnostic snapshot at noise = 0.10
            print(f"\n   [AVERAGED MACRO-F1 DECAY AT SIGMA=0.10 (Seeds: {successful_seeds})]")
            print(f"   Linear F1:      {avg_lin_curve['0.10']['mean_f1']:.4f}")
            print(f"   MLP F1:         {avg_mlp_curve['0.10']['mean_f1']:.4f}")
            print(f"   Deep AE F1:     {avg_ae_curve['0.10']['mean_f1']:.4f}")
            print(f"   Quantum F1:     {avg_q_curve['0.10']['mean_f1']:.4f}")
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRobustness evaluation complete. All metrics securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()