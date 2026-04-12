"""
Robustness Evaluation under Sensor Degradation.
Evaluates the decision boundary stability of End-to-End Hybrid Quantum-Classical 
models against injected Gaussian noise, utilizing optimal threshold locking.
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet
from models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"]
SCARCITY_TARGETS = {"breastmnist": 0.10, "pneumoniamnist": 0.01} 
BATCH_SIZE = 32
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2]
SEEDS = [42, 123, 2026]
RESULTS_FILE = "results/robustness_e2e_logs.json"

def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Applies Sensor Degradation noise and strictly clamps to physical bounds."""
    if sigma == 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    # Ensure pixels do not exceed biological/sensor validity [0, 1]
    return torch.clamp(images + noise, min=0.0, max=1.0)

def evaluate_robustness_curve(model, test_loader, device, seed, model_name="Model"):
    """
    Evaluates model decay across noise levels. 
    Crucially, it computes the optimal F1 threshold at sigma=0.0 and locks it 
    for all subsequent noise levels to measure true decision boundary degradation.
    """
    model.eval()
    robustness_curve = {}
    locked_thresh = 0.5
    
    for sigma in NOISE_LEVELS:
        # STRICT METHODOLOGICAL PARITY: 
        # Reset RNG uniquely for this sigma, but identically across architectures.
        # This guarantees all models face the exact same noise tensors per test sample.
        torch.manual_seed(seed + int(sigma * 1000))
        
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.view(-1, 1).float().to(device)
                
                x_noisy = add_gaussian_noise(x, sigma)
                
                logits = model(x_noisy)
                probs = torch.sigmoid(logits)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        # DYNAMIC THRESHOLD LOCKING (Only on Clean Data)
        if sigma == 0.0:
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            locked_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            print(f"      [{model_name}] Clean Threshold Locked at: {locked_thresh:.4f}")

        # Apply locked threshold to noisy predictions
        preds = [1 if p >= locked_thresh else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
            
        prob_std = np.std(all_probs)
        
        robustness_curve[f"{sigma:.2f}"] = {
            "auc": float(auc),
            "acc": float(acc),
            "f1": float(f1),
            "prob_std": float(prob_std) 
        }
        
    return robustness_curve

def load_and_test_seed(dataset, fraction, seed, device):
    _, _, test_loader = get_medmnist_loaders(
        dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=fraction, seed=seed
    )
    
    lin_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
    mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
    q_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
    
    lin_path = f"results/best_e2e_Classical_Linear_{dataset}_frac{fraction}_seed{seed}.pt"
    mlp_path = f"results/best_e2e_Classical_MLP_{dataset}_frac{fraction}_seed{seed}.pt"
    q_path = f"results/best_e2e_Quantum_VQC_{dataset}_frac{fraction}_seed{seed}.pt"
    
    if not os.path.exists(lin_path) or not os.path.exists(q_path) or not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Missing weights for {dataset} frac {fraction} seed {seed}.")
    
    lin_model.load_state_dict(torch.load(lin_path, map_location=device, weights_only=True))
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    q_model.load_state_dict(torch.load(q_path, map_location=device, weights_only=True))
    
    print(f"\n   --- RUNNING SEED: {seed} ---")
    lin_curve = evaluate_robustness_curve(lin_model, test_loader, device, seed, "Classical Linear")
    mlp_curve = evaluate_robustness_curve(mlp_model, test_loader, device, seed, "Classical MLP")
    q_curve = evaluate_robustness_curve(q_model, test_loader, device, seed, "Quantum VQC")
    
    return lin_curve, mlp_curve, q_curve

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "Robustness Decay (End-to-End Models)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        fractions_to_test = [SCARCITY_TARGETS[dataset], 1.0]
        
        for frac in fractions_to_test:
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | ROBUSTNESS: {frac*100}% DATA REGIME")
            print(f"=====================================================")
            
            metrics = ["auc", "acc", "f1", "prob_std"]
            agg_lin = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_mlp = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            agg_q =   {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
            
            for seed in SEEDS:
                try:
                    lin_curve, mlp_curve, q_curve = load_and_test_seed(dataset, frac, seed, device)
                    
                    for sigma in NOISE_LEVELS:
                        s_key = f"{sigma:.2f}"
                        for m in metrics:
                            agg_lin[s_key][m].append(lin_curve[s_key][m])
                            agg_mlp[s_key][m].append(mlp_curve[s_key][m])
                            agg_q[s_key][m].append(q_curve[s_key][m])
                        
                except FileNotFoundError as e:
                    print(f"[ERROR] {e}")
                    return

            avg_lin_curve, avg_mlp_curve, avg_q_curve = {}, {}, {}
            
            for sigma in NOISE_LEVELS:
                s_key = f"{sigma:.2f}"
                avg_lin_curve[s_key] = {}
                avg_mlp_curve[s_key] = {}
                avg_q_curve[s_key] = {}
                
                for m in metrics:
                    avg_lin_curve[s_key][f"mean_{m}"] = float(np.mean(agg_lin[s_key][m]))
                    avg_lin_curve[s_key][f"std_{m}"] = float(np.std(agg_lin[s_key][m]))
                    
                    avg_mlp_curve[s_key][f"mean_{m}"] = float(np.mean(agg_mlp[s_key][m]))
                    avg_mlp_curve[s_key][f"std_{m}"] = float(np.std(agg_mlp[s_key][m]))
                    
                    avg_q_curve[s_key][f"mean_{m}"] = float(np.mean(agg_q[s_key][m]))
                    avg_q_curve[s_key][f"std_{m}"] = float(np.std(agg_q[s_key][m]))

            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear_avg": avg_lin_curve,
                "classical_mlp_avg": avg_mlp_curve,
                "quantum_avg": avg_q_curve
            }
            
            print(f"\n   [AVERAGED F1 DECAY AT SIGMA=0.10]")
            print(f"   Linear F1: {avg_lin_curve['0.10']['mean_f1']:.4f}")
            print(f"   MLP F1:    {avg_mlp_curve['0.10']['mean_f1']:.4f}")
            print(f"   Quantum F1:{avg_q_curve['0.10']['mean_f1']:.4f}")
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRobustness evaluation complete. All metrics securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()