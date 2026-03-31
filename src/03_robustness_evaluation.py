import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet
from models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"]
DATA_FRACTIONS = [0.1, 1.0] 
BATCH_SIZE = 32
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2]
SEED = 42
RESULTS_FILE = "results/robustness_logs.json"

def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Applies Sensor Degradation noise to the raw input images."""
    if sigma == 0.0:
        return images
    noise = torch.randn_like(images) * sigma
    return images + noise

def evaluate_robustness_curve(model, test_loader, device, model_name="Model"):
    model.eval()
    robustness_curve = {}
    print(f"--- Running Noise Stress Test for {model_name} ---")
    
    for sigma in NOISE_LEVELS:
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                # Explicit shape enforcement to prevent BCE broadcasting bugs
                x, y = x.to(device), y.view(-1, 1).float().to(device)
                
                # Apply Sensor Degradation
                x_noisy = add_gaussian_noise(x, sigma)
                
                # Forward Pass 
                logits = model(x_noisy)
                probs = torch.sigmoid(logits)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
            
        prob_std = np.std(all_probs)
        print(f"  Sigma = {sigma:4.2f} | AUC: {auc:.4f} | Acc: {acc:.4f}")
        
        robustness_curve[str(sigma)] = {
            "auc": float(auc),
            "acc": float(acc),
            "prob_std": float(prob_std) 
        }
        
    return robustness_curve

def load_and_test(dataset, fraction, device):
    print(f"\n=====================================================")
    print(f"   {dataset.upper()} | ROBUSTNESS: {fraction*100}% REGIME")
    print(f"=====================================================")
    
    _, _, test_loader = get_medmnist_loaders(
        dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=fraction, seed=SEED
    )
    
    # Initialize all three architectures
    lin_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
    mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
    q_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
    
    # Construct expected file paths from script 02
    lin_weight_path = f"results/best_Linear_{fraction}_{dataset}.pt"
    mlp_weight_path = f"results/best_MLP_{fraction}_{dataset}.pt"
    q_weight_path = f"results/best_Quantum_{fraction}_{dataset}.pt"
    
    if not os.path.exists(lin_weight_path) or not os.path.exists(q_weight_path) or not os.path.exists(mlp_weight_path):
        raise FileNotFoundError(f"Missing weights for {dataset} at {fraction}. Run script 02 first.")
    
    print("Loading optimal checkpoint weights...")
    lin_model.load_state_dict(torch.load(lin_weight_path, map_location=device))
    mlp_model.load_state_dict(torch.load(mlp_weight_path, map_location=device))
    q_model.load_state_dict(torch.load(q_weight_path, map_location=device))
    
    # Execute Sweeps
    lin_curve = evaluate_robustness_curve(lin_model, test_loader, device, f"Linear {fraction}")
    mlp_curve = evaluate_robustness_curve(mlp_model, test_loader, device, f"MLP {fraction}")
    q_curve = evaluate_robustness_curve(q_model, test_loader, device, f"Quantum {fraction}")
    
    return {
        "classical_linear_curve": lin_curve, 
        "classical_mlp_curve": mlp_curve, 
        "quantum_curve": q_curve
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    results = {"experiment": "Robustness Decay", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        for frac in DATA_FRACTIONS:
            try:
                results["datasets"][dataset]["fractions"][str(frac)] = load_and_test(dataset, frac, device)
            except FileNotFoundError as e:
                print(f"\n[ERROR] {e}\nSkipping...")
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRobustness evaluation complete. Results securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()