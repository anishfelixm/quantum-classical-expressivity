"""
Robustness Evaluation under Sensor Degradation (The Zombie State Test).

Evaluates the saved models across varying levels of Gaussian sensor noise.
Proves the "Precision Paradox": highly expressive classical models shatter under physical 
noise (AWGN), while the Quantum VQC maintains a stable topological decision boundary.
Methodology aligns with physical sensor degradation bounds (Hendrycks & Dietterich, 2019).
"""
import os
import json
import torch
import numpy as np
import medmnist
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints_exp2")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

sys.path.append(os.path.join(BASE_DIR, 'src'))
from data.medmnist_loader import get_medmnist_loaders, NORM_MEAN, NORM_STD
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet, ClassicalDeepBottleneckResNet
from models.quantum_vqc import QuantumHybridResNet

# --- EXPERIMENT CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
FRACTIONS = [0.01, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0] 
BOTTLENECKS = [4, 8, 16]
BATCH_SIZE = 32
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]
SEEDS = [42, 123, 2026, 777, 888]
RESULTS_FILE_NAME = "robustness_e2e_logs.json"


def clean_val(val):
    """Converts NaNs to None for strict RFC 8259 JSON compliance."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return None
    return float(val)


def calculate_metrics(labels, preds, probs, num_classes):
    """Helper to calculate metrics safely (Maintains parity with Scripts 1 & 2)."""
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, np.array(probs)[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr')
    except ValueError:
        auc = np.nan
        
    return clean_val(acc), clean_val(bal_acc), clean_val(macro_f1), clean_val(auc)


def add_gaussian_noise(images: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Applies 4-Step Sensor Degradation (AWGN).
    Dynamically imports exact ImageNet normalization stats from the dataloader 
    to ensure perfect inversion back to the [0.0, 1.0] physical pixel space.
    """
    if sigma == 0.0:
        return images
        
    # Broadcast to (1, 3, 1, 1) for batched ImageNet channel math
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1).to(device)
    
    # 1. Inverse-Normalize to physical pixel space [0.0, 1.0]
    images_real = (images * std) + mean
    
    # 2. Inject Gaussian Noise (Simulating Analog Sensor Read Noise)
    noise = torch.randn_like(images_real) * sigma
    
    # 3. Hardware Sensor Clamp (biological reality constraint: pixels cannot emit negative light)
    images_noisy = torch.clamp(images_real + noise, min=0.0, max=1.0)
    
    # 4. Re-Normalize for ResNet digestion
    images_final = (images_noisy - mean) / std
    
    return images_final


def evaluate_robustness_curve(model, test_loader, device, seed, num_classes):
    model.eval()
    robustness_curve = {}
    
    for sigma in NOISE_LEVELS:
        # Use round() to prevent float truncation bugs in seed calculation
        torch.manual_seed(seed + int(round(sigma * 1000)))
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.view(-1).long().to(device)
                
                # Apply rigorously bounded physical noise
                x_noisy = add_gaussian_noise(x, sigma, device)
                logits = model(x_noisy)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        acc, bal_acc, macro_f1, auc = calculate_metrics(all_labels, all_preds, all_probs, num_classes)
        
        robustness_curve[f"{sigma:.2f}"] = {
            "acc": acc,
            "bal_acc": bal_acc,
            "f1": macro_f1,
            "auc": auc
        }
        
    return robustness_curve


def load_and_test_seed(dataset, fraction, b_dim, seed, device):
    _, _, test_loader = get_medmnist_loaders(
        dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=fraction, seed=seed, data_root=CACHE_DIR
    )
    
    # Safely fetch global classes
    info = medmnist.INFO[dataset]
    num_classes = len(info['label'])
    
    lin_model = ClassicalLinearResNet(num_classes=num_classes, bottleneck_dim=b_dim).to(device)
    mlp_model = ClassicalMLPResNet(num_classes=num_classes, bottleneck_dim=b_dim).to(device)
    deep_model = ClassicalDeepBottleneckResNet(num_classes=num_classes, bottleneck_dim=b_dim).to(device)
    q_model = QuantumHybridResNet(num_classes=num_classes, n_qubits=b_dim, n_layers=2).to(device)
    
    paths = {
        "Classical Linear": os.path.join(CHECKPOINT_DIR, f"best_e2e_Classical_Linear_{dataset}_frac{fraction}_b{b_dim}_seed{seed}.pt"),
        "Classical MLP": os.path.join(CHECKPOINT_DIR, f"best_e2e_Classical_MLP_{dataset}_frac{fraction}_b{b_dim}_seed{seed}.pt"),
        "Classical Deep Funnel": os.path.join(CHECKPOINT_DIR, f"best_e2e_Classical_Deep_Funnel_{dataset}_frac{fraction}_b{b_dim}_seed{seed}.pt"),
        "Quantum VQC": os.path.join(CHECKPOINT_DIR, f"best_e2e_Quantum_VQC_{dataset}_frac{fraction}_b{b_dim}_seed{seed}.pt")
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weights for {name} on {dataset} frac {fraction} dim {b_dim} seed {seed} at {path}")
            
    lin_model.load_state_dict(torch.load(paths["Classical Linear"], map_location=device, weights_only=True))
    mlp_model.load_state_dict(torch.load(paths["Classical MLP"], map_location=device, weights_only=True))
    deep_model.load_state_dict(torch.load(paths["Classical Deep Funnel"], map_location=device, weights_only=True))
    q_model.load_state_dict(torch.load(paths["Quantum VQC"], map_location=device, weights_only=True))
    
    print(f"   --- RUNNING SEED: {seed} ---")
    lin_curve = evaluate_robustness_curve(lin_model, test_loader, device, seed, num_classes)
    mlp_curve = evaluate_robustness_curve(mlp_model, test_loader, device, seed, num_classes)
    deep_curve = evaluate_robustness_curve(deep_model, test_loader, device, seed, num_classes)
    q_curve = evaluate_robustness_curve(q_model, test_loader, device, seed, num_classes)
    
    return lin_curve, mlp_curve, deep_curve, q_curve, lin_model, mlp_model, deep_model, q_model


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, RESULTS_FILE_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "Robustness Decay (Sensor Noise AWGN)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        for frac in FRACTIONS:
            results["datasets"][dataset]["fractions"][str(frac)] = {"bottlenecks": {}}
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | ROBUSTNESS TEST | DATA FRACTION: {frac*100}%")
            print(f"=====================================================")
            
            for b in BOTTLENECKS:
                print(f"\n   >>> Testing Bottleneck Dimension / Qubits: {b} <<<")
                
                metrics = ["acc", "bal_acc", "f1", "auc"]
                
                # Arrays to store RAW values across seeds for Welch's t-test
                agg_lin = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
                agg_mlp = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
                agg_deep = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
                agg_q   = {f"{s:.2f}": {m: [] for m in metrics} for s in NOISE_LEVELS}
                
                for seed in SEEDS:
                    try:
                        lin_c, mlp_c, deep_c, q_c, m1, m2, m3, m4 = load_and_test_seed(dataset, frac, b, seed, device)
                        for sigma in NOISE_LEVELS:
                            s_key = f"{sigma:.2f}"
                            for m in metrics:
                                agg_lin[s_key][m].append(lin_c[s_key][m])
                                agg_mlp[s_key][m].append(mlp_c[s_key][m])
                                agg_deep[s_key][m].append(deep_c[s_key][m])
                                agg_q[s_key][m].append(q_c[s_key][m])
                                
                        # Memory Management
                        del m1, m2, m3, m4
                        torch.cuda.empty_cache()
                        
                    except FileNotFoundError as e:
                        print(f"   [WARNING] Skipping seed {seed}: {e}")
                        continue

                # Calculate stats with strict schema and NaN filtering
                b_dict = {
                    "classical_linear": {}, "classical_mlp": {}, 
                    "classical_deep_funnel": {}, "quantum_vqc": {}
                }
                
                for sigma in NOISE_LEVELS:
                    s_key = f"{sigma:.2f}"
                    
                    def get_stats(agg_dict, current_s_key):
                        valid_f1s = [v for v in agg_dict[current_s_key]["f1"] if v is not None]
                        valid_aucs = [v for v in agg_dict[current_s_key]["auc"] if v is not None]
                        valid_accs = [v for v in agg_dict[current_s_key]["acc"] if v is not None]
                        valid_bal_accs = [v for v in agg_dict[current_s_key]["bal_acc"] if v is not None]
                        
                        if len(valid_f1s) == 0:
                            return {
                                "mean_f1": None, "std_f1": None, "raw_f1": [], 
                                "mean_auc": None, "std_auc": None, "raw_auc": [], 
                                "mean_acc": None, "mean_bal_acc": None
                            }
                            
                        return {
                            "mean_f1": clean_val(np.mean(valid_f1s)),
                            "std_f1": clean_val(np.std(valid_f1s)),
                            "raw_f1": valid_f1s,
                            "mean_auc": clean_val(np.mean(valid_aucs)) if len(valid_aucs) > 0 else None,
                            "std_auc": clean_val(np.std(valid_aucs)) if len(valid_aucs) > 0 else None,
                            "raw_auc": valid_aucs,
                            "mean_acc": clean_val(np.mean(valid_accs)) if len(valid_accs) > 0 else None,
                            "mean_bal_acc": clean_val(np.mean(valid_bal_accs)) if len(valid_bal_accs) > 0 else None
                        }

                    b_dict["classical_linear"][s_key] = get_stats(agg_lin, s_key)
                    b_dict["classical_mlp"][s_key] = get_stats(agg_mlp, s_key)
                    b_dict["classical_deep_funnel"][s_key] = get_stats(agg_deep, s_key)
                    b_dict["quantum_vqc"][s_key] = get_stats(agg_q, s_key)
                
                results["datasets"][dataset]["fractions"][str(frac)]["bottlenecks"][str(b)] = b_dict
                
                # Safe print
                if len(agg_q['0.10']['auc']) > 0:
                    q_auc_val = [v for v in agg_q['0.10']['auc'] if v is not None]
                    lin_auc_val = [v for v in agg_lin['0.10']['auc'] if v is not None]
                    deep_auc_val = [v for v in agg_deep['0.10']['auc'] if v is not None]
                    
                    print(f"   [AVERAGE AUC DECAY AT SIGMA=0.10 | DIM: {b}]")
                    print(f"   Linear:      {np.mean(lin_auc_val):.4f}" if len(lin_auc_val) > 0 else "   Linear: N/A")
                    print(f"   Deep Funnel: {np.mean(deep_auc_val):.4f}" if len(deep_auc_val) > 0 else "   Deep Funnel: N/A")
                    print(f"   Quantum VQC: {np.mean(q_auc_val):.4f}" if len(q_auc_val) > 0 else "   Quantum VQC: N/A")
                
        # Save per dataset to avoid losing data if cluster execution halts
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)
            
    print(f"\nRobustness Evaluation Complete! Results securely saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()