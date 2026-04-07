"""
End-to-End Hybrid Quantum-Classical Fine-Tuning.

This script unfreezes the final convolutional block (Layer 3) of the classical feature 
extractor. It allows gradients from the Variational Quantum Circuit (VQC) to backpropagate, 
actively reshaping the classical feature maps into a quantum-friendly geometry under 
severe information constraints.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet
from models.quantum_vqc import QuantumHybridResNet

# --- EXPERIMENT CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"]
SCARCITY_TARGETS = {"breastmnist": 0.10, "pneumoniamnist": 0.01} 

BATCH_SIZE = 32
EPOCHS = 50                 
LR_BACKBONE = 1e-4          # Conservative LR for Layer 3 to mitigate catastrophic overfitting
LR_HEAD = 1e-3              # Standard LR for classical latent projection heads
LR_QUANTUM = 5e-3           # VQC learning rate aligned with ablation study baseline
SEEDS = [42, 123, 2026] 
RESULTS_FILE = "results/end_to_end_logs.json"

def evaluate_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device, threshold: float = None):
    """
    Evaluates model performance and computes standard classification metrics.
    If no threshold is provided (validation phase), dynamically calculates the optimal 
    decision boundary via the Precision-Recall curve to maximize F1 under class imbalance.
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 # Fallback for ill-defined AUC in extremely small, single-class batches
        
    # Phase-dependent dynamic thresholding
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        best_thresh = threshold

    preds = [1 if p >= best_thresh else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
        
    return avg_loss, float(auc), float(acc), float(f1), float(best_thresh)


def train_finetune_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, device: torch.device, dataset_name: str, model_name: str, seed: int):
    """
    Executes the End-to-End training protocol.
    Enforces a strict parameter hierarchy: Layer 3 is unfrozen but throttled by a 
    lower learning rate, while earlier layers remain permanently immobilized to preserve 
    low-level feature extraction capabilities.
    """
    print(f"\n      Training {model_name} (End-to-End)...")

    # 1. Targeted Unfreezing of Layer 3
    # In nn.Sequential[:-3], index 6 corresponds exactly to ResNet Layer 3.
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = "backbone.6" in name
    
    backbone_params, head_params, quantum_params = [], [], []
    
    # Parameter routing for differential optimization
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.6" in name: 
            backbone_params.append(param)
        elif "q_layer" in name:
            quantum_params.append(param)
        else:
            head_params.append(param)
            
    # 2. Differential Learning Rates
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': quantum_params, 'lr': LR_QUANTUM, 'weight_decay': 0.0} 
    ])

    # 3. Class Imbalance Mitigation
    # Calculate global positive weight for BCE Loss using a single loop to preserve RNG state.
    num_pos = 0
    num_neg = 0
    for _, y_batch in train_loader:
        num_pos += y_batch.sum().item()
        num_neg += (len(y_batch) - y_batch.sum().item())
    
    pos_weight_val = num_neg / (num_pos + 1e-7) 
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_auc, best_locked_threshold = 0.0, 0.5
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(EPOCHS):
        model.train()

        # BatchNorm Protocol: Freeze statistics for immobilized layers; allow Layer 3 to update.
        for name, module in model.named_modules():
            if "backbone" in name and "backbone.6" not in name:
                module.eval()
        
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient Clipping: Mitigate gradient explosion in classical parameters
            torch.nn.utils.clip_grad_norm_(backbone_params + head_params, max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_auc, val_acc, val_f1, current_thresh = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_locked_threshold = current_thresh
            best_weights = copy.deepcopy(model.state_dict())
            print(f"         Epoch {epoch+1:02d} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} **(Best)** | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Thresh: {current_thresh:.2f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        safe_name = model_name.replace(' ', '_')
        torch.save(best_weights, f"results/best_e2e_{safe_name}_{dataset_name}_seed{seed}.pt")
        
    test_loss, test_auc, test_acc, test_f1, _ = evaluate_epoch(model, test_loader, criterion, device, threshold=best_locked_threshold)
    print(f"         -> Final Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Used Thresh: {best_locked_threshold:.2f}")
    
    return test_auc, test_acc, test_f1, history


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "End-to-End Fine-Tuning (Layer 3 Unfrozen)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        fractions_to_test = [SCARCITY_TARGETS[dataset], 1.0]
        
        for frac in fractions_to_test:
            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear": {"test_auc": [], "test_acc": [], "test_f1": [], "history": []},
                "classical_mlp": {"test_auc": [], "test_acc": [], "test_f1": [], "history": []},
                "quantum": {"test_auc": [], "test_acc": [], "test_f1": [], "history": []}
            }
            
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | DATA FRACTION: {frac*100}%")
            print(f"=====================================================")
            
            for seed in SEEDS:
                print(f"\n   --- RUNNING SEED: {seed} ---")
                
                train_loader, val_loader, test_loader = get_medmnist_loaders(
                    dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=frac, seed=seed
                )
                
                linear_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
                mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
                quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
                
                lin_auc, lin_acc, lin_f1, lin_hist = train_finetune_model(linear_model, train_loader, val_loader, test_loader, device, dataset, "Classical Linear", seed)
                mlp_auc, mlp_acc, mlp_f1, mlp_hist = train_finetune_model(mlp_model, train_loader, val_loader, test_loader, device, dataset, "Classical MLP", seed)
                q_auc, q_acc, q_f1, q_hist = train_finetune_model(quantum_model, train_loader, val_loader, test_loader, device, dataset, "Quantum VQC", seed)
                
                frac_results = results["datasets"][dataset]["fractions"][str(frac)]
                frac_results["classical_linear"]["test_auc"].append(lin_auc)
                frac_results["classical_linear"]["test_acc"].append(lin_acc)
                frac_results["classical_linear"]["test_f1"].append(lin_f1)
                frac_results["classical_linear"]["history"].append(lin_hist)
                
                frac_results["classical_mlp"]["test_auc"].append(mlp_auc)
                frac_results["classical_mlp"]["test_acc"].append(mlp_acc)
                frac_results["classical_mlp"]["test_f1"].append(mlp_f1)
                frac_results["classical_mlp"]["history"].append(mlp_hist)
                
                frac_results["quantum"]["test_auc"].append(q_auc)
                frac_results["quantum"]["test_acc"].append(q_acc)
                frac_results["quantum"]["test_f1"].append(q_f1)
                frac_results["quantum"]["history"].append(q_hist)
            
            # DYNAMIC AVERAGING: Calculate final metrics
            lin_avg_auc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_linear"]["test_auc"])
            mlp_avg_auc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_mlp"]["test_auc"])
            q_avg_auc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["quantum"]["test_auc"])
            
            lin_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_linear"]["test_acc"])
            mlp_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_mlp"]["test_acc"])
            q_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["quantum"]["test_acc"])
            
            print(f"\n   [AVERAGE RESULTS ACROSS {len(SEEDS)} SEEDS]")
            print(f"   Linear AUC: {lin_avg_auc:.4f} | MLP AUC: {mlp_avg_auc:.4f} | Quantum AUC: {q_avg_auc:.4f}")
            print(f"   Linear Acc: {lin_avg_acc:.4f} | MLP Acc: {mlp_avg_acc:.4f} | Quantum Acc: {q_avg_acc:.4f}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Multi-seed results safely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()