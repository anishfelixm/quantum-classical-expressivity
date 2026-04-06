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
from src.models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"]
# Dynamically target ~40-50 images (10% of 546 Breast, 1% of 4708 Pneumonia)
SCARCITY_TARGETS = {"breastmnist": 0.10, "pneumoniamnist": 0.01} 

BATCH_SIZE = 32
EPOCHS = 50          
LR_HEAD = 1e-3       
LR_QUANTUM = 5e-3 # Dropped slightly for stability while avoiding barren plateaus
SEEDS = [42, 123, 2026] 
RESULTS_FILE = "results/frozen_ablation_logs.json"

def evaluate_epoch(model, dataloader, criterion, device, threshold=None):
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
        auc = 0.5 
        
    # --- DYNAMIC THRESHOLDING FOR POS_WEIGHT ---
    if threshold is None:
        # If no threshold provided (Validation Phase), find the one that maximizes F1
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        # Avoid division by zero
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        # precision_recall_curve returns thresholds of length len(precisions)-1
        if best_idx < len(thresholds):
            best_thresh = thresholds[best_idx]
        else:
            best_thresh = 0.5
    else:
        # If threshold provided (Test Phase), use the locked-in validation threshold
        best_thresh = threshold

    preds = [1 if p >= best_thresh else 0 for p in all_probs]
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
        
    return avg_loss, float(auc), float(acc), float(f1), float(best_thresh)

def train_ablation_model(model, train_loader, val_loader, test_loader, device, model_name, dataset_name, seed):
    print(f"\n      Training {model_name}...")
    
    # 1. STRICT OVERRIDE: Ensure the ENTIRE backbone is completely frozen
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            
    head_params = []
    quantum_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "q_layer" in name:
            quantum_params.append(param)
        else:
            head_params.append(param)
            
    # 2. OPTIMIZER: Strict isolation of parameters
    optimizer = optim.Adam([
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': quantum_params, 'lr': LR_QUANTUM, 'weight_decay': 0.0}
    ])

    # 3. ANTI-COLLAPSE: Calculate global pos_weight 
    num_pos = 0
    num_neg = 0
    for _, y_batch in train_loader:
        num_pos += y_batch.sum().item()
        num_neg += (len(y_batch) - y_batch.sum().item())
    
    pos_weight_val = num_neg / (num_pos + 1e-7) 
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # STEPPING ON VAL_LOSS: The mathematically sound middle-ground
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_auc = 0.0
    best_weights = None
    best_locked_threshold = 0.5
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Keep BatchNorm frozen
        for name, module in model.named_modules():
            if "backbone" in name:
                module.eval()
                
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # Evaluate Validation
        val_loss, val_auc, val_acc, val_f1, current_thresh = evaluate_epoch(model, val_loader, criterion, device)
        
        # Step on VAL_LOSS
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_locked_threshold = current_thresh # Lock in the best threshold
            best_weights = copy.deepcopy(model.state_dict())
            print(f"         Epoch {epoch+1:02d} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} **(Best)** | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Thresh: {current_thresh:.2f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
        safe_name = model_name.replace(' ', '_')
        torch.save(best_weights, f"results/best_ablation_{safe_name}_{dataset_name}_seed{seed}.pt")
        
    # Evaluate Test using the locked threshold from the best validation epoch
    test_loss, test_auc, test_acc, test_f1, _ = evaluate_epoch(model, test_loader, criterion, device, threshold=best_locked_threshold)
    print(f"         -> Final Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Used Thresh: {best_locked_threshold:.2f}")
    
    return test_auc, test_acc, test_f1, history

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    results = {"experiment": "Frozen-Backbone Ablation (Layer 3)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        # Build our test fractions dynamically
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
                
                # Instantiate models fresh for each seed
                linear_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
                mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
                quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
                
                lin_auc, lin_acc, lin_f1, lin_hist = train_ablation_model(linear_model, train_loader, val_loader, test_loader, device, "Classical Linear", dataset, seed)
                mlp_auc, mlp_acc, mlp_f1, mlp_hist = train_ablation_model(mlp_model, train_loader, val_loader, test_loader, device, "Classical MLP", dataset, seed)
                q_auc, q_acc, q_f1, q_hist = train_ablation_model(quantum_model, train_loader, val_loader, test_loader, device, "Quantum VQC", dataset, seed)
                
                # Append metrics
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
            
            # Print Averages
            print(f"\n   [AVERAGE RESULTS ACROSS {len(SEEDS)} SEEDS]")
            lin_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_linear"]["test_acc"])
            mlp_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["classical_mlp"]["test_acc"])
            q_avg_acc = np.mean(results["datasets"][dataset]["fractions"][str(frac)]["quantum"]["test_acc"])
            
            print(f"   Linear Acc: {lin_avg_acc:.4f} | MLP Acc: {mlp_avg_acc:.4f} | Quantum Acc: {q_avg_acc:.4f}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Multi-seed results safely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()