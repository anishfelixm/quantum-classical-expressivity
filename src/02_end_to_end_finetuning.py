import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet
from models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"] 
DATA_FRACTIONS = [0.1, 1.0] # 10% scarcity regime vs 100% full dataset
BATCH_SIZE = 32
EPOCHS = 50                 
LR_BACKBONE = 1e-4          # Conservative LR for ResNet layer4 to prevent forgetting
LR_HEAD = 1e-3              # Faster LR for Bottleneck and Classification Heads
LR_QUANTUM = 1e-2
SEED = 42
RESULTS_FILE = "results/end_to_end_logs.json"

def evaluate_epoch(model, dataloader, criterion, device):
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
    
    # Standard threshold of 0.5 for Accuracy and F1
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 
        
    return avg_loss, float(auc), float(acc), float(f1)

def train_finetune_model(model, train_loader, val_loader, test_loader, device, dataset_name, model_name="Model"):
    print(f"\n--- Training {model_name} on {dataset_name} ---")

    # 1. FREEZING LOGIC: Explicitly freeze early ResNet layers
    for name, param in model.named_parameters():
        if "backbone" in name and "backbone.7" not in name:
            param.requires_grad = False
    
    backbone_params = []
    head_params = []
    quantum_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.7" in name: 
            backbone_params.append(param)
        elif "q_layer" in name:
            quantum_params.append(param)
        else:
            head_params.append(param)
            
    # 2. OPTIMIZER: Differential LRs applied fairly to all models
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': quantum_params, 'lr': LR_QUANTUM, 'weight_decay': 0.0} # Protected VQC params
    ])

    # Standard BCE: Crucial for evaluating bounded quantum observables fairly
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_auc = 0.0
    best_weights = None
    history = {"train_loss": [], "val_auc": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(EPOCHS):
        model.train()

        # Robust BatchNorm Freezing for Scarcity Regimes
        for name, module in model.named_modules():
            if "backbone" in name and "7" not in name:
                module.eval()
        
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient Clipping: Isolated to classical parameters to protect quantum topology
            torch.nn.utils.clip_grad_norm_(
                backbone_params + head_params, 
                max_norm=1.0
            )
            
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        val_loss, val_auc, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(train_loss)
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} **(New Best)** | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    print(f"\nLoading optimal weights for Test Set Evaluation...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        safe_name = model_name.replace(' ', '_')
        torch.save(best_weights, f"results/best_{safe_name}_{dataset_name}.pt")
        
    test_loss, test_auc, test_acc, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
    print(f"[{model_name}] Final Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    
    return test_auc, test_acc, test_f1, history

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "End-to-End Fine-Tuning", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        for frac in DATA_FRACTIONS:
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | DATA FRACTION: {frac*100}%")
            print(f"=====================================================")
            
            train_loader, val_loader, test_loader = get_medmnist_loaders(
                dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=frac, seed=SEED
            )
            
            linear_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
            mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
            quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
            
            lin_test_auc, lin_test_acc, lin_test_f1, lin_hist = train_finetune_model(linear_model, train_loader, val_loader, test_loader, device, dataset, f"Linear_{frac}")
            mlp_test_auc, mlp_test_acc, mlp_test_f1, mlp_hist = train_finetune_model(mlp_model, train_loader, val_loader, test_loader, device, dataset, f"MLP_{frac}")
            q_test_auc, q_test_acc, q_test_f1, q_hist = train_finetune_model(quantum_model, train_loader, val_loader, test_loader, device, dataset, f"Quantum_{frac}")
            
            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear": {"test_auc": lin_test_auc, "test_acc": lin_test_acc, "test_f1": lin_test_f1, "history": lin_hist},
                "classical_mlp": {"test_auc": mlp_test_auc, "test_acc": mlp_test_acc, "test_f1": mlp_test_f1, "history": mlp_hist},
                "quantum": {"test_auc": q_test_auc, "test_acc": q_test_acc, "test_f1": q_test_f1, "history": q_hist}
            }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Results securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()