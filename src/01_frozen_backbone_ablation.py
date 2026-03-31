import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet
from models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist"]
DATA_FRACTION = 1.0  
BATCH_SIZE = 32
EPOCHS = 50          
LR_HEAD = 1e-3       # Uniform LR for all classification heads (Fair comparison)
SEED = 42
RESULTS_FILE = "results/frozen_ablation_logs.json"

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
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 
        
    return avg_loss, float(auc), float(acc)

def train_ablation_model(model, train_loader, val_loader, test_loader, device, model_name="Model"):
    print(f"\n--- Training {model_name} (Frozen Backbone) ---")
    
    # 1. STRICT OVERRIDE: Ensure the ENTIRE backbone is frozen
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # 2. OPTIMIZER: Apply L2 Regularization uniformly
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR_HEAD, 
        weight_decay=1e-4 
    )

    # 3. CLASS IMBALANCE: Dynamically calculate pos_weight
    num_pos = sum(y.sum().item() for _, y in train_loader)
    num_neg = len(train_loader.dataset) - num_pos
    pos_weight_val = num_neg / (num_pos + 1e-5)
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_val_auc = 0.0
    best_weights = None
    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1, 1).float().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient Clipping (Protects all models equally)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
            
        # Validation Checkpoint
        val_loss, val_auc, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} **(New Best)** | Val Acc: {val_acc:.4f}")
            
    print(f"\nLoading optimal weights (Best Val AUC: {best_val_auc:.4f}) for Test Set Evaluation...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    test_loss, test_auc, test_acc = evaluate_epoch(model, test_loader, criterion, device)
    print(f"[{model_name}] Final Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    
    return test_auc, test_acc, history

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    results = {"experiment": "Frozen-Backbone Ablation", "datasets": {}}
    
    for dataset in DATASETS:
        print(f"\n=====================================================")
        print(f"   STARTING ABLATION STUDY: {dataset.upper()}")
        print(f"=====================================================")
        
        train_loader, val_loader, test_loader = get_medmnist_loaders(
            dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=DATA_FRACTION, seed=SEED
        )
        
        # Instantiate all three models
        linear_model = ClassicalLinearResNet(bottleneck_dim=4).to(device)
        mlp_model = ClassicalMLPResNet(bottleneck_dim=4).to(device)
        quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
        
        lin_test_auc, lin_test_acc, lin_hist = train_ablation_model(linear_model, train_loader, val_loader, test_loader, device, "Classical Linear Probe")
        mlp_test_auc, mlp_test_acc, mlp_hist = train_ablation_model(mlp_model, train_loader, val_loader, test_loader, device, "Classical MLP Head")
        q_test_auc, q_test_acc, q_hist = train_ablation_model(quantum_model, train_loader, val_loader, test_loader, device, "Quantum VQC Hybrid")
        
        results["datasets"][dataset] = {
            "classical_linear": {"test_auc": lin_test_auc, "test_acc": lin_test_acc, "history": lin_hist},
            "classical_mlp": {"test_auc": mlp_test_auc, "test_acc": mlp_test_acc, "history": mlp_hist},
            "quantum": {"test_auc": q_test_auc, "test_acc": q_test_acc, "history": q_hist}
        }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Consolidated results logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()