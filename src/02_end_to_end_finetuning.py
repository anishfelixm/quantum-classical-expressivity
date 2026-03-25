import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalResNetBottleneck
from models.quantum_vqc import QuantumHybridResNet

# --- CONFIGURATION ---
DATASET = "pneumoniamnist"  # Change to "breastmnist" to run the other dataset
DATA_FRACTIONS = [0.1, 1.0] # 10% scarcity regime vs 100% full dataset
BATCH_SIZE = 32
EPOCHS = 50                 # Maximum epochs
LR_BACKBONE = 1e-4          # Conservative LR for ResNet layer4 (Section V.B)
LR_HEAD = 1e-3              # Faster LR for Bottleneck and Classification Head
SEED = 42
RESULTS_FILE = f"results/end_to_end_logs_{DATASET}.json"

def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model and returns average loss and AUC-ROC."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.float().to(device)
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
        
    return avg_loss, float(auc)

def train_finetune_model(model, train_loader, val_loader, test_loader, device, model_name="Model"):
    """
    Trains the model using the End-to-End Fine-Tuning regime with Differential LRs.
    """
    print(f"\n--- Training {model_name} (End-to-End Fine-Tuning) ---")
    
    # Differential Learning Rate setup (Section V.B)
    # 1. layer4 gets the conservative 1e-4 LR
    # 2. Bottleneck and Heads get the standard 1e-3 LR
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.7" in name: # "7" corresponds to layer4 in the ResNet sequential model
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD}
    ])
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_val_auc = 0.0
    best_weights = None
    history = {"train_loss": [], "val_auc": []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation Checkpoint
        val_loss, val_auc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} **(New Best)**")
            
    # Load optimal weights for final test evaluation
    print(f"\nLoading optimal weights (Best Val AUC: {best_val_auc:.4f}) for Test Set Evaluation...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
        # NOTE: For the robustness test, you will need to save the actual model weights!
        torch.save(best_weights, f"results/best_{model_name.replace(' ', '_')}_{DATASET}.pt")
        
    test_loss, test_auc = evaluate_epoch(model, test_loader, criterion, device)
    print(f"[{model_name}] Final Test AUC: {test_auc:.4f}")
    
    return test_auc, history

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}\n")
    
    results = {"experiment": "End-to-End Fine-Tuning", "dataset": DATASET, "fractions": {}}
    
    for frac in DATA_FRACTIONS:
        print(f"=====================================================")
        print(f"   EVALUATING DATA FRACTION: {frac*100}%")
        print(f"=====================================================")
        
        train_loader, val_loader, test_loader = get_medmnist_loaders(
            dataset_name=DATASET, batch_size=BATCH_SIZE, train_frac=frac, seed=SEED
        )
        
        # Re-initialize models fresh for each fraction
        classical_model = ClassicalResNetBottleneck(bottleneck_dim=4).to(device)
        quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
        
        c_test_auc, c_hist = train_finetune_model(classical_model, train_loader, val_loader, test_loader, device, f"Classical_{frac}")
        q_test_auc, q_hist = train_finetune_model(quantum_model, train_loader, val_loader, test_loader, device, f"Quantum_{frac}")
        
        results["fractions"][str(frac)] = {
            "classical": {"test_auc": c_test_auc, "history": c_hist},
            "quantum": {"test_auc": q_test_auc, "history": q_hist}
        }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Results securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()