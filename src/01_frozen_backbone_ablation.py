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
DATASET = "breastmnist"
DATA_FRACTION = 1.0  # Ablation conducted on full data structure first
BATCH_SIZE = 32
EPOCHS = 50          # Maximum epochs as defined in Section V.B
LR_HEAD = 1e-3       # Classification head learning rate
SEED = 42
RESULTS_FILE = "results/frozen_ablation_logs.json"

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
    # Handle edge case where a batch might only have one class during severe scarcity
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 
        
    return avg_loss, float(auc)

def train_ablation_model(model, train_loader, val_loader, test_loader, device, model_name="Model"):
    """
    Trains the model using the strict Frozen-Backbone regime.
    """
    print(f"\n--- Training {model_name} (Frozen Backbone) ---")
    
    # STRICT OVERRIDE: Ensure the ENTIRE backbone is frozen for this ablation study
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Optimizer only updates parameters that require gradients (the bottleneck and head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    criterion = nn.BCEWithLogitsLoss()
    
    # Scheduler to prevent oscillatory divergence in quantum parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_val_auc = 0.0
    best_weights = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        # Validation Checkpoint
        val_loss, val_auc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch+1:02d} | Val AUC: {val_auc:.4f} **(New Best)**")
            
    # Load optimal weights for final pristine test evaluation
    print(f"\nLoading optimal weights (Best Val AUC: {best_val_auc:.4f}) for Test Set Evaluation...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    test_loss, test_auc = evaluate_epoch(model, test_loader, criterion, device)
    print(f"[{model_name}] Final Test AUC: {test_auc:.4f}")
    
    return test_auc

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_medmnist_loaders(
        dataset_name=DATASET, batch_size=BATCH_SIZE, train_frac=DATA_FRACTION, seed=SEED
    )
    
    # 2. Initialize Models
    classical_model = ClassicalResNetBottleneck(bottleneck_dim=4).to(device)
    quantum_model = QuantumHybridResNet(n_qubits=4, n_layers=2).to(device)
    
    # 3. Execute Ablation Training
    classical_test_auc = train_ablation_model(classical_model, train_loader, val_loader, test_loader, device, "Classical Baseline")
    quantum_test_auc = train_ablation_model(quantum_model, train_loader, val_loader, test_loader, device, "Quantum VQC Hybrid")
    
    # 4. Save Artifacts
    results = {
        "experiment": "Frozen-Backbone Ablation",
        "dataset": DATASET,
        "metrics": {
            "classical_auc": classical_test_auc,
            "quantum_auc": quantum_test_auc
        }
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Results securely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()