"""
Phase 1: Frozen Backbone Ablation Study.
Evaluates model expressivity under severe information constraints.
The classical backbone is permanently immobilized to isolate the representation
power of the bottlenecks (Linear, MLP, Deep Autoencoder, Quantum VQC).
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet, ClassicalDeepBottleneckResNet
from models.quantum_vqc import QuantumHybridResNet

# --- EXPERIMENT CONFIGURATION ---
# Note: For your dry run, uncomment the small config. 
# For the full A100 GPU run, use the full lists below.

# DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
# FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
# SEEDS = [42, 123, 2026, 777, 888]

# --- DRY RUN CONFIG (Uncomment to test) ---
DATASETS = ["breastmnist"]
FRACTIONS = [0.01]
SEEDS = [42]
# ------------------------------------------

BATCH_SIZE = 32
LR_HEAD = 5e-3       # Matched for a fair classical vs quantum fight
LR_QUANTUM = 5e-3    
RESULTS_FILE = "results/01_frozen_ablation_logs.json"


def evaluate_epoch(model, dataloader, criterion, device):
    """
    Evaluates model performance using strictly Argmax Multi-Class Metrics.
    No threshold shifting is permitted.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.squeeze().long().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
    return avg_loss, float(acc), float(bal_acc), float(macro_f1)


def train_ablation_model(model, train_loader, val_loader, test_loader, device, model_name, dataset_name, seed, frac, num_classes):
    print(f"\n      Training {model_name}...")
    
    # 1. STRICT OVERRIDE: Completely immobilize the feature extractor
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
            
    head_params, quantum_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "q_layer" in name:
            quantum_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.Adam([
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': quantum_params, 'lr': LR_QUANTUM, 'weight_decay': 0.0}
    ])

    # 2. Dynamic Class Weighting for Multi-Class Imbalance
    all_train_labels = []
    for _, y_batch in train_loader:
        # Handles batches safely ensuring 1D list extraction
        all_train_labels.extend(y_batch.squeeze(1).tolist() if y_batch.dim() > 1 else y_batch.tolist())
        
    class_counts = np.bincount(all_train_labels, minlength=num_classes)
    total_samples = len(all_train_labels)
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * (class_counts + 1e-5))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_f1 = 0.0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_f1": []}
    
    # DYNAMIC EPOCH SCALING: Give tiny datasets enough steps to actually learn
    batches_per_epoch = len(train_loader)
    max_epochs = 100
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        
        # Immobilize BatchNorm statistics for the frozen backbone
        for name, module in model.named_modules():
            if "backbone" in name:
                module.eval()
                
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze().long().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Prevent exploding gradients in classical heads
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc, val_bal_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_f1"].append(val_f1)
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0  # Reset patience
            print(f"         Epoch {epoch+1:03d}/{max_epochs} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Bal Acc: {val_bal_acc:.4f} | Macro-F1: {val_f1:.4f} **(Best)**")
        else:
            epochs_no_improve += 1
            
        # Trigger Early Stopping
        if epochs_no_improve >= patience:
            print(f"         -> Early Stopping triggered! No improvement for {patience} epochs.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        safe_name = model_name.replace(' ', '_')
        torch.save(best_weights, f"results/best_ablation_{safe_name}_{dataset_name}_frac{frac}_seed{seed}.pt")
        
    # Final Test Set Evaluation using the model state that maximized Validation Macro-F1
    test_loss, test_acc, test_bal_acc, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
    print(f"         -> Final Test | Acc: {test_acc:.4f} | Bal Acc: {test_bal_acc:.4f} | Macro-F1: {test_f1:.4f}")
    
    return test_acc, test_bal_acc, test_f1, history


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    results = {"experiment": "Frozen-Backbone Ablation (Control Group)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        for frac in FRACTIONS:
            results["datasets"][dataset]["fractions"][str(frac)] = {
                "classical_linear": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "history": []},
                "classical_mlp": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "history": []},
                "classical_deep_ae": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "history": []},
                "quantum_vqc_4q": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "history": []}
            }
            
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | DATA FRACTION: {frac*100}%")
            print(f"=====================================================")
            
            for seed in SEEDS:
                print(f"\n   --- RUNNING SEED: {seed} ---")
                
                train_loader, val_loader, test_loader = get_medmnist_loaders(
                    dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=frac, seed=seed, data_root="/home/jovyan/qml_exp_2026/data_cache"
                )
                
                # BUG FIX: Safely extract unique classes by converting list of (1,) numpy arrays directly
                all_raw_labels = [int(y[0]) for _, y in train_loader.dataset]
                num_classes = len(np.unique(all_raw_labels))
                
                # Model Instantiation
                lin_model = ClassicalLinearResNet(num_classes=num_classes, bottleneck_dim=4).to(device)
                mlp_model = ClassicalMLPResNet(num_classes=num_classes, bottleneck_dim=4).to(device)
                ae_model = ClassicalDeepBottleneckResNet(num_classes=num_classes, bottleneck_dim=4).to(device)
                q_model = QuantumHybridResNet(num_classes=num_classes, n_qubits=4, n_layers=2).to(device)
                
                # Train & Evaluate
                lin_acc, lin_bal, lin_f1, lin_hist = train_ablation_model(lin_model, train_loader, val_loader, test_loader, device, "Classical Linear", dataset, seed, frac, num_classes)
                mlp_acc, mlp_bal, mlp_f1, mlp_hist = train_ablation_model(mlp_model, train_loader, val_loader, test_loader, device, "Classical MLP", dataset, seed, frac, num_classes)
                ae_acc, ae_bal, ae_f1, ae_hist = train_ablation_model(ae_model, train_loader, val_loader, test_loader, device, "Classical Deep AE", dataset, seed, frac, num_classes)
                q_acc, q_bal, q_f1, q_hist = train_ablation_model(q_model, train_loader, val_loader, test_loader, device, "Quantum VQC 4Q", dataset, seed, frac, num_classes)
                
                # Metric Logging
                frac_res = results["datasets"][dataset]["fractions"][str(frac)]
                
                frac_res["classical_linear"]["test_acc"].append(lin_acc)
                frac_res["classical_linear"]["test_bal_acc"].append(lin_bal)
                frac_res["classical_linear"]["test_f1"].append(lin_f1)
                frac_res["classical_linear"]["history"].append(lin_hist)
                
                frac_res["classical_mlp"]["test_acc"].append(mlp_acc)
                frac_res["classical_mlp"]["test_bal_acc"].append(mlp_bal)
                frac_res["classical_mlp"]["test_f1"].append(mlp_f1)
                frac_res["classical_mlp"]["history"].append(mlp_hist)

                frac_res["classical_deep_ae"]["test_acc"].append(ae_acc)
                frac_res["classical_deep_ae"]["test_bal_acc"].append(ae_bal)
                frac_res["classical_deep_ae"]["test_f1"].append(ae_f1)
                frac_res["classical_deep_ae"]["history"].append(ae_hist)
                
                frac_res["quantum_vqc_4q"]["test_acc"].append(q_acc)
                frac_res["quantum_vqc_4q"]["test_bal_acc"].append(q_bal)
                frac_res["quantum_vqc_4q"]["test_f1"].append(q_f1)
                frac_res["quantum_vqc_4q"]["history"].append(q_hist)
            
            print(f"\n   [AVERAGE MACRO-F1 RESULTS ACROSS {len(SEEDS)} SEEDS]")
            lin_avg_f1 = np.mean(frac_res["classical_linear"]["test_f1"])
            mlp_avg_f1 = np.mean(frac_res["classical_mlp"]["test_f1"])
            ae_avg_f1 = np.mean(frac_res["classical_deep_ae"]["test_f1"])
            q_avg_f1 = np.mean(frac_res["quantum_vqc_4q"]["test_f1"])
            
            print(f"   Linear F1: {lin_avg_f1:.4f} | MLP F1: {mlp_avg_f1:.4f} | Deep AE F1: {ae_avg_f1:.4f} | Quantum F1: {q_avg_f1:.4f}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment complete. Multi-seed results safely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()