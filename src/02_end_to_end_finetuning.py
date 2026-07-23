"""
End-to-End Hybrid Quantum-Classical Fine-Tuning.

This script unfreezes the final convolutional block (Layer 3) of the classical feature 
extractor. It allows gradients from the bottlenecks (Linear, MLP, Deep Funnel, VQC) to 
backpropagate, actively reshaping the classical feature maps into an optimal geometry 
under severe information constraints and varying bottleneck dimensions.
"""

import os
import json
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import medmnist
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints_exp2")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

sys.path.append(os.path.join(BASE_DIR, 'src'))
from data.medmnist_loader import get_medmnist_loaders
from models.classical_resnet import ClassicalLinearResNet, ClassicalMLPResNet, ClassicalDeepBottleneckResNet
from models.quantum_vqc import QuantumHybridResNet

# --- EXPERIMENT CONFIGURATION ---
DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
FRACTIONS = [0.01, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
SEEDS = [42, 123, 2026, 777, 888]
BOTTLENECKS = [4, 8, 16]

BATCH_SIZE = 32
LR_BACKBONE = 1e-4       # Conservative LR for Layer 3 to mitigate catastrophic forgetting
LR_HEAD = 1e-3           # Standard LR for classical latent projection heads
LR_QUANTUM = 1e-3        # VQC learning rate aligned with ablation study baseline
RESULTS_FILE_NAME = "end_to_end_logs.json"


def calculate_metrics(labels, preds, probs, num_classes):
    """Helper to calculate metrics safely (Ported from Exp 1)."""
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
        
    return float(acc), float(bal_acc), float(macro_f1), float(auc)


def evaluate_epoch(model, dataloader, criterion, device, num_classes):
    """Evaluates model performance including Train vs Val Tracking for Overfitting Proofs."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.view(-1).long().to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    acc, bal_acc, macro_f1, auc = calculate_metrics(all_labels, all_preds, all_probs, num_classes)
        
    return avg_loss, acc, bal_acc, macro_f1, auc


def train_finetune_model(model, train_loader, val_loader, test_loader, device, model_name, dataset_name, seed, frac, num_classes, b_dim):
    print(f"\n      Training {model_name} (End-to-End) [d={b_dim}]...")

    # 1. Targeted Unfreezing of Layer 3
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = "backbone.6" in name
            
    backbone_params, head_params, quantum_params = [], [], []
    
    # Parameter routing for differential optimization
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "q_layer" in name:
            quantum_params.append(param)
        else:
            head_params.append(param)
            
    # 2. FIXED: Dynamic Optimizer Parameter Groups
    # Prevents PyTorch from crashing when classical models provide an empty quantum_params list.
    param_groups = [{'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': 1e-4}]
    if head_params:
        param_groups.append({'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-4})
    if quantum_params:
        param_groups.append({'params': quantum_params, 'lr': LR_QUANTUM, 'weight_decay': 0.0})
        
    optimizer = optim.Adam(param_groups)

    # 3. Dynamic Class Weighting for Multi-Class Imbalance
    all_train_labels = []
    for _, y_batch in train_loader:
        all_train_labels.extend(y_batch.view(-1).tolist())
        
    class_counts = np.bincount(all_train_labels, minlength=num_classes)
    total_samples = len(all_train_labels)
    class_weights = total_samples / (num_classes * (class_counts + 1e-5))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    
    best_val_f1 = -1.0
    best_weights = None
    history = {
        "train_loss": [], "train_f1": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_f1": [], "val_auc": [], 
        "epoch_times": []
    }
    
    max_epochs = 100
    patience = 30
    epochs_no_improve = 0
    min_epochs = max(20, 200 // len(train_loader))
    
    for epoch in range(max_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # BatchNorm Protocol: Freeze statistics for immobilized layers; allow Layer 3 to update.
        for name, module in model.named_modules():
            if "backbone" in name and "backbone.6" not in name:
                module.eval()
                
        total_loss = 0.0
        train_preds, train_probs, train_labels = [], [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1).long().to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            active_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(active_params, max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                train_probs.extend(probs.cpu().numpy())
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y.cpu().numpy())
            
        train_loss = total_loss / len(train_loader.dataset)
        _, _, train_f1, train_auc = calculate_metrics(train_labels, train_preds, train_probs, num_classes)
        
        val_loss, val_acc, val_bal_acc, val_f1, val_auc = evaluate_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["epoch_times"].append(time.time() - epoch_start_time)
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0  
            print(f"         Epoch {epoch+1:03d}/{max_epochs} | Val Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Macro-F1: {val_f1:.4f} **(Best)**")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience and epoch >= min_epochs:
            print(f"         -> Early Stopping triggered! No improvement for {patience} epochs.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        safe_name = model_name.replace(' ', '_')
        save_path = os.path.join(CHECKPOINT_DIR, f"best_e2e_{safe_name}_{dataset_name}_frac{frac}_b{b_dim}_seed{seed}.pt")
        torch.save(best_weights, save_path)
        
    test_loss, test_acc, test_bal_acc, test_f1, test_auc = evaluate_epoch(model, test_loader, criterion, device, num_classes)
    avg_epoch_time = np.mean(history["epoch_times"])
    
    print(f"         -> Final Test | AUC: {test_auc:.4f} | Bal Acc: {test_bal_acc:.4f} | Macro-F1: {test_f1:.4f} | Avg Epoch Time: {avg_epoch_time:.2f}s")
    
    return test_acc, test_bal_acc, test_f1, test_auc, avg_epoch_time, history


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, RESULTS_FILE_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware utilized: {device}")
    
    results = {"experiment": "End-to-End Fine-Tuning (Layer 3 Unfrozen)", "datasets": {}}
    
    for dataset in DATASETS:
        results["datasets"][dataset] = {"fractions": {}}
        
        info = medmnist.INFO[dataset]
        global_num_classes = len(info['label'])
        
        for frac in FRACTIONS:
            results["datasets"][dataset]["fractions"][str(frac)] = {"bottlenecks": {}}
            
            print(f"\n=====================================================")
            print(f"   {dataset.upper()} | DATA FRACTION: {frac*100}% | CLASSES: {global_num_classes}")
            print(f"=====================================================")
            
            for b in BOTTLENECKS:
                results["datasets"][dataset]["fractions"][str(frac)]["bottlenecks"][str(b)] = {
                    "classical_linear": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "test_auc": [], "avg_epoch_time": [], "history": []},
                    "classical_mlp": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "test_auc": [], "avg_epoch_time": [], "history": []},
                    "classical_deep_funnel": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "test_auc": [], "avg_epoch_time": [], "history": []},
                    "quantum_vqc": {"test_acc": [], "test_bal_acc": [], "test_f1": [], "test_auc": [], "avg_epoch_time": [], "history": []}
                }
            
            for seed in SEEDS:
                print(f"\n   --- RUNNING SEED: {seed} ---")
                
                train_loader, val_loader, test_loader = get_medmnist_loaders(
                    dataset_name=dataset, batch_size=BATCH_SIZE, train_frac=frac, seed=seed, data_root=CACHE_DIR
                )
                
                for b in BOTTLENECKS:
                    print(f"\n   >>> Evaluating Bottleneck Dimension / Qubits: {b} <<<")
                    
                    lin_model = ClassicalLinearResNet(num_classes=global_num_classes, bottleneck_dim=b).to(device)
                    mlp_model = ClassicalMLPResNet(num_classes=global_num_classes, bottleneck_dim=b).to(device)
                    deep_model = ClassicalDeepBottleneckResNet(num_classes=global_num_classes, bottleneck_dim=b).to(device)
                    q_model = QuantumHybridResNet(num_classes=global_num_classes, n_qubits=b, n_layers=2).to(device)
                    
                    l_acc, l_bal, l_f1, l_auc, l_time, l_hist = train_finetune_model(lin_model, train_loader, val_loader, test_loader, device, "Classical Linear", dataset, seed, frac, global_num_classes, b)
                    m_acc, m_bal, m_f1, m_auc, m_time, m_hist = train_finetune_model(mlp_model, train_loader, val_loader, test_loader, device, "Classical MLP", dataset, seed, frac, global_num_classes, b)
                    d_acc, d_bal, d_f1, d_auc, d_time, d_hist = train_finetune_model(deep_model, train_loader, val_loader, test_loader, device, "Classical Deep Funnel", dataset, seed, frac, global_num_classes, b)
                    q_acc, q_bal, q_f1, q_auc, q_time, q_hist = train_finetune_model(q_model, train_loader, val_loader, test_loader, device, "Quantum VQC", dataset, seed, frac, global_num_classes, b)
                    
                    b_res = results["datasets"][dataset]["fractions"][str(frac)]["bottlenecks"][str(b)]
                    
                    models_data = [
                        ("classical_linear", l_acc, l_bal, l_f1, l_auc, l_time, l_hist),
                        ("classical_mlp", m_acc, m_bal, m_f1, m_auc, m_time, m_hist),
                        ("classical_deep_funnel", d_acc, d_bal, d_f1, d_auc, d_time, d_hist),
                        ("quantum_vqc", q_acc, q_bal, q_f1, q_auc, q_time, q_hist)
                    ]
                    
                    for m_name, acc, bal, f1, auc, t_time, hist in models_data:
                        b_res[m_name]["test_acc"].append(acc)
                        b_res[m_name]["test_bal_acc"].append(bal)
                        b_res[m_name]["test_f1"].append(f1)
                        b_res[m_name]["test_auc"].append(auc)
                        b_res[m_name]["avg_epoch_time"].append(t_time)
                        b_res[m_name]["history"].append(hist)
                        
                    del lin_model, mlp_model, deep_model, q_model
                    torch.cuda.empty_cache()
            
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)
                
            for b in BOTTLENECKS:
                b_res = results["datasets"][dataset]["fractions"][str(frac)]["bottlenecks"][str(b)]
                print(f"\n   [AVERAGE AUC RESULTS ACROSS {len(SEEDS)} SEEDS | DIMENSION: {b}]")
                print(f"   Linear: {np.nanmean(b_res['classical_linear']['test_auc']):.4f} | "
                      f"MLP: {np.nanmean(b_res['classical_mlp']['test_auc']):.4f} | "
                      f"Deep Funnel: {np.nanmean(b_res['classical_deep_funnel']['test_auc']):.4f} | "
                      f"Quantum VQC: {np.nanmean(b_res['quantum_vqc']['test_auc']):.4f}")
    
    print(f"\nExperiment complete. Multi-seed results safely logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()