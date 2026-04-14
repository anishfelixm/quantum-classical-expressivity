"""
Manuscript Figure Generator for IEEE QCE Submission.
Generates the Bottleneck Gap, Expressivity Dynamics, and Precision Paradox plots
using the fully verified multi-seed JSON logs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ABLATION_LOG = "results/frozen_ablation_logs.json"
FINETUNE_LOG = "results/end_to_end_logs.json"
ROBUSTNESS_LOG = "results/robustness_e2e_logs.json"
OUTPUT_DIR = "paper/figures"

# IEEE Standard Formatting
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

COLORS = {
    "linear": "#1f77b4",  # Deep Blue
    "mlp": "#2ca02c",     # Forest Green
    "quantum": "#ff7f0e"  # Vibrant Orange
}

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[Error] Missing log file: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_history_stats(histories, metric):
    arrays = [h[metric] for h in histories]
    stacked = np.vstack(arrays)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)

def plot_bottleneck_gap():
    """Generates a Bar Chart comparing Frozen vs End-to-End AUC to prove latent reshaping."""
    ablation_data = load_json(ABLATION_LOG)
    finetune_data = load_json(FINETUNE_LOG)
    if not ablation_data or not finetune_data: return

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    targets = [("breastmnist", "0.1", "BreastMNIST (10% Data)"), 
               ("pneumoniamnist", "0.01", "PneumoniaMNIST (1% Data)")]
    
    models = ["classical_linear", "classical_mlp", "quantum"]
    labels = ["Linear", "MLP", "Quantum VQC"]
    colors = [COLORS["linear"], COLORS["mlp"], COLORS["quantum"]]
    
    x = np.arange(len(targets))
    width = 0.25

    for ax_idx, (data_source, title) in enumerate([(ablation_data, "Frozen Backbone (Ablation)"), 
                                                   (finetune_data, "Layer 3 Unfrozen (End-to-End)")]):
        
        for i, model in enumerate(models):
            means = []
            for ds, frac, _ in targets:
                try:
                    auc_list = data_source["datasets"][ds]["fractions"][frac][model]["test_auc"]
                    means.append(np.mean(auc_list))
                except KeyError:
                    means.append(0)
            
            axs[ax_idx].bar(x + i*width, means, width, label=labels[i], color=colors[i], edgecolor='black')

        axs[ax_idx].set_title(title)
        axs[ax_idx].set_ylabel('Mean Test AUC')
        axs[ax_idx].set_xticks(x + width)
        axs[ax_idx].set_xticklabels([t[2] for t in targets])
        axs[ax_idx].set_ylim(0.5, 1.0)
        axs[ax_idx].grid(axis='y', linestyle=':', alpha=0.7)

    axs[0].legend(loc='upper left')
    plt.suptitle("The Bottleneck Gap: Quantum Advantage via Latent Reshaping", y=1.05, fontsize=16)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "fig01_bottleneck_gap.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

def plot_expressivity_dynamics():
    """Generates the Training Curve grids to show Classical topological collapse."""
    data = load_json(FINETUNE_LOG)
    if not data: return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    targets = [("breastmnist", "0.1"), ("pneumoniamnist", "0.01")]

    for row, (dataset, frac) in enumerate(targets):
        try:
            frac_data = data["datasets"][dataset]["fractions"][frac]
        except KeyError: continue

        epochs = np.arange(1, len(frac_data["quantum"]["history"][0]["train_loss"]) + 1)

        for model_key, color, label, style in [
            ("classical_linear", COLORS["linear"], "Classical Linear", "--"),
            ("classical_mlp", COLORS["mlp"], "Classical MLP", "-."),
            ("quantum", COLORS["quantum"], "Quantum VQC", "-")
        ]:
            histories = frac_data[model_key]["history"]
            
            mean_loss, std_loss = extract_history_stats(histories, "train_loss")
            axs[row, 0].plot(epochs, mean_loss, label=label, color=color, linestyle=style)
            axs[row, 0].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)
            
            mean_auc, std_auc = extract_history_stats(histories, "val_auc")
            axs[row, 1].plot(epochs, mean_auc, label=label, color=color, linestyle=style)
            axs[row, 1].fill_between(epochs, mean_auc - std_auc, mean_auc + std_auc, color=color, alpha=0.15)

        ds_title = "BreastMNIST (10% Data)" if dataset == "breastmnist" else "PneumoniaMNIST (1% Data)"
        axs[row, 0].set_title(f'{ds_title} - Training Loss')
        axs[row, 0].set_ylabel('BCE Loss')
        axs[row, 0].grid(True, linestyle=':', alpha=0.7)

        axs[row, 1].set_title(f'{ds_title} - Validation AUC')
        axs[row, 1].set_ylabel('AUC-ROC')
        axs[row, 1].grid(True, linestyle=':', alpha=0.7)

        if row == 1:
            axs[row, 0].set_xlabel('Epochs')
            axs[row, 1].set_xlabel('Epochs')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle("Expressivity Dynamics Under Severe Information Constraints", y=1.03, fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig02_expressivity_dynamics.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

def plot_robustness_grid():
    """Generates the 2x2 grid showing the Precision Paradox and Data Abundance stabilization."""
    data = load_json(ROBUSTNESS_LOG)
    if not data: return

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    targets = [
        ("breastmnist", "0.1", 0, 0, "BreastMNIST (10% Data Regime)"),
        ("breastmnist", "1.0", 0, 1, "BreastMNIST (100% Data Regime)"),
        ("pneumoniamnist", "0.01", 1, 0, "PneumoniaMNIST (1% Data Regime)"),
        ("pneumoniamnist", "1.0", 1, 1, "PneumoniaMNIST (100% Data Regime)")
    ]

    for dataset, frac, row, col, title in targets:
        try:
            frac_data = data["datasets"][dataset]["fractions"][frac]
        except KeyError: continue

        sigmas = sorted([float(k) for k in frac_data["quantum_avg"].keys()])

        for model_key, color, label, marker, style in [
            ("classical_linear_avg", COLORS["linear"], "Classical Linear", "s", "--"),
            ("classical_mlp_avg", COLORS["mlp"], "Classical MLP", "^", "-."),
            ("quantum_avg", COLORS["quantum"], "Quantum VQC", "o", "-")
        ]:
            model_data = frac_data[model_key]
            mean_f1 = [model_data[f"{s:.2f}"]["mean_f1"] for s in sigmas]
            std_f1 = [model_data[f"{s:.2f}"]["std_f1"] for s in sigmas]
            
            axs[row, col].plot(sigmas, mean_f1, label=label, color=color, marker=marker, linestyle=style)
            axs[row, col].fill_between(sigmas, np.array(mean_f1) - np.array(std_f1), np.array(mean_f1) + np.array(std_f1), color=color, alpha=0.15)

        axs[row, col].set_title(title)
        axs[row, col].set_ylabel('Test F1 Score')
        axs[row, col].axvline(x=0.03, color='gray', linestyle=':', alpha=0.6, label='Phase Decoherence Threshold' if row==0 and col==0 else "")
        axs[row, col].grid(True, linestyle=':', alpha=0.7)
        axs[row, col].set_ylim(0.0, 1.0) 

        if row == 1:
            axs[row, col].set_xlabel('Gaussian Noise Std. Dev. ($\sigma$)')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle("The Precision Paradox: Phase Decoherence vs. Data Abundance", y=1.03, fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig03_precision_paradox.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Generating IEEE-Formatted PDF Figures ---")
    plot_bottleneck_gap()
    plot_expressivity_dynamics()
    plot_robustness_grid()
    print("Plotting complete. PDFs are ready for LaTeX compilation.")

if __name__ == "__main__":
    main()