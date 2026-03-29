import os
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ABLATION_LOG = "results/frozen_ablation_logs.json"
FINETUNE_LOG = "results/end_to_end_logs.json"
ROBUSTNESS_LOG = "results/robustness_logs.json"
OUTPUT_DIR = "paper/figures"

# IEEE Standard Formatting parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6
})

COLOR_CLASSICAL = '#1f77b4' # Deep Blue
COLOR_QUANTUM = '#ff7f0e'   # Vibrant Orange

def plot_2x2_dynamics(log_file, output_filename, title_prefix, fraction_key=None):
    """Generates a 2x2 grid: Top row = BreastMNIST (AUC/Acc), Bottom row = PneumoniaMNIST (AUC/Acc)"""
    if not os.path.exists(log_file):
        print(f"[Warning] Missing {log_file}. Skipping {output_filename}.")
        return

    with open(log_file, 'r') as f:
        data = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    datasets = ["breastmnist", "pneumoniamnist"]
    
    for row, dataset in enumerate(datasets):
        if dataset not in data["datasets"]:
            continue
            
        # Extract data depending on whether it's the ablation log or finetune log
        if fraction_key:
            if fraction_key not in data["datasets"][dataset]["fractions"]:
                continue
            c_hist = data["datasets"][dataset]["fractions"][fraction_key]["classical"]["history"]
            q_hist = data["datasets"][dataset]["fractions"][fraction_key]["quantum"]["history"]
        else:
            c_hist = data["datasets"][dataset]["classical"]["history"]
            q_hist = data["datasets"][dataset]["quantum"]["history"]

        epochs = range(1, len(c_hist["val_auc"]) + 1)

        # Plot AUC (Left Column)
        axs[row, 0].plot(epochs, c_hist["val_auc"], label='Classical', color=COLOR_CLASSICAL, linestyle='--')
        axs[row, 0].plot(epochs, q_hist["val_auc"], label='Quantum (VQC)', color=COLOR_QUANTUM, linestyle='-')
        axs[row, 0].set_title(f'{dataset.capitalize()} - AUC-ROC')
        axs[row, 0].set_ylabel('Validation AUC')
        axs[row, 0].grid(True, linestyle=':', alpha=0.7)

        # Plot Accuracy (Right Column)
        axs[row, 1].plot(epochs, c_hist["val_acc"], label='Classical', color=COLOR_CLASSICAL, linestyle='--')
        axs[row, 1].plot(epochs, q_hist["val_acc"], label='Quantum (VQC)', color=COLOR_QUANTUM, linestyle='-')
        axs[row, 1].set_title(f'{dataset.capitalize()} - Accuracy')
        axs[row, 1].set_ylabel('Validation Accuracy')
        axs[row, 1].grid(True, linestyle=':', alpha=0.7)

        # Bottom row gets X-axis labels
        if row == 1:
            axs[row, 0].set_xlabel('Training Epochs')
            axs[row, 1].set_xlabel('Training Epochs')

    # Single unified legend at the top
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle(title_prefix, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300, bbox_inches='tight') # 300 DPI is required for IEEE publications
    plt.close()
    print(f"Generated: {output_filename}")

def plot_2x2_robustness():
    """Generates a 2x2 grid showing the glass cannon effect for 10% and 100% data."""
    if not os.path.exists(ROBUSTNESS_LOG):
        print(f"[Warning] Missing {ROBUSTNESS_LOG}. Skipping robustness plots.")
        return

    with open(ROBUSTNESS_LOG, 'r') as f:
        data = json.load(f)

    datasets = ["breastmnist", "pneumoniamnist"]
    fractions = ["0.1", "1.0"]
    titles = ["10% Data Regime", "100% Data Regime"]

    for dataset in datasets:
        if dataset not in data["datasets"]:
            continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        for row, frac in enumerate(fractions):
            if frac not in data["datasets"][dataset]["fractions"]:
                continue
            c_curve = data["datasets"][dataset]["fractions"][frac]["classical_curve"]
            q_curve = data["datasets"][dataset]["fractions"][frac]["quantum_curve"]
            
            # Extract sorted noise levels (sigma)
            sigmas = sorted([float(k) for k in c_curve.keys()])
            
            c_aucs = [c_curve[str(s)]["auc"] for s in sigmas]
            q_aucs = [q_curve[str(s)]["auc"] for s in sigmas]
            c_accs = [c_curve[str(s)]["acc"] for s in sigmas]
            q_accs = [q_curve[str(s)]["acc"] for s in sigmas]

            # Plot AUC
            axs[row, 0].plot(sigmas, c_aucs, label='Classical', color=COLOR_CLASSICAL, marker='s', linestyle='--')
            axs[row, 0].plot(sigmas, q_aucs, label='Quantum', color=COLOR_QUANTUM, marker='o', linestyle='-')
            axs[row, 0].set_title(f'{titles[row]} - AUC-ROC')
            axs[row, 0].set_ylabel('Test AUC')
            axs[row, 0].axvline(x=0.02, color='gray', linestyle=':', alpha=0.6)
            axs[row, 0].grid(True, linestyle=':', alpha=0.7)

            # Plot Accuracy
            axs[row, 1].plot(sigmas, c_accs, label='Classical', color=COLOR_CLASSICAL, marker='s', linestyle='--')
            axs[row, 1].plot(sigmas, q_accs, label='Quantum', color=COLOR_QUANTUM, marker='o', linestyle='-')
            axs[row, 1].set_title(f'{titles[row]} - Accuracy')
            axs[row, 1].set_ylabel('Test Accuracy')
            axs[row, 1].axvline(x=0.02, color='gray', linestyle=':', alpha=0.6)
            axs[row, 1].grid(True, linestyle=':', alpha=0.7)

            if row == 1:
                axs[row, 0].set_xlabel('Gaussian Noise Std. Dev. ($\sigma$)')
                axs[row, 1].set_xlabel('Gaussian Noise Std. Dev. ($\sigma$)')

        # Prevent legend duplication
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
        
        plt.suptitle(f"Robustness Decay Profiles ({dataset.capitalize()})", y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save a unique grid for each dataset
        out_name = f"robustness_grid_{dataset}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {out_name}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Generating IEEE-Formatted Figures ---")
    plot_2x2_dynamics(ABLATION_LOG, "ablation_dynamics_grid.png", "Frozen-Backbone Ablation Dynamics")
    plot_2x2_dynamics(FINETUNE_LOG, "finetune_dynamics_grid.png", "End-to-End Fine-Tuning Dynamics (100% Data)", fraction_key="1.0")
    plot_2x2_robustness()
    print("Plotting complete. Images are ready for LaTeX compilation.")

if __name__ == "__main__":
    main()