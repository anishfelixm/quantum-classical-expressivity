import os
import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DATASET = "pneumoniamnist" # Ensure this matches the dataset you ran the training on
FINETUNE_LOG = f"results/end_to_end_logs_{DATASET}.json"
ROBUSTNESS_LOG = f"results/robustness_logs_{DATASET}.json"
OUTPUT_DIR = "paper/figures"

# IEEE Standard Formatting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

COLOR_CLASSICAL = '#1f77b4' # Deep Blue
COLOR_QUANTUM = '#ff7f0e'   # Vibrant Orange

def plot_finetune_dynamics():
    """Generates Figure 1: Training convergence over epochs (100% Data)."""
    if not os.path.exists(FINETUNE_LOG):
        print(f"[Warning] Missing {FINETUNE_LOG}. Skipping Figure 1.")
        return

    with open(FINETUNE_LOG, 'r') as f:
        data = json.load(f)

    # We plot the 100% data fraction to show the ultimate optimization plateau
    if "1.0" not in data["fractions"]:
        print("[Warning] 100% data fraction not found in logs. Skipping Figure 1.")
        return

    c_history = data["fractions"]["1.0"]["classical"]["history"]["val_auc"]
    q_history = data["fractions"]["1.0"]["quantum"]["history"]["val_auc"]
    epochs = range(1, len(c_history) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, c_history, label='Classical Hybrid', color=COLOR_CLASSICAL, linestyle='--')
    plt.plot(epochs, q_history, label='Quantum Hybrid (VQC)', color=COLOR_QUANTUM, linestyle='-')

    plt.title(f'Fine-Tuning Dynamics ({DATASET.capitalize()} 100%)')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation AUC-ROC')
    plt.ylim(0.70, 1.0) # Scaled to highlight the plateau gap
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "finetune_dynamics.png")
    plt.savefig(out_path, dpi=300) # 300 DPI is required for IEEE publications
    print(f"Successfully generated: {out_path}")
    plt.close()

def plot_robustness_glass_cannon():
    """Generates Figure 2: The Precision Paradox under Gaussian Noise (10% Data)."""
    if not os.path.exists(ROBUSTNESS_LOG):
        print(f"[Warning] Missing {ROBUSTNESS_LOG}. Skipping Figure 2.")
        return

    with open(ROBUSTNESS_LOG, 'r') as f:
        data = json.load(f)

    # We plot the 10% data fraction as specified in Section VII of the paper
    if "0.1" not in data["fractions"]:
        print("[Warning] 10% data fraction not found in robustness logs. Skipping Figure 2.")
        return

    c_curve = data["fractions"]["0.1"]["classical_curve"]
    q_curve = data["fractions"]["0.1"]["quantum_curve"]

    # Extract sorted noise levels (sigma)
    sigmas = sorted([float(k) for k in c_curve.keys()])
    
    c_aucs = [c_curve[str(s)]["auc"] for s in sigmas]
    q_aucs = [q_curve[str(s)]["auc"] for s in sigmas]

    plt.figure(figsize=(8, 6))
    
    # Using markers to clearly define the specific noise levels tested
    plt.plot(sigmas, c_aucs, label='Classical Hybrid', color=COLOR_CLASSICAL, marker='s', linestyle='--')
    plt.plot(sigmas, q_aucs, label='Quantum Hybrid (VQC)', color=COLOR_QUANTUM, marker='o', linestyle='-')

    # Highlight the crossover point mentioned in the paper
    plt.axvline(x=0.02, color='gray', linestyle=':', alpha=0.6)
    plt.text(0.022, 0.75, 'Crossover Point\n($\sigma \approx 0.02$)', color='gray', fontsize=10)

    plt.title(f'Robustness Decay (The Glass Cannon Effect)')
    plt.xlabel('Gaussian Noise Std. Dev. ($\sigma$)')
    plt.ylabel('Test Set AUC-ROC')
    plt.ylim(0.45, 1.0) # 0.5 is random guessing
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "robustness_glass_cannon.png")
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated: {out_path}")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Generating IEEE-Formatted Figures ---")
    plot_finetune_dynamics()
    plot_robustness_glass_cannon()
    print("Plotting complete. Images are ready for LaTeX compilation.")

if __name__ == "__main__":
    main()