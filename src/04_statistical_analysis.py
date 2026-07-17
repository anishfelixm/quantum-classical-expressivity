"""
Phase 4: Statistical Significance Analysis (Welch's t-test).
"""
import os
import json
import numpy as np
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PHASE2_LOG = os.path.join(RESULTS_DIR, "end_to_end_logs.json")
PHASE3_LOG = os.path.join(RESULTS_DIR, "03_robustness_e2e_logs.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "statistical_significance_report.txt")

def compute_welchs_t_test(quantum_dist, classical_dist):
    if len(quantum_dist) < 2 or len(classical_dist) < 2:
        return np.nan, np.nan
    # Mathematical formulation for unequal variances: Welch's t-test
    t_stat, p_val = stats.ttest_ind(quantum_dist, classical_dist, equal_var=False)
    return t_stat, p_val

def analyze_phase2_scaling():
    if not os.path.exists(PHASE2_LOG): return "Phase 2 logs not found."
    with open(PHASE2_LOG, "r") as f: data = json.load(f)

    report = "=========================================================\n"
    report += " PHASE 2: SCALING EXPRESSIVITY (WELCH'S T-TEST)\n"
    report += "=========================================================\n\n"

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        report += f"--- DATASET: {dataset_name.upper()} ---\n"
        for frac_key, frac_data in dataset_data.get("fractions", {}).items():
            report += f"\nData Fraction: {float(frac_key)*100}%\n"
            b_dim = "4" 
            if b_dim not in frac_data.get("bottlenecks", {}): continue
                
            metrics = frac_data["bottlenecks"][b_dim]
            try:
                q_f1 = metrics["quantum_vqc"]["test_f1"]
                lin_f1 = metrics["classical_linear"]["test_f1"]
                mlp_f1 = metrics["classical_mlp"]["test_f1"]
                ae_f1 = metrics["classical_deep_ae"]["test_f1"]
                
                _, p_lin = compute_welchs_t_test(q_f1, lin_f1)
                _, p_mlp = compute_welchs_t_test(q_f1, mlp_f1)
                _, p_ae = compute_welchs_t_test(q_f1, ae_f1)
                
                report += f"  Quantum VQC:      {np.mean(q_f1):.4f} ± {np.std(q_f1):.4f}\n"
                report += f"  vs Linear Model:  p-value = {p_lin:.4e} " + ("(SIG)\n" if p_lin < 0.05 else "(NS)\n")
                report += f"  vs Classical MLP: p-value = {p_mlp:.4e} " + ("(SIG)\n" if p_mlp < 0.05 else "(NS)\n")
                report += f"  vs Deep AE:       p-value = {p_ae:.4e} " + ("(SIG)\n" if p_ae < 0.05 else "(NS)\n")
            except KeyError:
                report += "  [Missing Data]\n"
    return report

def analyze_phase3_robustness():
    if not os.path.exists(PHASE3_LOG): return "\nPhase 3 logs not found."
    with open(PHASE3_LOG, "r") as f: data = json.load(f)

    report = "\n=========================================================\n"
    report += " PHASE 3: ZOMBIE STATE ROBUSTNESS (NOISE = 0.10)\n"
    report += "=========================================================\n\n"

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        report += f"--- DATASET: {dataset_name.upper()} ---\n"
        for frac_key, frac_data in dataset_data.get("fractions", {}).items():
            report += f"\nModel Trained on Data Fraction: {float(frac_key)*100}%\n"
            target_noise = "0.10" 
            
            try:
                # We now pull the RAW 5-seed arrays mathematically saved in Script 3
                q_f1 = frac_data["quantum_vqc"][target_noise]["raw_f1"]
                ae_f1 = frac_data["classical_deep_ae"][target_noise]["raw_f1"]
                
                _, p_ae = compute_welchs_t_test(q_f1, ae_f1)
                
                report += f"  Quantum F1 Decay: {np.mean(q_f1):.4f} ± {np.std(q_f1):.4f}\n"
                report += f"  Deep AE F1 Decay: {np.mean(ae_f1):.4f} ± {np.std(ae_f1):.4f}\n"
                report += f"  Statistical Significance (Welch's t-test): p-value = {p_ae:.4e} "
                report += "(SIGNIFICANT SEPARATION)\n" if p_ae < 0.05 else "(No separation)\n"
                    
            except KeyError:
                report += "  [Missing Data for Noise = 0.10]\n"
    return report

def main():
    report_content = analyze_phase2_scaling()
    report_content += analyze_phase3_robustness()
    with open(REPORT_FILE, "w") as f: f.write(report_content)
    print(report_content)

if __name__ == "__main__":
    main()