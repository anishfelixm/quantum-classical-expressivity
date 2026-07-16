"""
Phase 4: Statistical Significance Analysis (Welch's t-test).

This script parses the 5-seed distributions from the Phase 2 and Phase 3 JSON logs.
It computes Welch's t-tests (assuming unequal variances) between the Quantum VQC
and the Classical baselines to mathematically prove Quantum Advantage.
"""

import os
import json
import numpy as np
from scipy import stats

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PHASE2_LOG = os.path.join(RESULTS_DIR, "end_to_end_logs.json")
PHASE3_LOG = os.path.join(RESULTS_DIR, "03_robustness_e2e_logs.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "statistical_significance_report.txt")

def compute_welchs_t_test(quantum_dist, classical_dist):
    """
    Computes Welch's t-test for two arrays of seed results.
    Returns the t-statistic and the p-value.
    """
    if len(quantum_dist) < 2 or len(classical_dist) < 2:
        return np.nan, np.nan
    
    # equal_var=False enforces Welch's t-test rather than Student's t-test
    t_stat, p_val = stats.ttest_ind(quantum_dist, classical_dist, equal_var=False)
    return t_stat, p_val


def analyze_phase2_scaling():
    """Analyzes the Phase 2 JSON to prove Quantum Advantage across data fractions."""
    if not os.path.exists(PHASE2_LOG):
        return "Phase 2 logs not found."

    with open(PHASE2_LOG, "r") as f:
        data = json.load(f)

    report = "=========================================================\n"
    report += " PHASE 2: SCALING EXPRESSIVITY (WELCH'S T-TEST)\n"
    report += "=========================================================\n\n"

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        report += f"--- DATASET: {dataset_name.upper()} ---\n"
        
        for frac_key, frac_data in dataset_data.get("fractions", {}).items():
            report += f"\nData Fraction: {float(frac_key)*100}%\n"
            
            # Using Bottleneck 4 as our standardized evaluation benchmark
            b_dim = "4" 
            if b_dim not in frac_data.get("bottlenecks", {}):
                continue
                
            metrics = frac_data["bottlenecks"][b_dim]
            
            try:
                q_f1_dist = metrics["quantum_vqc"]["test_f1"]
                lin_f1_dist = metrics["classical_linear"]["test_f1"]
                mlp_f1_dist = metrics["classical_mlp"]["test_f1"]
                
                # Compare Quantum vs Linear (The low-data survivor)
                t_lin, p_lin = compute_welchs_t_test(q_f1_dist, lin_f1_dist)
                
                # Compare Quantum vs MLP (The deep classical baseline)
                t_mlp, p_mlp = compute_welchs_t_test(q_f1_dist, mlp_f1_dist)
                
                q_mean, q_std = np.mean(q_f1_dist), np.std(q_f1_dist)
                
                report += f"  Quantum VQC:      {q_mean:.4f} ± {q_std:.4f}\n"
                report += f"  vs Linear Model:  p-value = {p_lin:.4e} "
                report += "(SIGNIFICANT)\n" if p_lin < 0.05 else "(Not Significant)\n"
                
                report += f"  vs Classical MLP: p-value = {p_mlp:.4e} "
                report += "(SIGNIFICANT)\n" if p_mlp < 0.05 else "(Not Significant)\n"
                
            except KeyError:
                report += "  [Missing Data for seeds]\n"

    return report


def analyze_phase3_robustness():
    """Analyzes the Phase 3 JSON to prove Quantum Robustness in the Zombie State."""
    if not os.path.exists(PHASE3_LOG):
        return "\nPhase 3 logs not found."

    # NOTE: Phase 3 JSON (as currently structured in Script 3) stores averaged metrics.
    # To run a true t-test, Script 3's output dict must retain the raw arrays.
    # For the sake of this script, we will calculate bounds based on the mean/std, 
    # but actual publication requires comparing the raw lists of 5 seeds.
    
    with open(PHASE3_LOG, "r") as f:
        data = json.load(f)

    report = "\n=========================================================\n"
    report += " PHASE 3: ZOMBIE STATE ROBUSTNESS (NOISE = 0.10)\n"
    report += "=========================================================\n\n"

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        report += f"--- DATASET: {dataset_name.upper()} ---\n"
        
        for frac_key, frac_data in dataset_data.get("fractions", {}).items():
            report += f"\nBase Model Trained on Data Fraction: {float(frac_key)*100}%\n"
            
            # Target the "Zombie State" noise threshold we care about
            target_noise = "0.10" 
            
            try:
                q_mean = frac_data["quantum_avg"][target_noise]["mean_f1"]
                q_std = frac_data["quantum_avg"][target_noise]["std_f1"]
                
                ae_mean = frac_data["classical_deep_ae_avg"][target_noise]["mean_f1"]
                ae_std = frac_data["classical_deep_ae_avg"][target_noise]["std_f1"]
                
                report += f"  Quantum F1 Decay:      {q_mean:.4f} ± {q_std:.4f}\n"
                report += f"  Deep AE F1 Decay:      {ae_mean:.4f} ± {ae_std:.4f}\n"
                
                # Check for statistical separation (> 2 standard deviations)
                if (q_mean - (2 * q_std)) > (ae_mean + (2 * ae_std)):
                    report += "  -> Conclusion: Statistically separated advantage for Quantum mapping.\n"
                else:
                    report += "  -> Conclusion: Within margin of error.\n"
                    
            except KeyError:
                report += "  [Missing Data for Noise = 0.10]\n"

    return report


def main():
    print("Running Statistical Significance Analysis (Welch's t-test)...")
    
    report_content = analyze_phase2_scaling()
    report_content += analyze_phase3_robustness()
    
    with open(REPORT_FILE, "w") as f:
        f.write(report_content)
        
    print(f"\nAnalysis complete. Full statistical report saved to:\n{REPORT_FILE}")
    print(report_content)

if __name__ == "__main__":
    main()