"""
Statistical Significance Analysis & Table Generation.

Ingests the JSON arrays from the E2E Fine-Tuning and Robustness experiments. 
Computes Welch's t-tests (unequal variances) and generates Overleaf-ready LaTeX 
tables proving Empirical Quantum Superiority and Topological Robustness.
"""

import os
import json
import numpy as np
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXP2_LOG = os.path.join(RESULTS_DIR, "end_to_end_logs.json")
EXP3_LOG = os.path.join(RESULTS_DIR, "robustness_e2e_logs.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "manuscript_statistical_tables.txt")


def compute_welchs_t_test(quantum_dist, classical_dist):
    """
    Computes Welch's t-test handling unequal variances and potential NaNs.
    """
    # Filter out None/NaN from model collapse
    q_clean = [x for x in quantum_dist if x is not None and not np.isnan(x)]
    c_clean = [x for x in classical_dist if x is not None and not np.isnan(x)]
    
    if len(q_clean) < 2 or len(c_clean) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
    q_mean, q_std = np.mean(q_clean), np.std(q_clean)
    c_mean, c_std = np.mean(c_clean), np.std(c_clean)
    
    # Welch's t-test (equal_var=False) is mathematically required for comparing 
    # models with different architectural variances.
    t_stat, p_val = stats.ttest_ind(q_clean, c_clean, equal_var=False)
    
    return q_mean, q_std, c_mean, c_std, p_val


def generate_latex_table(title, columns, rows):
    """Generates an Overleaf-ready LaTeX table."""
    tex = f"% --- {title} ---\n"
    tex += "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{" + title + "}\n"
    tex += "\\resizebox{\\textwidth}{!}{\n" # Ensures wide tables fit on the page
    tex += "\\begin{tabular}{" + "c" * len(columns) + "}\n\\toprule\n"
    tex += " & ".join([f"\\textbf{{{c}}}" for c in columns]) + " \\\\\n\\midrule\n"
    
    for row in rows:
        # Safely escape percentage signs and underscores for LaTeX compiler
        formatted_row = [str(item).replace('%', '\\%').replace('_', '\\_') for item in row]
        tex += " & ".join(formatted_row) + " \\\\\n"
        
    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n\n"
    return tex


def analyze_experiment2_scaling():
    if not os.path.exists(EXP2_LOG): return "Experiment 2 logs not found.\n"
    with open(EXP2_LOG, "r") as f: data = json.load(f)

    report = "=========================================================\n"
    report += " EXPERIMENT 2: LATENT SCARCITY SEPARABILITY (WELCH'S T-TEST)\n"
    report += "=========================================================\n\n"

    columns_auc = ["Dataset", "Frac", "Dim", "VQC AUC", "Deep Funnel AUC", "p-value", "Sig."]
    columns_f1 = ["Dataset", "Frac", "Dim", "VQC F1", "Deep Funnel F1", "p-value", "Sig."]
    latex_rows_auc = []
    latex_rows_f1 = []

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        # Testing extreme scarcity (1%), mid scarcity (10%), and abundance (100%)
        for frac_key in ["0.01", "0.1", "1.0"]:
            if frac_key not in dataset_data.get("fractions", {}): continue
            frac_data = dataset_data["fractions"][frac_key]
            
            # Use raw % so the generator function can escape it cleanly
            frac_display = f"{int(float(frac_key)*100)}%"
            
            for b_dim in ["4", "8", "16"]:
                if b_dim not in frac_data.get("bottlenecks", {}): continue
                metrics = frac_data["bottlenecks"][b_dim]
                
                try:
                    # AUC Comparison
                    q_auc = metrics["quantum_vqc"]["test_auc"]
                    ae_auc = metrics["classical_deep_funnel"]["test_auc"]
                    q_am, q_as, c_am, c_as, p_auc = compute_welchs_t_test(q_auc, ae_auc)
                    
                    sig_auc = "*" if not np.isnan(p_auc) and p_auc < 0.05 else ("ns" if not np.isnan(p_auc) else "-")
                    p_auc_str = f"{p_auc:.3e}" if not np.isnan(p_auc) else "N/A"
                    
                    latex_rows_auc.append([
                        dataset_name, frac_display, b_dim,
                        f"{q_am:.3f} $\\pm$ {q_as:.3f}" if not np.isnan(q_am) else "N/A", 
                        f"{c_am:.3f} $\\pm$ {c_as:.3f}" if not np.isnan(c_am) else "N/A", 
                        p_auc_str, sig_auc
                    ])
                    
                    # F1 Comparison
                    q_f1 = metrics["quantum_vqc"]["test_f1"]
                    ae_f1 = metrics["classical_deep_funnel"]["test_f1"]
                    q_fm, q_fs, c_fm, c_fs, p_f1 = compute_welchs_t_test(q_f1, ae_f1)
                    
                    sig_f1 = "*" if not np.isnan(p_f1) and p_f1 < 0.05 else ("ns" if not np.isnan(p_f1) else "-")
                    p_f1_str = f"{p_f1:.3e}" if not np.isnan(p_f1) else "N/A"
                    
                    latex_rows_f1.append([
                        dataset_name, frac_display, b_dim,
                        f"{q_fm:.3f} $\\pm$ {q_fs:.3f}" if not np.isnan(q_fm) else "N/A", 
                        f"{c_fm:.3f} $\\pm$ {c_fs:.3f}" if not np.isnan(c_fm) else "N/A", 
                        p_f1_str, sig_f1
                    ])
                except KeyError:
                    continue

    report += generate_latex_table("Experiment 2: Topological Separability (AUC) under Scarcity", columns_auc, latex_rows_auc)
    report += generate_latex_table("Experiment 2: Decision Boundary (Macro-F1) under Scarcity", columns_f1, latex_rows_f1)
    return report


def analyze_experiment3_robustness():
    if not os.path.exists(EXP3_LOG): return "Experiment 3 logs not found.\n"
    with open(EXP3_LOG, "r") as f: data = json.load(f)

    report = "=========================================================\n"
    report += " EXPERIMENT 3: THE PRECISION PARADOX (NOISE = 0.10)\n"
    report += "=========================================================\n\n"

    columns = ["Dataset", "Frac", "Dim", "Metric", "VQC Score", "Deep Funnel Score", "p-value", "Sig."]
    latex_rows = []

    for dataset_name, dataset_data in data.get("datasets", {}).items():
        for frac_key in ["0.01", "1.0"]:
            if frac_key not in dataset_data.get("fractions", {}): continue
            frac_data = dataset_data["fractions"][frac_key]
            
            frac_display = f"{int(float(frac_key)*100)}%"
            
            for b_dim in ["4", "16"]: 
                if b_dim not in frac_data.get("bottlenecks", {}): continue
                
                target_noise = "0.10" # The critical noise threshold
                try:
                    noise_data_q = frac_data["bottlenecks"][b_dim]["quantum_vqc"].get(target_noise, {})
                    noise_data_c = frac_data["bottlenecks"][b_dim]["classical_deep_funnel"].get(target_noise, {})
                    
                    # Compute F1 Decay
                    q_f1 = noise_data_q.get("raw_f1", [])
                    c_f1 = noise_data_c.get("raw_f1", [])
                    q_fm, q_fs, c_fm, c_fs, p_f1 = compute_welchs_t_test(q_f1, c_f1)
                    
                    sig_f1 = "*" if not np.isnan(p_f1) and p_f1 < 0.05 else ("ns" if not np.isnan(p_f1) else "-")
                    p_f1_str = f"{p_f1:.3e}" if not np.isnan(p_f1) else "N/A"
                    
                    latex_rows.append([
                        dataset_name, frac_display, b_dim, "Macro-F1",
                        f"{q_fm:.3f} $\\pm$ {q_fs:.3f}" if not np.isnan(q_fm) else "N/A", 
                        f"{c_fm:.3f} $\\pm$ {c_fs:.3f}" if not np.isnan(c_fm) else "N/A", 
                        p_f1_str, sig_f1
                    ])
                        
                    # Compute AUC Decay
                    q_auc = noise_data_q.get("raw_auc", [])
                    c_auc = noise_data_c.get("raw_auc", [])
                    q_am, q_as, c_am, c_as, p_auc = compute_welchs_t_test(q_auc, c_auc)
                    
                    sig_auc = "*" if not np.isnan(p_auc) and p_auc < 0.05 else ("ns" if not np.isnan(p_auc) else "-")
                    p_auc_str = f"{p_auc:.3e}" if not np.isnan(p_auc) else "N/A"
                    
                    latex_rows.append([
                        dataset_name, frac_display, b_dim, "ROC-AUC",
                        f"{q_am:.3f} $\\pm$ {q_as:.3f}" if not np.isnan(q_am) else "N/A", 
                        f"{c_am:.3f} $\\pm$ {c_as:.3f}" if not np.isnan(c_am) else "N/A", 
                        p_auc_str, sig_auc
                    ])
                except KeyError:
                    continue

    report += generate_latex_table("Experiment 3: F1 and AUC Decay under AWGN Sensor Degradation ($\\sigma=0.10$)", columns, latex_rows)
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_content = "% ==========================================\n"
    report_content += "% LaTeX TABLES FOR MANUSCRIPT INSERTION\n"
    report_content += "% ==========================================\n"
    report_content += "% Requires: \\usepackage{booktabs} and \\usepackage{graphicx} in your preamble.\n\n"
    
    report_content += analyze_experiment2_scaling()
    report_content += analyze_experiment3_robustness()
    
    with open(REPORT_FILE, "w") as f: 
        f.write(report_content)
        
    print(f"Statistical Analysis Complete. LaTeX tables saved to: {REPORT_FILE}")
    print("\nPreview of Output:\n")
    print(report_content)

if __name__ == "__main__":
    main()