# Scripts/05_optimize.py
# Implements a simplified and powerful UCB optimization where all geometric
# and operational parameters are optimized simultaneously.

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from joblib import load
from pathlib import Path
import path_utils # Import the utility

def main():
    C = path_utils.load_cfg() # Use the utility to load config
    P = C["paths"]
    OPT = C["optimization"]

    # --- 1. Load Data and GPR Models ---
    print("--- Loading surrogate models and data ---")
    models_dir = Path(P["outputs_models"])
    master_parquet_path = Path(P["master_parquet"])
    
    try:
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        gpr_cruise = load(models_dir / "gpr_cruise_model.joblib")
        df_full = pd.read_parquet(master_parquet_path)
    except Exception as e:
        raise SystemExit(f"Error loading files. Have you run scripts 01-04? Details: {e}")

    # --- 2. Define Full Optimization Space and Bounds ---
    df_hover = df_full[df_full['flight_mode'] == 'hover']
    df_cruise = df_full[df_full['flight_mode'] == 'cruise']

    # ALL parameters are now optimization variables
    opt_vars = [
        'AR', 'lambda', 'twist (deg)', 
        'aoa_root_hover (deg)', 'op_speed_hover (rpm)',
        'aoa_root_cruise (deg)', 'op_speed_cruise (m/s)'
    ]
    
    bounds = [
        (df_full['AR'].min(), df_full['AR'].max()),
        (df_full['lambda'].min(), df_full['lambda'].max()),
        (df_full['twist (deg)'].min(), df_full['twist (deg)'].max()),
        (df_hover['aoa_root (deg)'].min(), df_hover['aoa_root (deg)'].max()),
        (df_hover['op_speed'].min(), df_hover['op_speed'].max()),
        (df_cruise['aoa_root (deg)'].min(), df_cruise['aoa_root (deg)'].max()),
        (df_cruise['op_speed'].min(), df_cruise['op_speed'].max())
    ]
    
    # --- 3. Establish Baselines from Raw Data ---
    baseline_hover_eta = df_hover['performance'].median()
    baseline_cruise_ld = df_cruise['performance'].median()
    print(f"Baseline Hover Score (Median η): {baseline_hover_eta:.4f}")
    print(f"Baseline Cruise Score (Median L/D): {baseline_cruise_ld:.4f}")

    # --- 4. Define the Simplified UCB Objective Function ---
    def objective_function(X, w_hover):
        kappa = OPT.get("kappa", 1.0)
        
        # Unpack the design vector X
        AR, lam, twist, aoa_h, rpm_h, aoa_c, v_c = X
        
        # Create input DataFrames for the models
        # Note: aoa_root (deg) is a single column in the training data
        df_hover_point = pd.DataFrame([[AR, lam, aoa_h, twist, rpm_h]], columns=gpr_hover.feature_names_in_)
        df_cruise_point = pd.DataFrame([[AR, lam, aoa_c, twist, v_c]], columns=gpr_cruise.feature_names_in_)

        # Get predictions and uncertainties
        mean_h, std_h = gpr_hover.predict(df_hover_point, return_std=True)
        mean_c, std_c = gpr_cruise.predict(df_cruise_point, return_std=True)

        # Exploitation Term (Normalized Performance)
        I_hover = (mean_h[0] - baseline_hover_eta) / baseline_hover_eta
        I_cruise = (mean_c[0] - baseline_cruise_ld) / baseline_cruise_ld
        J_exploit = w_hover * I_hover + (1.0 - w_hover) * I_cruise

        # Exploration Term (Normalized Uncertainty)
        U_hover = std_h[0] / baseline_hover_eta
        U_cruise = std_c[0] / baseline_cruise_ld
        J_explore = w_hover * U_hover + (1.0 - w_hover) * U_cruise
        
        J_final = J_exploit + kappa * J_explore
        return -J_final

    # --- 5. Run Optimization ---
    print("\n--- Starting Full UCB Optimization ---")
    all_results = []
    
    for w in OPT["w_hover"]:
        print(f"Optimizing for w_hover = {w}...")
        result = differential_evolution(func=objective_function, bounds=bounds, args=(w,), maxiter=400, popsize=70, seed=42)
        
        X_opt = result.x
        AR, lam, twist, aoa_h, rpm_h, aoa_c, v_c = X_opt
        
        df_h = pd.DataFrame([[AR, lam, aoa_h, twist, rpm_h]], columns=gpr_hover.feature_names_in_)
        df_c = pd.DataFrame([[AR, lam, aoa_c, twist, v_c]], columns=gpr_cruise.feature_names_in_)
        final_eta = gpr_hover.predict(df_h)[0]
        final_ld = gpr_cruise.predict(df_c)[0]

        result_data = {
            'w_hover': w, 'peak_hover_eta': final_eta, 'peak_cruise_ld': final_ld,
            'AR': AR, 'lambda': lam, 'twist (deg)': twist, 
            'aoa_root_hover (deg)': aoa_h, 'op_speed_hover (rpm)': rpm_h,
            'aoa_root_cruise (deg)': aoa_c, 'op_speed_cruise (m/s)': v_c
        }
        all_results.append(result_data)
        print(f"  > Found Design: η={final_eta:.3f}, L/D={final_ld:.2f}")

    # --- 6. Save Results ---
    df_results = pd.DataFrame(all_results)
    tables_dir = Path(P["outputs_tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "05_optimization_results.csv"
    df_results.to_csv(results_path, index=False, float_format="%.4f")
    
    print(f"\nOptimization complete. Results saved to: {results_path}")

if __name__ == "__main__":
    main()