# Scripts/05_optimize.py
# Implements a robustly normalized UCB optimization.

import os
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from joblib import load
import yaml
from pathlib import Path

# --- Functions (load_config, get_peak_ucb_performance) are unchanged ---
def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

def get_peak_ucb_performance(df_designs, gpr_hover, gpr_cruise, hover_aoa_sweep, cruise_aoa_sweep, C):
    """
    Calculates the peak Upper Confidence Bound (UCB) performance and the AoA at which it occurs.
    """
    kappa = C["optimization"].get("kappa", 1.0) # Load kappa from config
    hover_features = gpr_hover.feature_names_in_
    cruise_features = gpr_cruise.feature_names_in_
    
    perf_data = []
    for _, design in df_designs.iterrows():
        AR, lam, twist = design[['AR', 'lambda', 'twist (deg)']]
        hover_speed = design["op_speed_hover (rpm)"]
        cruise_speed = design["op_speed_cruise (m/s)"]

        # Hover Peak UCB Performance
        hover_inputs = [[AR, lam, aoa, twist, hover_speed] for aoa in hover_aoa_sweep]
        df_hover_sweep = pd.DataFrame(hover_inputs, columns=hover_features)
        mean_h, std_h = gpr_hover.predict(df_hover_sweep, return_std=True)
        ucb_h = mean_h + kappa * std_h
        
        peak_hover_ucb = np.max(ucb_h)
        peak_hover_eta_at_ucb = mean_h[np.argmax(ucb_h)]
        optimal_hover_aoa = hover_aoa_sweep[np.argmax(ucb_h)]

        # Cruise Peak UCB Performance
        cruise_inputs = [[AR, lam, aoa, twist, cruise_speed] for aoa in cruise_aoa_sweep]
        df_cruise_sweep = pd.DataFrame(cruise_inputs, columns=cruise_features)
        mean_c, std_c = gpr_cruise.predict(df_cruise_sweep, return_std=True)
        ucb_c = mean_c + kappa * std_c

        peak_cruise_ucb = np.max(ucb_c)
        peak_cruise_ld_at_ucb = mean_c[np.argmax(ucb_c)]
        optimal_cruise_aoa = cruise_aoa_sweep[np.argmax(ucb_c)]
        
        perf_data.append({
            'peak_hover_ucb': peak_hover_ucb, 'peak_hover_eta': peak_hover_eta_at_ucb, 'optimal_hover_aoa': optimal_hover_aoa,
            'peak_cruise_ucb': peak_cruise_ucb, 'peak_cruise_ld': peak_cruise_ld_at_ucb, 'optimal_cruise_aoa': optimal_cruise_aoa
        })
    return pd.DataFrame(perf_data)


def main():
    C = load_config()
    P = C["paths"]
    OPT = C["optimization"]
    script_dir = Path(__file__).parent

    # --- 1. Load Data and GPR Models ---
    print("--- Loading surrogate models and data ---")
    models_dir = (script_dir / P["outputs_models"]).resolve()
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    
    try:
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        gpr_cruise = load(models_dir / "gpr_cruise_model.joblib")
        df_full = pd.read_parquet(master_parquet_path)
    except Exception as e:
        raise SystemExit(f"Error loading files. Have you run scripts 01-04? Details: {e}")

    # --- 2. Dynamically Determine Bounds and Sweeps ---
    print("--- Automatically determining optimization bounds and sweeps from data ---")
    df_hover = df_full[df_full['flight_mode'] == 'hover']
    df_cruise = df_full[df_full['flight_mode'] == 'cruise']

    fixed_geom_vars = ['AR', 'lambda', 'twist (deg)']
    opt_vars = fixed_geom_vars + ["op_speed_hover (rpm)", "op_speed_cruise (m/s)"]
    bounds = [(df_full[col].min(), df_full[col].max()) for col in fixed_geom_vars]
    bounds.append((df_hover['op_speed'].min(), df_hover['op_speed'].max()))
    bounds.append((df_cruise['op_speed'].min(), df_cruise['op_speed'].max()))

    n_op_points = OPT.get("n_op_points", 10)
    hover_aoa_sweep = np.linspace(df_hover['aoa_root (deg)'].min(), df_hover['aoa_root (deg)'].max(), n_op_points)
    cruise_aoa_sweep = np.linspace(df_cruise['aoa_root (deg)'].min(), df_cruise['aoa_root (deg)'].max(), n_op_points)
    
    # --- 3. Establish a Robust Baseline using the GPR Models ---
    df_designs = df_full.drop_duplicates(subset=fixed_geom_vars).copy()
    df_designs['op_speed_hover (rpm)'] = df_hover['op_speed'].median()
    df_designs['op_speed_cruise (m/s)'] = df_cruise['op_speed'].median()

    initial_peak_perf = get_peak_ucb_performance(df_designs, gpr_hover, gpr_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)
    baseline_hover_eta = initial_peak_perf['peak_hover_eta'].median()
    baseline_cruise_ld = initial_peak_perf['peak_cruise_ld'].median()
    
    print(f"Robust Baseline Hover Score (Median Peak η): {baseline_hover_eta:.4f}")
    print(f"Robust Baseline Cruise Score (Median Peak L/D): {baseline_cruise_ld:.4f}")

    # --- 4. Define the UCB Objective Function with Robust Normalization ---
    def objective_function(X, w_hover):
        design_df = pd.DataFrame([X], columns=opt_vars)
        peak_perf = get_peak_ucb_performance(design_df, gpr_hover, gpr_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)
        
        # --- MODIFIED: Use robust baseline normalization for the objective ---
        I_hover_ucb = (peak_perf['peak_hover_ucb'][0] - baseline_hover_eta) / baseline_hover_eta
        I_cruise_ucb = (peak_perf['peak_cruise_ucb'][0] - baseline_cruise_ld) / baseline_cruise_ld
        
        J = w_hover * I_hover_ucb + (1.0 - w_hover) * I_cruise_ucb
        return -J

    # --- 5. Run Optimization ---
    print("\n--- Starting UCB Optimization ---")
    all_results = []
    
    for w in C["optimization"]["w_hover"]:
        print(f"Optimizing for w_hover = {w}...")
        result = differential_evolution(func=objective_function, bounds=bounds, args=(w,), maxiter=200, popsize=20, seed=42)
        
        optimal_X = result.x
        optimal_X_df = pd.DataFrame([optimal_X], columns=opt_vars)
        
        final_perf = get_peak_ucb_performance(optimal_X_df, gpr_hover, gpr_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)
        final_eta, final_ld = final_perf['peak_hover_eta'][0], final_perf['peak_cruise_ld'][0]
        final_aoa_h, final_aoa_c = final_perf['optimal_hover_aoa'][0], final_perf['optimal_cruise_aoa'][0]

        result_data = {
            'w_hover': w,
            'peak_hover_eta': final_eta, 'peak_cruise_ld': final_ld,
            'AR': optimal_X[0], 'lambda': optimal_X[1], 'twist (deg)': optimal_X[2], 
            'op_speed_hover (rpm)': optimal_X[3], 'op_speed_cruise (m/s)': optimal_X[4],
            'aoa_root_hover (deg)': final_aoa_h, 'aoa_root_cruise (deg)': final_aoa_c
        }
        all_results.append(result_data)
        print(f"  > Found Design: η_max={final_eta:.3f}, (L/D)_max={final_ld:.2f}")

    # --- 6. Save Results ---
    df_results = pd.DataFrame(all_results)
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "05_optimization_results.csv"
    df_results.to_csv(results_path, index=False, float_format="%.4f")
    
    print(f"\nOptimization complete. Results saved to: {results_path}")

if __name__ == "__main__":
    main()