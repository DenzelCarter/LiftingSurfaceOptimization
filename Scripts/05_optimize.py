# Scripts/05_optimize.py
# Runs the multi-objective optimization to find the Pareto optimal designs
# based on a normalized improvement score.

import os
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from joblib import load
import xgboost as xgb
from pathlib import Path
import yaml

# --- Configuration Loading ---
def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

def main():
    C = load_config()
    P = C["paths"]
    B = C["bounds"]
    OPT = C["optimization"]
    script_dir = Path(__file__).parent

    # --- 1. Load Data and Models ---
    print("--- Loading surrogate models and data ---")
    models_dir = (script_dir / P["outputs_models"]).resolve()
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    
    try:
        model_hover = xgb.XGBRegressor(); model_hover.load_model(models_dir / "xgboost_hover_model.json")
        model_cruise = xgb.XGBRegressor(); model_cruise.load_model(models_dir / "xgboost_cruise_model.json")
        df_full = pd.read_parquet(master_parquet_path)
    except Exception as e:
        raise SystemExit(f"Error loading models or data. Have you run scripts 01-03? Details: {e}")

    # --- 2. Establish Baseline Performance from Original DOE ---
    df_hover_doe = df_full[df_full['flight_mode'] == 'hover']
    df_cruise_doe = df_full[df_full['flight_mode'] == 'cruise']
    
    baseline_hover_eta = df_hover_doe['performance'].median()
    baseline_cruise_ld = df_cruise_doe['performance'].median()
    
    print(f"Baseline Hover Median Efficiency: {baseline_hover_eta:.3f}")
    print(f"Baseline Cruise Median L/D: {baseline_cruise_ld:.3f}")
    
    # --- 3. Define the Objective Function ---
    hover_features = model_hover.get_booster().feature_names
    cruise_features = model_cruise.get_booster().feature_names
    
    hover_op_points = np.array(OPT.get("hover_op_points", [2, 4, 6, 8, 10]))
    cruise_op_points = np.array(OPT.get("cruise_op_points", [4, 6, 8, 10, 12]))

    def objective_function(X, w_hover):
        """
        Objective function to be maximized. Calculates normalized improvement.
        """
        df_hover_input = pd.DataFrame([[*X, op] for op in hover_op_points], columns=hover_features)
        df_cruise_input = pd.DataFrame([[*X, op] for op in cruise_op_points], columns=cruise_features)
        
        # Predict absolute performance
        median_hover_eta = np.median(model_hover.predict(df_hover_input))
        median_cruise_ld = np.median(model_cruise.predict(df_cruise_input))

        # --- MODIFIED: Calculate Normalized Improvement Score ---
        I_hover = (median_hover_eta - baseline_hover_eta) / baseline_hover_eta
        I_cruise = (median_cruise_ld - baseline_cruise_ld) / baseline_cruise_ld
        
        J = w_hover * I_hover + (1.0 - w_hover) * I_cruise
        return -J

    # --- 4. Run Optimization ---
    print("\n--- Starting Optimization ---")
    bounds = [B["AR"], B["lambda"], B["i_r (deg)"], B["epsilon (deg)"]]
    all_results = []
    
    for w in OPT["w_hover"]:
        print(f"Optimizing for w_hover = {w}...")
        result = differential_evolution(func=objective_function, bounds=bounds, args=(w,), maxiter=200, popsize=20, seed=42)
        
        # Recalculate final performance for the optimal design
        optimal_X = result.x
        df_hover_final = pd.DataFrame([[*optimal_X, op] for op in hover_op_points], columns=hover_features)
        df_cruise_final = pd.DataFrame([[*optimal_X, op] for op in cruise_op_points], columns=cruise_features)

        final_median_eta = np.median(model_hover.predict(df_hover_final))
        final_median_ld = np.median(model_cruise.predict(df_cruise_final))

        I_hover_final = (final_median_eta - baseline_hover_eta) / baseline_hover_eta
        I_cruise_final = (final_median_ld - baseline_cruise_ld) / baseline_cruise_ld
        
        result_data = {
            'w_hover': w, 'I_hover': I_hover_final, 'I_cruise': I_cruise_final,
            'AR': optimal_X[0], 'lambda': optimal_X[1], 'i_r (deg)': optimal_X[2], 'epsilon (deg)': optimal_X[3]
        }
        all_results.append(result_data)
        print(f"  > Found Design: I_hover = {I_hover_final:+.1%}, I_cruise = {I_cruise_final:+.1%}")

    # --- 5. Save Results ---
    df_results = pd.DataFrame(all_results)
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "05_optimization_results.csv"
    df_results.to_csv(results_path, index=False, float_format="%.4f")
    
    print(f"\nOptimization complete. Results saved to: {results_path}")

if __name__ == "__main__":
    main()