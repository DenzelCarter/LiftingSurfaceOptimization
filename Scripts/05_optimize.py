# Scripts/05_optimize.py
# Optimizes geometry and operational speeds, and determines the optimal AoA for each flight mode.

import os
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from joblib import load
import xgboost as xgb
from pathlib import Path
import yaml

def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

def get_peak_performance(df_designs, model_hover, model_cruise, C):
    """
    Calculates the peak performance AND the AoA at which it occurs.
    """
    OPT = C["optimization"]
    hover_aoa_sweep = np.array(OPT.get("hover_op_points"))
    cruise_aoa_sweep = np.array(OPT.get("cruise_op_points"))
    
    hover_features = model_hover.get_booster().feature_names
    cruise_features = model_cruise.get_booster().feature_names
    
    perf_data = []
    for _, design in df_designs.iterrows():
        # --- MODIFIED: Robustly select only the geometric parameters for unpacking ---
        # This ignores the old aoa_root column from the initial data.
        geom_cols_to_unpack = [col for col in C["geometry_cols"] if col != 'aoa_root (deg)']
        AR, lam, twist = design[geom_cols_to_unpack]

        hover_speed = design["op_speed_hover (rpm)"]
        cruise_speed = design["op_speed_cruise (m/s)"]

        # --- Hover Peak Performance ---
        hover_inputs = [[AR, lam, aoa, twist, hover_speed] for aoa in hover_aoa_sweep]
        df_hover_sweep = pd.DataFrame(hover_inputs, columns=hover_features)
        hover_predictions = model_hover.predict(df_hover_sweep)
        peak_hover_eta = np.max(hover_predictions)
        optimal_hover_aoa = hover_aoa_sweep[np.argmax(hover_predictions)]

        # --- Cruise Peak Performance ---
        cruise_inputs = [[AR, lam, aoa, twist, cruise_speed] for aoa in cruise_aoa_sweep]
        df_cruise_sweep = pd.DataFrame(cruise_inputs, columns=cruise_features)
        cruise_predictions = model_cruise.predict(df_cruise_sweep)
        peak_cruise_ld = np.max(cruise_predictions)
        optimal_cruise_aoa = cruise_aoa_sweep[np.argmax(cruise_predictions)]
        
        perf_data.append({
            'peak_hover_eta': peak_hover_eta, 'optimal_hover_aoa': optimal_hover_aoa,
            'peak_cruise_ld': peak_cruise_ld, 'optimal_cruise_aoa': optimal_cruise_aoa
        })
        
    return pd.DataFrame(perf_data)

def main():
    C = load_config()
    P = C["paths"]
    B = C["bounds"]
    GEO_COLS = C["geometry_cols"]
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
        raise SystemExit(f"Error loading files. Have you run scripts 01-04? Details: {e}")

    # --- 2. Establish a Robust Baseline ---
    print("--- Establishing a robust baseline from the peak performance of DOE designs ---")
    
    # Use only the true geometric columns to define a unique design
    unique_geo_cols = [col for col in GEO_COLS if col != 'aoa_root (deg)']
    df_designs = df_full.drop_duplicates(subset=unique_geo_cols).copy()
    
    df_designs['op_speed_hover (rpm)'] = df_full[df_full['flight_mode']=='hover']['op_speed'].median()
    df_designs['op_speed_cruise (m/s)'] = df_full[df_full['flight_mode']=='cruise']['op_speed'].median()

    initial_peak_perf = get_peak_performance(df_designs, model_hover, model_cruise, C)
    
    baseline_hover_eta = initial_peak_perf['peak_hover_eta'].median()
    baseline_cruise_ld = initial_peak_perf['peak_cruise_ld'].median()
    
    print(f"Robust Baseline Hover Score (Median Peak η): {baseline_hover_eta:.4f}")
    print(f"Robust Baseline Cruise Score (Median Peak L/D): {baseline_cruise_ld:.4f}")
    
    # --- 3. Define the Objective Function ---
    def objective_function(X, w_hover):
        # X is now [AR, lambda, twist, op_speed_hover, op_speed_cruise]
        design_cols = [*unique_geo_cols, "op_speed_hover (rpm)", "op_speed_cruise (m/s)"]
        design_df = pd.DataFrame([X], columns=design_cols)
        
        peak_perf = get_peak_performance(design_df, model_hover, model_cruise, C)
        
        I_hover = (peak_perf['peak_hover_eta'][0] - baseline_hover_eta) / baseline_hover_eta
        I_cruise = (peak_perf['peak_cruise_ld'][0] - baseline_cruise_ld) / baseline_cruise_ld
        
        J = w_hover * I_hover + (1.0 - w_hover) * I_cruise
        return -J

    # --- 4. Run Optimization ---
    print("\n--- Starting Optimization ---")
    bounds = [
        B["AR"], B["lambda"], B["twist (deg)"],
        B["op_speed_hover (rpm)"], B["op_speed_cruise (m/s)"]
    ]
    all_results = []
    
    for w in C["optimization"]["w_hover"]:
        print(f"Optimizing for w_hover = {w}...")
        result = differential_evolution(func=objective_function, bounds=bounds, args=(w,), maxiter=200, popsize=20, seed=42)
        
        optimal_X = result.x
        design_cols = [*unique_geo_cols, "op_speed_hover (rpm)", "op_speed_cruise (m/s)"]
        optimal_X_df = pd.DataFrame([optimal_X], columns=design_cols)
        
        final_peak_perf = get_peak_performance(optimal_X_df, model_hover, model_cruise, C)
        final_eta = final_peak_perf['peak_hover_eta'][0]
        final_ld = final_peak_perf['peak_cruise_ld'][0]
        final_aoa_h = final_peak_perf['optimal_hover_aoa'][0]
        final_aoa_c = final_peak_perf['optimal_cruise_aoa'][0]

        I_hover_final = (final_eta - baseline_hover_eta) / baseline_hover_eta
        I_cruise_final = (final_ld - baseline_cruise_ld) / baseline_cruise_ld
        
        result_data = {
            'w_hover': w, 'I_hover': I_hover_final, 'I_cruise': I_cruise_final,
            'peak_hover_eta': final_eta, 'peak_cruise_ld': final_ld,
            'AR': optimal_X[0], 'lambda': optimal_X[1], 'twist (deg)': optimal_X[2], 
            'op_speed_hover (rpm)': optimal_X[3], 'op_speed_cruise (m/s)': optimal_X[4],
            'aoa_root_hover (deg)': final_aoa_h, 'aoa_root_cruise (deg)': final_aoa_c
        }
        all_results.append(result_data)
        print(f"  > Found Design: η_max={final_eta:.3f} @ {final_aoa_h}°, (L/D)_max={final_ld:.2f} @ {final_aoa_c}°")

    # --- 5. Save Results ---
    df_results = pd.DataFrame(all_results)
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "05_optimization_results.csv"
    df_results.to_csv(results_path, index=False, float_format="%.4f")
    
    print(f"\nOptimization complete. Results saved to: {results_path}")

if __name__ == "__main__":
    main()