# Scripts/06_analyze_results.py
# Analyzes and plots the final optimization results, and includes a "failsafe"
# analysis of the best designs found in the original measured data.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import yaml
from joblib import load
import xgboost as xgb

def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

def get_peak_performance(df_designs, model_hover, model_cruise, hover_aoa_sweep, cruise_aoa_sweep, C):
    """
    Calculates the peak performance (max eta and L/D) for a set of designs.
    """
    hover_features = model_hover.get_booster().feature_names
    cruise_features = model_cruise.get_booster().feature_names
    
    perf_data = []
    for _, design in df_designs.iterrows():
        fixed_geom_vars = ['AR', 'lambda', 'twist (deg)']
        AR, lam, twist = design[fixed_geom_vars]
        hover_speed = design["op_speed_hover (rpm)"]
        cruise_speed = design["op_speed_cruise (m/s)"]

        hover_inputs = [[AR, lam, aoa, twist, hover_speed] for aoa in hover_aoa_sweep]
        df_hover_sweep = pd.DataFrame(hover_inputs, columns=hover_features)
        peak_hover_eta = np.max(model_hover.predict(df_hover_sweep))

        cruise_inputs = [[AR, lam, aoa, twist, cruise_speed] for aoa in cruise_aoa_sweep]
        df_cruise_sweep = pd.DataFrame(cruise_inputs, columns=cruise_features)
        peak_cruise_ld = np.max(model_cruise.predict(df_cruise_sweep))
        
        perf_data.append({'peak_hover_eta': peak_hover_eta, 'peak_cruise_ld': peak_cruise_ld})
    return pd.DataFrame(perf_data)

def main():
    C = load_config()
    P = C["paths"]
    OPT = C["optimization"]
    script_dir = Path(__file__).parent

    # --- 1. Load All Data ---
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    results_path = tables_dir / "05_optimization_results.csv"
    if not results_path.exists():
        raise SystemExit(f"Error: Optimization results not found at '{results_path}'. Please run 05_optimize.py first.")
    df_results = pd.read_csv(results_path)

    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    df_full = pd.read_parquet(master_parquet_path)
    
    models_dir = (script_dir / P["outputs_models"]).resolve()
    try:
        model_hover = xgb.XGBRegressor(); model_hover.load_model(models_dir / "xgboost_hover_model.json")
        model_cruise = xgb.XGBRegressor(); model_cruise.load_model(models_dir / "xgboost_cruise_model.json")
    except Exception as e:
        raise SystemExit(f"Error loading models. Have you run scripts 01-04? Details: {e}")

    # --- 2. Calculate Performance of Initial DOE for Plotting Context ---
    df_hover = df_full[df_full['flight_mode'] == 'hover']
    df_cruise = df_full[df_full['flight_mode'] == 'cruise']
    
    n_op_points = OPT.get("n_op_points", 10)
    hover_aoa_sweep = np.linspace(df_hover['aoa_root (deg)'].min(), df_hover['aoa_root (deg)'].max(), n_op_points)
    cruise_aoa_sweep = np.linspace(df_cruise['aoa_root (deg)'].min(), df_cruise['aoa_root (deg)'].max(), n_op_points)

    fixed_geom_vars = ['AR', 'lambda', 'twist (deg)']
    df_designs_initial = df_full.drop_duplicates(subset=fixed_geom_vars).copy()
    df_designs_initial['op_speed_hover (rpm)'] = df_hover['op_speed'].median()
    df_designs_initial['op_speed_cruise (m/s)'] = df_cruise['op_speed'].median()
    
    initial_doe_perf = get_peak_performance(df_designs_initial, model_hover, model_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)
    
    baseline_hover_eta = initial_doe_perf['peak_hover_eta'].median()
    baseline_cruise_ld = initial_doe_perf['peak_cruise_ld'].median()

    # --- 3. Find Best Measured Designs (Failsafe Analysis) ---
    best_measured_hover_point = df_hover.loc[df_hover['performance'].idxmax()]
    best_measured_cruise_point = df_cruise.loc[df_cruise['performance'].idxmax()]

    # To plot these, we need their peak performance scores across all AoAs
    df_best_h_geom = pd.DataFrame([best_measured_hover_point])
    df_best_h_geom['op_speed_hover (rpm)'] = best_measured_hover_point['op_speed']
    df_best_h_geom['op_speed_cruise (m/s)'] = df_cruise['op_speed'].median() # Assign a median cruise speed
    best_h_peak_perf = get_peak_performance(df_best_h_geom, model_hover, model_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)
    
    df_best_c_geom = pd.DataFrame([best_measured_cruise_point])
    df_best_c_geom['op_speed_hover (rpm)'] = df_hover['op_speed'].median() # Assign a median hover speed
    df_best_c_geom['op_speed_cruise (m/s)'] = best_measured_cruise_point['op_speed']
    best_c_peak_perf = get_peak_performance(df_best_c_geom, model_hover, model_cruise, hover_aoa_sweep, cruise_aoa_sweep, C)

    # --- 4. Create Combined Pareto Front Plot ---
    print("\n--- Generating Combined Pareto Front Plot ---")
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "06_pareto_front_combined.pdf"

    with plt.rc_context({'font.size': 12}):
        fig, ax1 = plt.subplots(figsize=(13, 10))
        # ... (Plotting setup is the same)
        ax1.set_title('Pareto Front: Peak Hover vs. Cruise Performance', fontsize=18, pad=20)
        ax1.set_xlabel('Peak Cruise Performance ($(L/D)_{max}$)', fontsize=14)
        ax1.set_ylabel('Peak Hover Performance ($\\eta_{max}$)', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax1.scatter(initial_doe_perf['peak_cruise_ld'], initial_doe_perf['peak_hover_eta'], c='gray', s=60, label='Initial DOE Designs', alpha=0.8, zorder=3)
        scatter = ax1.scatter(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], c=df_results['w_hover'], cmap='viridis', s=250, edgecolors='k', zorder=10, label='Optimized Designs')
        ax1.plot(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], 'k--', alpha=0.6, zorder=5)
        
        # --- MODIFIED: Add best measured points to the plot ---
        ax1.scatter(best_h_peak_perf['peak_cruise_ld'], best_h_peak_perf['peak_hover_eta'], c='red', s=250, marker='*', edgecolors='k', label='Best Measured Hover Design', zorder=16)
        ax1.scatter(best_c_peak_perf['peak_cruise_ld'], best_c_peak_perf['peak_hover_eta'], c='blue', s=250, marker='*', edgecolors='k', label='Best Measured Cruise Design', zorder=16)

        ax1.legend(loc='upper left')
        # ... (Secondary axes and colorbar are the same)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

    print(f"Successfully generated combined Pareto front plot: {plot_path}")

    # --- 5. Parameter vs. Weight Plot ---
    # ... (This plot remains unchanged)

    # --- 6. Print Failsafe Analysis to Console ---
    print("\n\n--- Failsafe Analysis: Best Designs from Measured Data ---")
    print("\nBest Measured Hover Design:")
    print(best_measured_hover_point[['AR', 'lambda', 'twist (deg)', 'aoa_root (deg)', 'op_speed', 'performance']].to_string())
    print(f"\n  > Its peak (predicted) cruise L/D is: {best_h_peak_perf['peak_cruise_ld'].iloc[0]:.2f}")

    print("\nBest Measured Cruise Design:")
    print(best_measured_cruise_point[['AR', 'lambda', 'twist (deg)', 'aoa_root (deg)', 'op_speed', 'performance']].to_string())
    print(f"\n  > Its peak (predicted) hover efficiency is: {best_c_peak_perf['peak_hover_eta'].iloc[0]:.2f}")

if __name__ == "__main__":
    main()