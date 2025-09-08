# Scripts/06_analyze_results.py
# Analyzes and plots the final, comprehensive optimization results from 05_optimize.py.

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

def get_peak_performance(df_designs, model_hover, model_cruise, C):
    """
    Calculates the peak performance (max eta and L/D) for a set of designs
    by sweeping over a range of Angles of Attack. This is needed to plot the initial DOE.
    """
    OPT = C["optimization"]
    hover_aoa_sweep = np.array(OPT.get("hover_op_points"))
    cruise_aoa_sweep = np.array(OPT.get("cruise_op_points"))
    
    hover_features = model_hover.get_booster().feature_names
    cruise_features = model_cruise.get_booster().feature_names
    
    perf_data = []
    for _, design in df_designs.iterrows():
        geom_cols = [col for col in C["geometry_cols"] if col != 'aoa_root (deg)']
        AR, lam, twist = design[geom_cols]
        hover_speed = design["op_speed_hover (rpm)"]
        cruise_speed = design["op_speed_cruise (m/s)"]

        # Hover Peak Performance
        hover_inputs = [[AR, lam, aoa, twist, hover_speed] for aoa in hover_aoa_sweep]
        df_hover_sweep = pd.DataFrame(hover_inputs, columns=hover_features)
        peak_hover_eta = np.max(model_hover.predict(df_hover_sweep))

        # Cruise Peak Performance
        cruise_inputs = [[AR, lam, aoa, twist, cruise_speed] for aoa in cruise_aoa_sweep]
        df_cruise_sweep = pd.DataFrame(cruise_inputs, columns=cruise_features)
        peak_cruise_ld = np.max(model_cruise.predict(df_cruise_sweep))
        
        perf_data.append({'peak_hover_eta': peak_hover_eta, 'peak_cruise_ld': peak_cruise_ld})
        
    return pd.DataFrame(perf_data)

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    script_dir = Path(__file__).parent

    # --- 1. Load Optimization Results and Original Data ---
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    results_path = tables_dir / "05_optimization_results.csv"
    if not results_path.exists():
        raise SystemExit(f"Error: Optimization results not found at '{results_path}'. Please run 05_optimize.py first.")
    df_results = pd.read_csv(results_path)

    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    df_full = pd.read_parquet(master_parquet_path)
    
    # --- 2. Load Models to Calculate Performance of Initial DOE for Context ---
    models_dir = (script_dir / P["outputs_models"]).resolve()
    try:
        model_hover = xgb.XGBRegressor(); model_hover.load_model(models_dir / "xgboost_hover_model.json")
        model_cruise = xgb.XGBRegressor(); model_cruise.load_model(models_dir / "xgboost_cruise_model.json")
    except Exception as e:
        raise SystemExit(f"Error loading models. Have you run scripts 01-04? Details: {e}")

    # --- 3. Calculate Baseline and Performance of Original DOE ---
    unique_geo_cols = [col for col in GEO_COLS if col != 'aoa_root (deg)']
    df_designs_initial = df_full.drop_duplicates(subset=unique_geo_cols).copy()
    
    df_designs_initial['op_speed_hover (rpm)'] = df_full[df_full['flight_mode']=='hover']['op_speed'].median()
    df_designs_initial['op_speed_cruise (m/s)'] = df_full[df_full['flight_mode']=='cruise']['op_speed'].median()
    
    initial_doe_perf = get_peak_performance(df_designs_initial, model_hover, model_cruise, C)
    
    baseline_hover_eta = initial_doe_perf['peak_hover_eta'].median()
    baseline_cruise_ld = initial_doe_perf['peak_cruise_ld'].median()

    # --- 4. Create Combined Pareto Front Plot ---
    print("\n--- Generating Combined Pareto Front Plot ---")
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "06_pareto_front_combined.pdf"

    with plt.rc_context({'font.size': 12}):
        fig, ax1 = plt.subplots(figsize=(13, 10))
        ax1.set_title('Pareto Front: Peak Hover vs. Cruise Performance', fontsize=18, pad=20)
        ax1.set_xlabel('Peak Cruise Performance ($(L/D)_{max}$)', fontsize=14)
        ax1.set_ylabel('Peak Hover Performance ($\\eta_{max}$)', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax1.scatter(baseline_cruise_ld, baseline_hover_eta, c='black', s=200, marker='X', label='Baseline DOE Median', zorder=15)
        ax1.scatter(initial_doe_perf['peak_cruise_ld'], initial_doe_perf['peak_hover_eta'], c='gray', s=60, label='Initial DOE Designs', alpha=0.8, zorder=3)
        scatter = ax1.scatter(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], c=df_results['w_hover'], cmap='viridis', s=250, edgecolors='k', zorder=10)
        ax1.plot(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], 'k--', alpha=0.6, zorder=5)
        ax1.legend(loc='upper left')

        # Secondary Axes
        ax2 = ax1.twinx(); ax3 = ax1.twiny()
        ax2.set_ylabel('Hover Improvement Index ($I_{hover}$)', fontsize=14)
        ax3.set_xlabel('Cruise Improvement Index ($I_{cruise}$)', fontsize=14)
        y1_lim = ax1.get_ylim(); ax2.set_ylim([(y - baseline_hover_eta) / baseline_hover_eta for y in y1_lim])
        x1_lim = ax1.get_xlim(); ax3.set_xlim([(x - baseline_cruise_ld) / baseline_cruise_ld for x in x1_lim])
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax3.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Layout and Colorbar
        fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        cbar_ax = fig.add_axes([0.83, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Hover Weight ($w_{hover}$)', fontsize=14)
        
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
    print(f"Successfully generated combined Pareto front plot: {plot_path}")

    # --- 5. Parameter vs. Weight Plot ---
    print("\n--- Generating Parameter vs. Weight Plot ---")
    param_cols = [
        'AR', 'lambda', 'twist (deg)', 'op_speed_hover (rpm)', 
        'op_speed_cruise (m/s)', 'aoa_root_hover (deg)', 'aoa_root_cruise (deg)'
    ]
    fig_geom, axes = plt.subplots(4, 2, figsize=(12, 20))
    fig_geom.suptitle('Optimal Parameters vs. Hover Weight', fontsize=18)
    
    # Flatten axes array and remove the last one if there's an odd number of plots
    axes_flat = axes.flatten()
    if len(param_cols) % 2 != 0:
        fig_geom.delaxes(axes_flat[-1])

    for i, param in enumerate(param_cols):
        ax_i = axes_flat[i]
        ax_i.plot(df_results['w_hover'], df_results[param], 'o-')
        ax_i.set_xlabel('Hover Weight ($w_{hover}$)')
        ax_i.set_ylabel(param)
        ax_i.grid(True, linestyle='--', alpha=0.6)
        
    fig_geom.tight_layout(rect=[0, 0.03, 1, 0.97])
    geom_plot_path = plots_dir / "06_parameters_vs_weight.pdf"
    fig_geom.savefig(geom_plot_path)
    plt.close(fig_geom)
    print(f"Saved parameter analysis plot to: '{geom_plot_path}'")

if __name__ == "__main__":
    main()