# Scripts/06_analyze_results.py
# Analyzes optimization results and generates a Pareto front plot, 
# including the original training data and clear labels for optimized points.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import yaml
from joblib import load
import xgboost as xgb

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

def get_performance_indices(df_designs, model_hover, model_cruise, baselines, C):
    """
    Calculates both normalized and absolute performance indices for a dataframe of designs.
    """
    OPT = C["optimization"]
    hover_op_points = np.array(OPT.get("hover_op_points", [2, 4, 6, 8, 10]))
    cruise_op_points = np.array(OPT.get("cruise_op_points", [4, 6, 8, 10, 12]))
    
    hover_features = model_hover.get_booster().feature_names
    cruise_features = model_cruise.get_booster().feature_names
    
    baseline_hover_eta, baseline_cruise_ld = baselines
    
    indices = []
    for _, design in df_designs.iterrows():
        X_geom = design[C["geometry_cols"]]
        df_hover_input = pd.DataFrame([[*X_geom, op] for op in hover_op_points], columns=hover_features)
        df_cruise_input = pd.DataFrame([[*X_geom, op] for op in cruise_op_points], columns=cruise_features)
        
        abs_hover_eta = np.median(model_hover.predict(df_hover_input))
        abs_cruise_ld = np.median(model_cruise.predict(df_cruise_input))

        I_hover = (abs_hover_eta - baseline_hover_eta) / baseline_hover_eta
        I_cruise = (abs_cruise_ld - baseline_cruise_ld) / baseline_cruise_ld
        
        indices.append({
            'I_hover': I_hover, 'I_cruise': I_cruise,
            'abs_hover_eta': abs_hover_eta, 'abs_cruise_ld': abs_cruise_ld
        })
        
    return pd.DataFrame(indices)

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    script_dir = Path(__file__).parent

    # --- 1. Load All Necessary Files ---
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
        raise SystemExit(f"Error loading models from '{models_dir}'. Have you run 03_train_models.py yet? Details: {e}")
        
    # --- 2. Establish Baselines and Predict Performance for All Designs ---
    print("--- Calculating performance for original DOE and optimized points ---")
    
    baseline_hover_eta = df_full[df_full['flight_mode'] == 'hover']['performance'].median()
    baseline_cruise_ld = df_full[df_full['flight_mode'] == 'cruise']['performance'].median()
    baselines = (baseline_hover_eta, baseline_cruise_ld)
    
    df_hover_designs = df_full[df_full['flight_mode'] == 'hover'].drop_duplicates(subset=GEO_COLS)
    df_cruise_designs = df_full[df_full['flight_mode'] == 'cruise'].drop_duplicates(subset=GEO_COLS)

    hover_doe_perf = get_performance_indices(df_hover_designs, model_hover, model_cruise, baselines, C)
    cruise_doe_perf = get_performance_indices(df_cruise_designs, model_hover, model_cruise, baselines, C)
    optimized_perf = get_performance_indices(df_results, model_hover, model_cruise, baselines, C)
    
    # --- 3. Create and Save Pareto Front Plots ---
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "06_pareto_fronts.pdf"

    with plt.rc_context({'font.size': 12}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
        
        # --- Plot 1: Normalized Improvement ---
        ax1.set_title('Pareto Front: Normalized Performance Improvement', fontsize=18)
        ax1.scatter(0, 0, c='black', s=200, marker='X', label='Baseline DOE Median', zorder=15)
        ax1.scatter(cruise_doe_perf['I_cruise'], cruise_doe_perf['I_hover'], c='lightskyblue', s=50, label='CFD Cruise DOE', alpha=0.8, zorder=4)
        ax1.scatter(hover_doe_perf['I_cruise'], hover_doe_perf['I_hover'], c='lightcoral', s=50, label='Experimental Hover DOE', alpha=0.8, zorder=3)
        scatter1 = ax1.scatter(optimized_perf['I_cruise'], optimized_perf['I_hover'], c=df_results['w_hover'], cmap='viridis', s=200, edgecolors='k', zorder=10)
        ax1.plot(optimized_perf['I_cruise'], optimized_perf['I_hover'], 'k--', alpha=0.6, zorder=5)
        ax1.set_xlabel('Cruise Improvement Index (I_cruise)', fontsize=14)
        ax1.set_ylabel('Hover Improvement Index (I_hover)', fontsize=14)
        ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)
        cbar1 = fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Hover Weight (w_hover)', fontsize=14)

        # --- Plot 2: Absolute Performance ---
        ax2.set_title('Pareto Front: Absolute Performance', fontsize=18)
        ax2.scatter(baseline_cruise_ld, baseline_hover_eta, c='black', s=200, marker='X', label='Baseline DOE Median', zorder=15)
        ax2.scatter(cruise_doe_perf['abs_cruise_ld'], cruise_doe_perf['abs_hover_eta'], c='lightskyblue', s=50, label='CFD Cruise DOE', alpha=0.8, zorder=4)
        ax2.scatter(hover_doe_perf['abs_cruise_ld'], hover_doe_perf['abs_hover_eta'], c='lightcoral', s=50, label='Experimental Hover DOE', alpha=0.8, zorder=3)
        scatter2 = ax2.scatter(optimized_perf['abs_cruise_ld'], optimized_perf['abs_hover_eta'], c=df_results['w_hover'], cmap='viridis', s=200, edgecolors='k', zorder=10)
        ax2.plot(optimized_perf['abs_cruise_ld'], optimized_perf['abs_hover_eta'], 'k--', alpha=0.6, zorder=5)
        ax2.set_xlabel('Cruise Performance (L/D)', fontsize=14)
        ax2.set_ylabel('Hover Performance (η)', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
        cbar2 = fig.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Hover Weight (w_hover)', fontsize=14)

        # --- MODIFIED: Add direct text labels to the optimized points on both plots ---
        for i, row in df_results.iterrows():
            # Annotate normalized plot
            ax1.text(optimized_perf.loc[i, 'I_cruise'], optimized_perf.loc[i, 'I_hover'] + 0.01, f"w={row['w_hover']}", ha='center', fontsize=9, weight='bold')
            # Annotate absolute plot
            ax2.text(optimized_perf.loc[i, 'abs_cruise_ld'], optimized_perf.loc[i, 'abs_hover_eta'] + 0.005, f"w={row['w_hover']}", ha='center', fontsize=9, weight='bold')

        fig.tight_layout(pad=3.0)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

    print(f"\nSuccessfully generated Pareto front plots: {plot_path}")

    # --- 4. Geometry vs. Weight Plot ---
    fig_geom, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig_geom.suptitle('Optimal Geometry vs. Hover Weight', fontsize=16)
    
    for ax_i, param in zip(axes.flatten(), GEO_COLS):
        ax_i.plot(df_results['w_hover'], df_results[param], 'o-')
        ax_i.set_xlabel('Hover Weight (w_hover)'); ax_i.set_ylabel(param)
        ax_i.grid(True, linestyle='--', alpha=0.6)
        
    fig_geom.tight_layout(rect=[0, 0, 1, 0.96])
    geom_plot_path = plots_dir / "06_geometry_vs_weight.pdf"
    fig_geom.savefig(geom_plot_path)
    plt.close(fig_geom)
    
    print(f"Saved geometry analysis plot to: '{geom_plot_path}'")

if __name__ == "__main__":
    main()