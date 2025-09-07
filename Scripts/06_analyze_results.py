# Scripts/06_analyze_results.py
# Analyzes optimization results and generates a Pareto front plot, 
# including the original training data for context.

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

def get_improvement_indices(df_designs, model_hover, model_cruise, baselines, C):
    """
    Calculates the normalized I_hover and I_cruise improvement scores for a dataframe of designs.
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
        
        median_hover_eta = np.median(model_hover.predict(df_hover_input))
        median_cruise_ld = np.median(model_cruise.predict(df_cruise_input))

        I_hover = (median_hover_eta - baseline_hover_eta) / baseline_hover_eta
        I_cruise = (median_cruise_ld - baseline_cruise_ld) / baseline_cruise_ld
        indices.append({'I_hover': I_hover, 'I_cruise': I_cruise})
        
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
        
    # --- 2. Establish Baseline and Predict Performance for DOE Points ---
    print("--- Calculating performance improvement for original DOE points ---")
    
    # Establish baselines from the original data
    baseline_hover_eta = df_full[df_full['flight_mode'] == 'hover']['performance'].median()
    baseline_cruise_ld = df_full[df_full['flight_mode'] == 'cruise']['performance'].median()
    baselines = (baseline_hover_eta, baseline_cruise_ld)
    
    # Get unique geometries from the experimental and CFD DOEs
    df_hover_designs = df_full[df_full['flight_mode'] == 'hover'].drop_duplicates(subset=GEO_COLS)
    df_cruise_designs = df_full[df_full['flight_mode'] == 'cruise'].drop_duplicates(subset=GEO_COLS)

    # Calculate the full performance indices for each set of designs
    hover_doe_perf = get_improvement_indices(df_hover_designs, model_hover, model_cruise, baselines, C)
    cruise_doe_perf = get_improvement_indices(df_cruise_designs, model_hover, model_cruise, baselines, C)
    
    # --- 3. Create and Save Pareto Front Plot ---
    print("\n--- Generating Pareto Front Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot the baseline median marker at (0, 0)
    ax.scatter(0, 0, c='black', s=200, marker='X', label='Baseline DOE Median', zorder=15)
    
    # Plot the original DOE points as a background scatter
    ax.scatter(hover_doe_perf['I_hover'], hover_doe_perf['I_cruise'], 
               c='lightcoral', s=50, label='Experimental Hover DOE', alpha=0.8, zorder=3)
    ax.scatter(cruise_doe_perf['I_hover'], cruise_doe_perf['I_cruise'], 
               c='lightskyblue', s=50, label='CFD Cruise DOE', alpha=0.8, zorder=4)

    # Plot the Pareto Front (optimized results)
    scatter = ax.scatter(df_results['I_hover'], df_results['I_cruise'], 
                         c=df_results['w_hover'], cmap='viridis', s=200, 
                         edgecolors='k', zorder=10, label='Optimized Designs (Pareto Front)')
    ax.plot(df_results['I_hover'], df_results['I_cruise'], 'k--', alpha=0.6, zorder=5)

    # --- 4. Customize and Save the Plot ---
    ax.set_title('Pareto Front: Normalized Performance Improvement', fontsize=18)
    ax.set_xlabel('Hover Improvement Index (I_hover)', fontsize=14)
    ax.set_ylabel('Cruise Improvement Index (I_cruise)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Hover Weight (w_hover)', fontsize=14)
    
    ax.legend(loc='best', fontsize=12)
    
    for i, row in df_results.iterrows():
        ax.text(row['I_hover'], row['I_cruise'] + 0.01, f"w={row['w_hover']}", ha='center', fontsize=9, weight='bold')

    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "06_pareto_front_with_doe.pdf"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSuccessfully generated Pareto front plot.")
    print(f"Output saved to: '{plot_path}'")

    # --- 5. Optional: Geometry vs. Weight Plot ---
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