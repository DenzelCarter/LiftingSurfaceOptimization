# Scripts/04_evaluate_models.py
# Evaluates models trained on the new 'op_speed' feature set and generates
# parity and feature importance plots with updated labels.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
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

def evaluate_and_plot(df_mode, models, mode_name, unit_name, features, C):
    """
    Performs LOOCV evaluation and generates parity plots for a specific flight mode.
    """
    P = C["paths"]
    script_dir = Path(__file__).parent
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    
    print(f"\n--- Evaluating Models for {mode_name.upper()} Performance ---")
    
    X = df_mode[features]
    y = df_mode['performance']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    loo = LeaveOneOut()
    all_metrics = []

    for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
        predictions = cross_val_predict(model, X, y, cv=loo)
        
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        spearman_corr, _ = spearmanr(y, predictions)
        
        metrics = {'Flight Mode': mode_name, 'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'Spearman': spearman_corr}
        all_metrics.append(metrics)

        ax.scatter(y, predictions, edgecolors=(0, 0, 0, 0.6), alpha=0.8, s=30)
        lims = [np.min([y.min(), predictions.min()])*0.98, np.max([y.max(), predictions.max()])*1.02]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)
        
        ax.set_title(f"{name} Model")
        ax.set_xlabel(f"Measured {unit_name}")
        ax.set_ylabel(f"Predicted {unit_name} (LOOCV)")
        ax.grid(True, linestyle='--', alpha=0.5)

        stats_text = f"R² = {r2:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f"{mode_name} Performance Parity Plots (LOOCV)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = plots_dir / f"04_parity_plots_{mode_name.lower()}.pdf"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {mode_name} parity plot to: {plot_path}")
    
    return all_metrics

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    script_dir = Path(__file__).parent

    # --- 1. Load Data and Trained Models ---
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    df_full = pd.read_parquet(master_parquet_path)
    
    models_dir = (script_dir / P["outputs_models"]).resolve()
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    try:
        xgb_hover = xgb.XGBRegressor(); xgb_hover.load_model(models_dir / "xgboost_hover_model.json")
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        models_hover = {"XGBoost": xgb_hover, "GPR": gpr_hover}
        
        xgb_cruise = xgb.XGBRegressor(); xgb_cruise.load_model(models_dir / "xgboost_cruise_model.json")
        gpr_cruise = load(models_dir / "gpr_cruise_model.joblib")
        models_cruise = {"XGBoost": xgb_cruise, "GPR": gpr_cruise}
    except Exception as e:
        raise SystemExit(f"Error loading models from '{models_dir}'. Have you run 03_train_models.py yet? Details: {e}")

    # --- 2. Evaluate Models ---
    df_hover = df_full[df_full['flight_mode'] == 'hover'].copy()
    hover_features = [*GEO_COLS, 'op_speed']
    hover_metrics = evaluate_and_plot(df_hover, models_hover, "Hover", "Efficiency (η)", hover_features, C)

    df_cruise = df_full[df_full['flight_mode'] == 'cruise'].copy()
    cruise_features = [*GEO_COLS, 'op_speed']
    cruise_metrics = evaluate_and_plot(df_cruise, models_cruise, "Cruise", "L/D Ratio", cruise_features, C)

    # --- 3. Display and Save Combined Metrics ---
    df_metrics = pd.DataFrame(hover_metrics + cruise_metrics)
    print("\n\n--- Combined Model Performance (LOOCV) ---")
    print(df_metrics.to_string(index=False))
    
    metrics_path = tables_dir / "04_model_performance.csv"
    df_metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"\nSaved performance metrics to: {metrics_path}")

    # --- 4. Generate and Save Feature Importance Plots with Custom Labels ---
    
    # --- MODIFIED: Update label maps for the new 'op_speed' feature ---
    hover_label_map = {
        'AR': 'AR',
        'lambda': r'$\lambda$',
        'aoa_root (deg)': 'Root AoA',
        'twist (deg)': 'Twist',
        'op_speed': 'RPM'
    }
    
    cruise_label_map = {
        'AR': 'AR',
        'lambda': r'$\lambda$',
        'aoa_root (deg)': 'Root AoA',
        'twist (deg)': 'Twist',
        'op_speed': 'Cruise Speed'
    }

    # Hover Importance Plot
    fig_hover_imp, ax = plt.subplots(figsize=(10, 6))
    importances = xgb_hover.feature_importances_
    sorted_idx = importances.argsort()
    display_labels = [hover_label_map[feat] for feat in np.array(hover_features)[sorted_idx]]
    ax.barh(display_labels, importances[sorted_idx], color='skyblue')
    ax.set_xlabel("XGBoost Feature Importance")
    ax.set_title("Hover Model Feature Importance")
    fig_hover_imp.tight_layout()
    hover_imp_path = plots_dir / "04_feature_importance_hover.pdf"
    fig_hover_imp.savefig(hover_imp_path)
    plt.close(fig_hover_imp)
    print(f"Saved hover feature importance plot.")

    # Cruise Importance Plot
    fig_cruise_imp, ax = plt.subplots(figsize=(10, 6))
    importances = xgb_cruise.feature_importances_
    sorted_idx = importances.argsort()
    display_labels = [cruise_label_map[feat] for feat in np.array(cruise_features)[sorted_idx]]
    ax.barh(display_labels, importances[sorted_idx], color='seagreen')
    ax.set_xlabel("XGBoost Feature Importance")
    ax.set_title("Cruise Model Feature Importance")
    fig_cruise_imp.tight_layout()
    cruise_imp_path = plots_dir / "04_feature_importance_cruise.pdf"
    fig_cruise_imp.savefig(cruise_imp_path)
    plt.close(fig_cruise_imp)
    print(f"Saved cruise feature importance plot.")

if __name__ == "__main__":
    main()