# Scripts/04_evaluate_models.py
# Evaluates hover surrogate models using LOOCV, generates parity plots and feature importance plots.

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

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    script_dir = Path(__file__).parent

    # --- 1. Load Data and Trained Models ---
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    if not master_parquet_path.exists():
        raise SystemExit(f"Error: Master dataset not found at '{master_parquet_path}'")
        
    df_full = pd.read_parquet(master_parquet_path)
    
    # Filter for hover data only
    df_hover = df_full[df_full['flight_mode'] == 'hover'].copy()
    if df_hover.empty:
        raise SystemExit("Error: No hover data found in the master dataset.")

    # Load pre-trained hover models
    models_dir = (script_dir / P["outputs_models"]).resolve()
    try:
        xgb_hover = xgb.XGBRegressor()
        xgb_hover.load_model(models_dir / "xgboost_hover_model.json")
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        models = {"XGBoost": xgb_hover, "GPR": gpr_hover}
    except Exception as e:
        raise SystemExit(f"Error loading models from '{models_dir}'. Have you run 03_train_models.py yet? Details: {e}")

    # --- 2. Prepare Data for Evaluation ---
    features = [*GEO_COLS, 'op_point']
    target = 'performance'
    
    X = df_hover[features]
    y = df_hover[target]

    # --- 3. Evaluate Models using Leave-One-Out Cross-Validation (LOOCV) ---
    print(f"\n--- Evaluating Hover Models on {len(X)} data points using LOOCV ---")
    loo = LeaveOneOut()
    all_metrics = []

    for name, model in models.items():
        predictions = cross_val_predict(model, X, y, cv=loo)
        
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        spearman_corr, _ = spearmanr(y, predictions)
        
        metrics = {'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'Spearman': spearman_corr}
        all_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(all_metrics)
    print("\n--- Hover Model Performance (LOOCV) ---")
    print(df_metrics.to_string(index=False))

    # Save metrics to a CSV table
    tables_dir = (script_dir / P["outputs_tables"]).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "04_hover_model_performance.csv"
    df_metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"\nSaved performance metrics to: {metrics_path}")

    # --- 4. Generate and Save Parity Plots ---
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig_parity, axes = plt.subplots(1, 2, figsize=(12, 6))
    subplot_labels = ['(a)', '(b)']

    for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
        predictions = cross_val_predict(model, X, y, cv=loo)
        
        ax.scatter(y, predictions, edgecolors=(0, 0, 0, 0.6), alpha=0.8, s=30)
        lims = [np.min([y.min(), predictions.min()])*0.98, np.max([y.max(), predictions.max()])*1.02]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)
        
        ax.set_title(f"{subplot_labels[i]} {name} Model")
        ax.set_xlabel("Measured Hover Efficiency")
        ax.set_ylabel("Predicted Hover Efficiency (LOOCV)")
        ax.grid(True, linestyle='--', alpha=0.5)

        r2 = r2_score(y, predictions)
        stats_text = f"R² = {r2:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_parity.suptitle("Hover Performance Parity Plots (LOOCV)", fontsize=16)
    fig_parity.tight_layout(rect=[0, 0, 1, 0.96])
    
    parity_plot_path = plots_dir / "04_hover_parity_plots.pdf"
    fig_parity.savefig(parity_plot_path)
    plt.close(fig_parity)
    print(f"Saved hover parity plot to: {parity_plot_path}")

    # --- 5. Generate and Save Feature Importance Plot ---
    fig_importance, ax = plt.subplots(figsize=(10, 6))
    
    importances = xgb_hover.feature_importances_
    sorted_idx = importances.argsort()
    
    ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color='skyblue')
    ax.set_xlabel("XGBoost Feature Importance")
    ax.set_title("Hover Model Feature Importance")
    
    fig_importance.tight_layout()
    
    importance_plot_path = plots_dir / "04_hover_feature_importance.pdf"
    fig_importance.savefig(importance_plot_path)
    plt.close(fig_importance)
    print(f"Saved feature importance plot to: {importance_plot_path}")


if __name__ == "__main__":
    main()