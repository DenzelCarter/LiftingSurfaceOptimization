# Scripts/04_evaluate_models.py
# Evaluates dual-output models using LOOCV and generates separate parity plots for hover and cruise.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from joblib import load
import xgboost as xgb
from path_utils import load_cfg

def evaluate_and_plot(df_mode, models, mode_name, unit_name, plot_dir):
    """
    Performs LOOCV evaluation and generates parity plots for a specific flight mode.

    Returns:
        A list of dictionaries containing the performance metrics for each model.
    """
    print(f"\n--- Evaluating Models for {mode_name.upper()} Performance ---")
    
    # --- 1. Prepare Data for this Flight Mode ---
    features = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'material', 'flight_mode', 'op_point']
    target = 'performance'
    
    X_raw = df_mode[features]
    y = df_mode[target]
    X = pd.get_dummies(X_raw, columns=['material', 'flight_mode'])

    # Ensure all possible columns are present, filling missing with 0
    all_cols = X.columns
    for model_name, model in models.items():
        try:
            model_cols = model.get_booster().feature_names
            for col in model_cols:
                if col not in all_cols:
                    X[col] = 0
            X = X[model_cols]
            break # Assume all models were trained on the same feature set
        except (AttributeError, ValueError): # GPR doesn't have get_booster
            continue

    # --- 2. Create Figure and Evaluate Models ---
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    if not isinstance(axes, np.ndarray): axes = [axes] # Handle case of 1 model
    
    loo = LeaveOneOut()
    subplot_labels = ['(a)', '(b)']
    all_metrics = []

    for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
        predictions = cross_val_predict(model, X, y, cv=loo)
        
        # Calculate metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        spearman_corr, _ = spearmanr(y, predictions)
        
        metrics = {
            'Flight Mode': mode_name, 'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'Spearman': spearman_corr
        }
        all_metrics.append(metrics)
        print(pd.DataFrame([metrics]).to_string(index=False))

        # Create Parity Plot
        ax.scatter(y, predictions, edgecolors=(0, 0, 0, 0.6), alpha=0.8, s=30)
        lims = [np.min([y.min(), predictions.min()])*0.98, np.max([y.max(), predictions.max()])*1.02]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)
        
        ax.set_title(f"{subplot_labels[i]} {name} Model - {mode_name} Performance")
        ax.set_xlabel(f"Measured {unit_name}")
        ax.set_ylabel(f"Predicted {unit_name} (LOOCV)")
        ax.grid(True, linestyle='--', alpha=0.5)

        stats_text = (f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nSpearman's ρ = {spearman_corr:.4f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 3. Save the Figure ---
    plot_path = os.path.join(plot_dir, f"parity_plots_{mode_name.lower()}.pdf")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\nSaved {mode_name} parity plot to: {plot_path}")
    
    return all_metrics

def main():
    C = load_cfg()
    P = C["paths"]

    # --- 1. Load Data and Models ---
    input_path = P["master_parquet"]
    os.makedirs(P["outputs_plots"], exist_ok=True)

    df_all = pd.read_parquet(input_path)
    
    # Load models
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(P["outputs_models"], "xgboost_dual_model.json"))
    gpr_model = load(os.path.join(P["outputs_models"], "gpr_dual_model.joblib"))
    models = {"XGBoost": xgb_model, "GPR": gpr_model}
    
    # --- 2. Split Data by Flight Mode ---
    df_hover = df_all[df_all['flight_mode'] == 'hover'].copy()
    df_cruise = df_all[df_all['flight_mode'] == 'cruise'].copy()

    # --- 3. Evaluate and Plot for Each Mode ---
    hover_metrics = evaluate_and_plot(
        df_mode=df_hover, models=models, mode_name="Hover", 
        unit_name="Efficiency", plot_dir=P["outputs_plots"]
    )
    
    cruise_metrics = evaluate_and_plot(
        df_mode=df_cruise, models=models, mode_name="Cruise", 
        unit_name="L/D Ratio", plot_dir=P["outputs_plots"]
    )

    # --- 4. Display Combined Metrics Table ---
    df_metrics = pd.DataFrame(hover_metrics + cruise_metrics)
    print("\n\n--- Combined Model Performance (LOOCV) ---")
    print(df_metrics.to_string(index=False))
    
    # Save metrics to a CSV for the paper
    metrics_path = os.path.join(P["outputs_tables"], "model_loocv_performance.csv")
    df_metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"\nSaved performance metrics to: {metrics_path}")

if __name__ == "__main__":
    main()