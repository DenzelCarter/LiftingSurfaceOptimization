# Scripts/04_evaluate_models.py
# Creates a clean, vertical, two-panel parity plot figure without a main title.

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

def main():
    C = load_cfg()
    P = C["paths"]

    # --- 1. Load Data and Models ---
    input_path = P["master_parquet"]
    model_dir = os.path.join(P["outputs_dir"], "models")
    plot_dir = P["outputs_plots_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_parquet(input_path)
    
    features = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'rpm_bin_center', 'material']
    target = 'prop_efficiency'
    X_raw = df[features]
    y = df[target]
    X = pd.get_dummies(X_raw, columns=['material'], drop_first=False)

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgboost_model.json"))
    gpr_model = load(os.path.join(model_dir, "gpr_model.joblib"))

    models = {"XGBoost": xgb_model, "GPR": gpr_model}
    
    # --- 2. Create Figure for Vertical Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    # --- REMOVED: fig.suptitle(...) ---

    # --- 3. Perform LOOCV and Plot ---
    loo = LeaveOneOut()
    subplot_labels = ['(a)', '(b)']

    for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
        predictions = cross_val_predict(model, X, y, cv=loo)
        
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        spearman_corr, _ = spearmanr(y, predictions)

        # --- Create Parity Plot ---
        ax.scatter(y, predictions, edgecolors=(0, 0, 0, 0.6), alpha=0.8, s=30)
        lims = [np.min([y.min(), predictions.min()])*0.98, np.max([y.max(), predictions.max()])*1.02]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)
        
        ax.set_title(f"{subplot_labels[i]} {name} Model")
        ax.set_xlabel("Measured Propeller Efficiency")
        ax.set_ylabel("Predicted Propeller Efficiency")
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Add stats box to the plot ---
        stats_text = (
            f"R² = {r2:.4f}\n"
            f"MAE = {mae:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
            f"Spearman's ρ = {spearman_corr:.4f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 4. Save the Combined Figure ---
    plot_path = os.path.join(plot_dir, "parity_plots_comparison.pdf")
    # --- UPDATED: Simpler tight_layout for automatic adjustment ---
    fig.tight_layout() 
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\nSaved combined vertical parity plot to: {plot_path}")
    
    # ... (rest of the script for printing console output)

if __name__ == "__main__":
    main()