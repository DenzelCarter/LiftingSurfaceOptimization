# Scripts/04_evaluate_models.py
# Evaluates trained models using LOOCV and generates a single PDF
# with side-by-side parity plots for direct comparison.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
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

    features = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'rpm_bin_center']
    target = 'prop_efficiency'

    X = df[features]
    y = df[target]

    # Load the trained models
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgboost_model.json"))

    gpr_model = load(os.path.join(model_dir, "gpr_model.joblib"))

    models = {
        "XGBoost": xgb_model,
        "GPR": gpr_model
    }

    print(f"Evaluating models using Leave-One-Out Cross-Validation on {len(X)} samples.")

    # --- 2. Create Figure for Side-by-Side Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Surrogate Model Performance Evaluation", fontsize=16)

    # --- 3. Perform LOOCV and Generate Parity Plots ---
    loo = LeaveOneOut()

    for ax, (name, model) in zip(axes, models.items()):
        print(f"\n--- Evaluating {name} Model ---")

        # Get predictions for each data point using LOOCV
        predictions = cross_val_predict(model, X, y, cv=loo)

        # Calculate the overall R^2 score
        r2 = r2_score(y, predictions)
        print(f"{name} LOOCV R^2 score: {r2:.4f}")

        # --- Create Parity Plot on the designated subplot ---
        ax.scatter(y, predictions, edgecolors=(0, 0, 0, 0.6), alpha=0.8)

        lims = [
            np.min([y.min(), predictions.min()]) * 0.98,
            np.max([y.max(), predictions.max()]) * 1.02,
        ]
        
        # Add a 1:1 "perfect fit" line
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(f"{name} Model (LOOCV R² = {r2:.3f})")
        ax.set_xlabel("Measured Propeller Efficiency")
        ax.set_ylabel("Predicted Propeller Efficiency")
        ax.grid(True, linestyle='--', alpha=0.5)

    # --- 4. Save the Combined Figure ---
    plot_path = os.path.join(plot_dir, "parity_plots_comparison.pdf")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"\nSaved combined parity plot to: {plot_path}")

if __name__ == "__main__":
    main()