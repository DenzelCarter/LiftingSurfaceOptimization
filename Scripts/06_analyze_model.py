# Scripts/07_analyze_model.py
# Loads the final trained model and plots the feature importance.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from path_utils import load_cfg

def main():
    C = load_cfg()
    P = C["paths"]

    # --- 1. Load Trained Model ---
    model_path = os.path.join(P["outputs_dir"], "models", "xgboost_model.json")
    if not os.path.exists(model_path):
        raise SystemExit(f"Error: Trained model not found at '{model_path}'")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)
    print("Successfully loaded trained XGBoost model.")

    # --- 2. Get Feature Importance ---
    # The booster object contains the feature names in the correct order
    booster = xgb_model.get_booster()
    feature_names = booster.feature_names
    importances = xgb_model.feature_importances_

    # Create a DataFrame for easy sorting and plotting
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # --- 3. Clean Up Feature Names for Plotting ---
    # Make the labels more readable for the paper
    df_importance['feature'] = df_importance['feature'].str.replace('_', ' ').str.title()
    df_importance['feature'] = df_importance['feature'].str.replace('(Deg)', '(deg)', regex=False)
    df_importance['feature'] = df_importance['feature'].str.replace('Rpm Bin Center', 'RPM')
    df_importance['feature'] = df_importance['feature'].str.replace('Material Pla', 'Material (PLA)')
    df_importance['feature'] = df_importance['feature'].str.replace('Material Plap', 'Material (PLA Poor Finish)')
    df_importance['feature'] = df_importance['feature'].str.replace('Material Petgcf', 'Material (PETG)')


    # --- 4. Create and Save Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.barh(df_importance['feature'], df_importance['importance'], color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_ylabel('Design Parameter')
    ax.set_title('XGBoost Model Feature Importance')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Ensure all labels are visible
    fig.tight_layout()

    # Save the plot
    plot_dir = P["outputs_plots_dir"]
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "feature_importance.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f"\nSaved feature importance plot to: {plot_path}")

if __name__ == "__main__":
    main()