# Scripts/06_analyze_model.py
# Loads the final dual-output model and plots its global feature importance.

import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from path_utils import load_cfg

def main():
    C = load_cfg()
    P = C["paths"]

    # --- 1. Load the Trained Dual-Output Model ---
    model_path = os.path.join(P["outputs_models"], "xgboost_dual_model.json")
    if not os.path.exists(model_path):
        raise SystemExit(f"Error: Trained model not found at '{model_path}'")

    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(model_path)
    print("Successfully loaded trained dual-output XGBoost model.")

    # --- 2. Get Feature Importance ---
    booster = xgb_model.get_booster()
    feature_names = booster.feature_names
    importances = xgb_model.feature_importances_

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # --- 3. Clean Up Feature Names for Plotting ---
    # This section is updated for the new dual-output feature set
    feature_map = {
        'aoaRoot (deg)': 'Root AoA (deg)',
        'aoaTip (deg)': 'Tip AoA (deg)',
        'op_point': 'Operational Point (RPM/Airspeed)',
        'flight_mode_hover': 'Flight Mode (Hover)',
        'flight_mode_cruise': 'Flight Mode (Cruise)',
        'material_PLA': 'Material (PLA)',
        'material_PLAP': 'Material (PLA Poor Finish)',
        'material_PETGCF': 'Material (PETG-CF)'
    }
    # Use .get(x, x.title()) to apply the map or just title-case the original name
    df_importance['feature_clean'] = [feature_map.get(f, f.title()) for f in df_importance['feature']]
    
    # --- 4. Create and Save Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.barh(df_importance['feature_clean'], df_importance['importance'], color='cornflowerblue', edgecolor='black')
    
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_ylabel('Model Feature')
    ax.set_title('Dual-Output Surrogate Model Feature Importance')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig.tight_layout()

    os.makedirs(P["outputs_plots"], exist_ok=True)
    plot_path = os.path.join(P["outputs_plots"], "feature_importance_dual_model.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f"\nSaved feature importance plot to: {plot_path}")
    print("\n--- Feature Importances ---")
    print(df_importance[['feature', 'importance']].sort_values('importance', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()