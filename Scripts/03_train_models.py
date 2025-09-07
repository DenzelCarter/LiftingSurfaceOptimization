# Scripts/03_train_models.py
# Trains surrogate models for hover performance.

import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from joblib import dump
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

    # --- 1. Load Data and Define Features for Hover Model ---
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    if not master_parquet_path.exists():
        raise SystemExit(f"Error: Master dataset not found at '{master_parquet_path}'")
        
    print(f"Reading dataset from: {master_parquet_path}")
    df_full = pd.read_parquet(master_parquet_path)
    
    # Filter for hover data only
    df = df_full[df_full['flight_mode'] == 'hover'].copy()
    if df.empty:
        raise SystemExit("Error: No hover data found in the master dataset.")

    # Define the features the hover model will learn from.
    # 'op_point' (disk loading) is now a critical input feature.
    features = [*GEO_COLS, 'op_point']
    target = 'performance' # This is the hover efficiency (eta_hover)
    
    X = df[features]
    y = df[target]

    print(f"\nTraining hover-only surrogate models on {len(X)} data points.")
    print(f"Features for the model: {list(X.columns)}")

    # --- 2. Train and Save XGBoost Model for Hover ---
    print("\n--- Training Hover XGBoost Model ---")
    
    xgbr_hover = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        seed=42
    )
    
    xgbr_hover.fit(X, y)
    
    models_dir = (script_dir / P["outputs_models"]).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    xgb_path = models_dir / "xgboost_hover_model.json"
    xgbr_hover.save_model(xgb_path)
    print(f"Saved hover XGBoost model to: {xgb_path}")

    # --- 3. Train and Save Gaussian Process Regression (GPR) Model for Hover ---
    print("\n--- Training Hover Gaussian Process Model ---")
    
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    
    gpr_pipeline_hover = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-10)
    )
    
    gpr_pipeline_hover.fit(X, y)
    
    gpr_path = models_dir / "gpr_hover_model.joblib"
    dump(gpr_pipeline_hover, gpr_path)
    print(f"Saved hover GPR model to: {gpr_path}")

    # ========================================================================
    # --- FUTURE: CRUISE MODEL TRAINING ---
    # ========================================================================
    # print("\n--- FUTURE: Training Cruise Surrogate Models ---")
    #
    # # 1. Filter for cruise data
    # df_cruise = df_full[df_full['flight_mode'] == 'cruise'].copy()
    #
    # # 2. Define cruise features (e.g., AoA might be a feature)
    # cruise_features = [*GEO_COLS, 'angle_of_attack']
    # cruise_target = 'performance' # This would be L/D
    #
    # X_cruise = df_cruise[cruise_features]
    # y_cruise = df_cruise[cruise_target]
    #
    # # 3. Train and save your cruise models (XGBoost, GPR) here
    # # ...
    #
    # ========================================================================

if __name__ == "__main__":
    main()