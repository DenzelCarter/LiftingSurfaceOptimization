# Scripts/03_train_models.py
# Trains separate, dedicated surrogate models for hover and cruise performance.

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

    # --- 1. Load Unified Dataset ---
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    if not master_parquet_path.exists():
        raise SystemExit(f"Error: Master dataset not found at '{master_parquet_path}'")
        
    print(f"Reading unified dataset from: {master_parquet_path}")
    df_full = pd.read_parquet(master_parquet_path)
    
    # Create the directory for saving models
    models_dir = (script_dir / P["outputs_models"]).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # --- 2. Train HOVER Surrogate Models ---
    # ========================================================================
    print("\n--- Training Hover Surrogate Models ---")
    df_hover = df_full[df_full['flight_mode'] == 'hover'].copy()
    
    if df_hover.empty:
        print("No hover data found. Skipping hover model training.")
    else:
        # Define the features for the hover model
        hover_features = [*GEO_COLS, 'op_point'] # op_point is Disk Loading
        hover_target = 'performance' # performance is Hover Efficiency
        
        X_hover = df_hover[hover_features]
        y_hover = df_hover[hover_target]

        print(f"Training on {len(X_hover)} hover data points.")
        print(f"Hover features: {list(X_hover.columns)}")

        # Train and Save XGBoost Model for Hover
        xgbr_hover = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, seed=42)
        xgbr_hover.fit(X_hover, y_hover)
        xgb_hover_path = models_dir / "xgboost_hover_model.json"
        xgbr_hover.save_model(xgb_hover_path)
        print(f"Saved hover XGBoost model to: {xgb_hover_path}")

        # Train and Save GPR Model for Hover
        gpr_hover_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        gpr_hover_pipeline = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=gpr_hover_kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-10))
        gpr_hover_pipeline.fit(X_hover, y_hover)
        gpr_hover_path = models_dir / "gpr_hover_model.joblib"
        dump(gpr_hover_pipeline, gpr_hover_path)
        print(f"Saved hover GPR model to: {gpr_hover_path}")

    # ========================================================================
    # --- 3. Train CRUISE Surrogate Models ---
    # ========================================================================
    print("\n--- Training Cruise Surrogate Models ---")
    df_cruise = df_full[df_full['flight_mode'] == 'cruise'].copy()

    if df_cruise.empty:
        print("No cruise data found. Skipping cruise model training.")
    else:
        # Define the features for the cruise model
        cruise_features = [*GEO_COLS, 'op_point'] # op_point is Root AoA
        cruise_target = 'performance' # performance is L/D Ratio
        
        X_cruise = df_cruise[cruise_features]
        y_cruise = df_cruise[cruise_target]

        print(f"Training on {len(X_cruise)} cruise data points.")
        print(f"Cruise features: {list(X_cruise.columns)}")

        # Train and Save XGBoost Model for Cruise
        xgbr_cruise = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, seed=42)
        xgbr_cruise.fit(X_cruise, y_cruise)
        xgb_cruise_path = models_dir / "xgboost_cruise_model.json"
        xgbr_cruise.save_model(xgb_cruise_path)
        print(f"Saved cruise XGBoost model to: {xgb_cruise_path}")

        # Train and Save GPR Model for Cruise
        gpr_cruise_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        gpr_cruise_pipeline = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=gpr_cruise_kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-10))
        gpr_cruise_pipeline.fit(X_cruise, y_cruise)
        gpr_cruise_path = models_dir / "gpr_cruise_model.joblib"
        dump(gpr_cruise_pipeline, gpr_cruise_path)
        print(f"Saved cruise GPR model to: {gpr_cruise_path}")

if __name__ == "__main__":
    main()