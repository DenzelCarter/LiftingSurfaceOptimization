# Scripts/03_train_models.py
# Trains surrogate models using a Matern kernel for the GPR.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
# Note the alias 'C' for ConstantKernel
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import xgboost as xgb
from joblib import dump
from pathlib import Path
import path_utils

def main():
    # --- CORRECTED: Use 'cfg' to avoid name collision with the kernel 'C' ---
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # --- 1. Load Data ---
    master_parquet_path = Path(P["master_parquet"])
    if not master_parquet_path.exists():
        raise SystemExit(f"Master dataset not found. Please run 01_process_data.py first.")
    df_full = pd.read_parquet(master_parquet_path)

    df_hover = df_full[df_full['flight_mode'] == 'hover']
    df_cruise = df_full[df_full['flight_mode'] == 'cruise']

    # Define features for the models based on config.yaml
    features = cfg["geometry_cols"] + ["op_speed"]

    # --- 2. Train Hover Models ---
    print("\n--- Training Hover Models ---")
    X_hover, y_hover = df_hover[features], df_hover['performance']
    
    # Define a robust Matern kernel for the GPR
    # Now that 'C' refers to ConstantKernel again, this will work
    matern_kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
    
    gpr_hover_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=15, random_state=42))
    ])
    gpr_hover_pipeline.fit(X_hover, y_hover)
    
    xgb_hover = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_hover.fit(X_hover, y_hover)

    # --- 3. Train Cruise Models ---
    print("--- Training Cruise Models ---")
    X_cruise, y_cruise = df_cruise[features], df_cruise['performance']
    
    gpr_cruise_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=15, random_state=42))
    ])
    gpr_cruise_pipeline.fit(X_cruise, y_cruise)

    xgb_cruise = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_cruise.fit(X_cruise, y_cruise)
    
    # --- 4. Save Models ---
    models_dir = Path(P["outputs_models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    dump(gpr_hover_pipeline, models_dir / "gpr_hover_model.joblib")
    xgb_hover.save_model(models_dir / "xgboost_hover_model.json")
    dump(gpr_cruise_pipeline, models_dir / "gpr_cruise_model.joblib")
    xgb_cruise.save_model(models_dir / "xgboost_cruise_model.json")
    
    print(f"\nSuccessfully trained and saved models to: {models_dir}")
    print("Run 04_evaluate_models.py to see performance metrics.")

if __name__ == "__main__":
    main()