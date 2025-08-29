# Scripts/06_train_models.py
# Trains final surrogate models on the ENTIRE dataset.

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from joblib import dump

from path_utils import load_cfg

def main():
    C = load_cfg()
    P = C["paths"]
    
    # --- 1. Load Data and Define Features ---
    input_path = P["master_parquet"]
    if not os.path.exists(input_path):
        raise SystemExit(f"Error: Master dataset not found at '{input_path}'")
        
    print(f"Reading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    features = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'rpm_bin_center','material']
    target = 'prop_efficiency'
    
    X_raw = df[features]
    y = df[target]

    # --- NEW: Use one-hot encoding for the material feature ---
    X = pd.get_dummies(X_raw, columns=['material'], drop_first=False)

    print(f"Training models on the full dataset of {len(X)} samples.")
    print(f"Final features include: {list(X.columns)}")

    # --- 2. Train and Save XGBoost Model ---
    print("\n--- Training final XGBoost Model ---")
    
    xgbr = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, # A reasonable number for the final model
        learning_rate=0.05,
        max_depth=5,
    )
    
    xgbr.fit(X, y)
    
    model_output_dir = os.path.join(P["outputs_dir"], "models")
    os.makedirs(model_output_dir, exist_ok=True)
    xgb_path = os.path.join(model_output_dir, "xgboost_model.json")
    xgbr.save_model(xgb_path)
    print(f"Saved final XGBoost model to: {xgb_path}")

    # --- 3. Train and Save Gaussian Process Regression (GPR) Model ---
    print("\n--- Training final Gaussian Process Model ---")
    
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    
    gpr_pipeline = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    )
    
    gpr_pipeline.fit(X, y)
    
    gpr_path = os.path.join(model_output_dir, "gpr_model.joblib")
    dump(gpr_pipeline, gpr_path)
    print(f"Saved final GPR model to: {gpr_path}")

if __name__ == "__main__":
    main()