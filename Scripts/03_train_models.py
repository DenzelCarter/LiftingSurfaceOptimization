# Scripts/03_train_models.py
# Trains a single, dual-output surrogate model on the entire unified dataset.

import os
import pandas as pd
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
    
    # --- 1. Load Data and Define Features for Dual-Mode Model ---
    input_path = P["master_parquet"]
    if not os.path.exists(input_path):
        raise SystemExit(f"Error: Master dataset not found at '{input_path}'")
        
    print(f"Reading unified dataset from: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Define the features the model will learn from.
    # 'flight_mode' and 'op_point' are now critical input features.
    features = [
        'AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 
        'material', 'flight_mode', 'op_point'
    ]
    target = 'performance' # This column contains both hover efficiency and cruise L/D
    
    X_raw = df[features]
    y = df[target]

    # Use one-hot encoding for the categorical features: 'material' and 'flight_mode'
    X = pd.get_dummies(X_raw, columns=['material', 'flight_mode'], drop_first=False)

    print(f"\nTraining dual-output models on {len(X)} total data points.")
    print(f"Final features for the model: {list(X.columns)}")

    # --- 2. Train and Save XGBoost Model ---
    print("\n--- Training final XGBoost Model ---")
    
    xgbr = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        enable_categorical=True, # Good practice for XGBoost
        seed=42
    )
    
    xgbr.fit(X, y)
    
    os.makedirs(P["outputs_models"], exist_ok=True)
    xgb_path = os.path.join(P["outputs_models"], "xgboost_dual_model.json")
    xgbr.save_model(xgb_path)
    print(f"Saved final dual-output XGBoost model to: {xgb_path}")

    # --- 3. Train and Save Gaussian Process Regression (GPR) Model ---
    print("\n--- Training final Gaussian Process Model ---")
    
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    
    gpr_pipeline = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    )
    
    gpr_pipeline.fit(X, y)
    
    gpr_path = os.path.join(P["outputs_models"], "gpr_dual_model.joblib")
    dump(gpr_pipeline, gpr_path)
    print(f"Saved final dual-output GPR model to: {gpr_path}")

if __name__ == "__main__":
    main()