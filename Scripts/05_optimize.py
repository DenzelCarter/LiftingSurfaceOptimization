# Scripts/05_optimize.py
# Finds the optimal propeller design and saves the results to a CSV file.

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import differential_evolution
from path_utils import load_cfg

def main():
    C = load_cfg()
    P = C["paths"]
    PROC_CFG = C.get("process", {})

    # --- 1. Load Model and Data ---
    model_path = os.path.join(P["outputs_dir"], "models", "xgboost_model.json")
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(model_path)
    print("Successfully loaded trained XGBoost model.")

    df_doe = pd.read_csv(P["doe_csv"])

    # --- 2. Define Design Space ---
    bounds = [
        (df_doe['AR'].min(), df_doe['AR'].max()),
        (df_doe['lambda'].min(), df_doe['lambda'].max()),
        (df_doe['aoaRoot (deg)'].min(), df_doe['aoaRoot (deg)'].max()),
        (df_doe['aoaTip (deg)'].min(), df_doe['aoaTip (deg)'].max()),
    ]
    rpm_window = PROC_CFG.get("rpm_window", [800, 1600])
    optim_rpm = np.mean(rpm_window)
    materials = df_doe['material'].unique()
    
    # --- 3. Run Optimization ---
    print("\n--- Starting Optimization ---")
    optimal_results = []

    for material in materials:
        print(f"\nOptimizing for material: {material.upper()}")

        def objective_function(params):
            ar, lam, aoa_r, aoa_t = params
            data = {'AR': [ar], 'lambda': [lam], 'aoaRoot (deg)': [aoa_r], 
                    'aoaTip (deg)': [aoa_t], 'rpm_bin_center': [optim_rpm], 'material': [material]}
            X_pred_raw = pd.DataFrame(data)
            X_pred = pd.get_dummies(X_pred_raw, columns=['material'])
            for col in xgb_model.get_booster().feature_names:
                if col not in X_pred.columns: X_pred[col] = 0
            X_pred = X_pred[xgb_model.get_booster().feature_names]
            prediction = xgb_model.predict(X_pred)
            return -prediction[0]

        result = differential_evolution(objective_function, bounds, strategy='best1bin', 
                                        maxiter=200, popsize=20, tol=0.001, seed=42)

        if result.success:
            optimal_params = result.x
            predicted_efficiency = -result.fun
            
            # Append results to the list
            optimal_results.append({
                'material': material,
                'predicted_efficiency': predicted_efficiency,
                'AR': optimal_params[0],
                'lambda': optimal_params[1],
                'aoaRoot (deg)': optimal_params[2],
                'aoaTip (deg)': optimal_params[3],
                'optim_rpm': optim_rpm
            })
            print(f"  Optimization Successful! Predicted Efficiency: {predicted_efficiency:.4f}")
        else:
            print(f"  Optimization for {material.upper()} did not converge.")

    # --- 4. Save Optimal Designs to CSV ---
    if optimal_results:
        df_optimal = pd.DataFrame(optimal_results)
        output_dir = P.get("outputs_tables_dir", os.path.join(P["outputs_dir"], "tables"))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "optimal_designs.csv")
        df_optimal.to_csv(output_path, index=False, float_format="%.4f")
        print(f"\nSaved optimal designs to: {output_path}")
        print("\n--- Optimal Designs ---")
        print(df_optimal.to_string(index=False))

if __name__ == "__main__":
    main()