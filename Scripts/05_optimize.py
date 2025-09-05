# Scripts/05_optimize.py
# Solves for the optimal geometry and saves performance and improvement scores.

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import differential_evolution
from path_utils import load_cfg

def main():
    C = load_cfg()
    P = C["paths"]
    OPT_CFG = C.get("optimization", {})
    HOVER_CFG = C.get("hover_process", {})
    CRUISE_CFG = C.get("cruise_process", {})

    # --- 1. Load Model, Data, and Config ---
    model_path = os.path.join(P["outputs_models"], "xgboost_dual_model.json")
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(model_path)
    print("Successfully loaded trained dual-output XGBoost model.")

    df_all = pd.read_parquet(P["master_parquet"])
    w_hover_list = OPT_CFG.get("w_hover", [0.5])
    if not isinstance(w_hover_list, list): w_hover_list = [w_hover_list]
    print(f"Optimizing for w_hover values: {w_hover_list}")

    # --- 2. Calculate Baseline Performance ---
    eta_baseline = df_all[df_all['flight_mode'] == 'hover']['performance'].median()
    ld_baseline = df_all[df_all['flight_mode'] == 'cruise']['performance'].median()
    print(f"Calculated baselines: Hover Eff. = {eta_baseline:.3f}, Cruise L/D = {ld_baseline:.3f}")

    # --- 3. Define Optimization Bounds and Parameters ---
    bounds_config = C.get("bounds", {})
    bounds = [
        bounds_config.get('AR', [6.0, 10.0]),
        bounds_config.get('lambda', [0.5, 1.0]),
        bounds_config.get('aoaRoot (deg)', [10.0, 20.0]),
        bounds_config.get('aoaTip (deg)', [3.0, 8.0]),
    ]
    hover_op_point = np.mean(HOVER_CFG.get("rpm_window", [800, 1600]))
    cruise_op_point = np.mean(CRUISE_CFG.get("airspeed_window", [15, 25]))
    materials = df_all['material'].unique()

    # --- 4. Define the Objective Function ---
    def objective_function(params, material, w_hover):
        ar, lam, aoa_r, aoa_t = params
        hover_data = {'AR': [ar], 'lambda': [lam], 'aoaRoot (deg)': [aoa_r], 'aoaTip (deg)': [aoa_t],
                      'material': [material], 'flight_mode': ['hover'], 'op_point': [hover_op_point]}
        cruise_data = {'AR': [ar], 'lambda': [lam], 'aoaRoot (deg)': [aoa_r], 'aoaTip (deg)': [aoa_t],
                       'material': [material], 'flight_mode': ['cruise'], 'op_point': [cruise_op_point]}
        X_pred_raw = pd.concat([pd.DataFrame(hover_data), pd.DataFrame(cruise_data)], ignore_index=True)
        X_pred = pd.get_dummies(X_pred_raw, columns=['material', 'flight_mode'])
        model_cols = xgb_model.get_booster().feature_names
        for col in model_cols:
            if col not in X_pred.columns: X_pred[col] = 0
        X_pred = X_pred[model_cols]
        predictions = xgb_model.predict(X_pred)
        eta_hover_pred, ld_cruise_pred = predictions[0], predictions[1]
        i_hover = (eta_hover_pred - eta_baseline) / eta_baseline
        i_cruise = (ld_cruise_pred - ld_baseline) / ld_baseline
        j_score = (w_hover * i_hover) + ((1 - w_hover) * i_cruise)
        return -j_score

    # --- 5. Run Optimization for Each Weight and Material ---
    print("\n--- Starting Optimization Sweep ---")
    optimal_results = []

    for w_hover in w_hover_list:
        print(f"\n===== Optimizing for w_hover = {w_hover} =====")
        for material in materials:
            if not isinstance(material, str): continue
            print(f"  Material: {material.upper()}...")
            
            result = differential_evolution(
                objective_function, bounds, args=(material, w_hover),
                strategy='best1bin', maxiter=200, popsize=20, tol=0.001, seed=42
            )

            if result.success:
                optimal_params = result.x
                max_j_score = -result.fun
                
                # Predict final performance with optimal params
                hover_data = {'AR': [optimal_params[0]], 'lambda': [optimal_params[1]], 'aoaRoot (deg)': [optimal_params[2]], 
                              'aoaTip (deg)': [optimal_params[3]], 'material': [material], 'flight_mode': ['hover'], 'op_point': [hover_op_point]}
                cruise_data = {'AR': [optimal_params[0]], 'lambda': [optimal_params[1]], 'aoaRoot (deg)': [optimal_params[2]], 
                               'aoaTip (deg)': [optimal_params[3]], 'material': [material], 'flight_mode': ['cruise'], 'op_point': [cruise_op_point]}
                X_final_raw = pd.concat([pd.DataFrame(hover_data), pd.DataFrame(cruise_data)], ignore_index=True)
                X_final = pd.get_dummies(X_final_raw, columns=['material', 'flight_mode'])
                model_cols = xgb_model.get_booster().feature_names
                for col in model_cols:
                    if col not in X_final.columns: X_final[col] = 0
                final_preds = xgb_model.predict(X_final[model_cols])
                
                # *** NEW: Calculate final improvement scores ***
                i_hover_final = (final_preds[0] - eta_baseline) / eta_baseline
                i_cruise_final = (final_preds[1] - ld_baseline) / ld_baseline
                
                optimal_results.append({
                    'w_hover': w_hover,
                    'material': material,
                    'predicted_hover_eff': final_preds[0],
                    'predicted_cruise_LD': final_preds[1],
                    # *** NEW: Add scores to the output dictionary ***
                    'improvement_hover': i_hover_final,
                    'improvement_cruise': i_cruise_final,
                    'objective_score_J': max_j_score,
                    'AR': optimal_params[0],
                    'lambda': optimal_params[1],
                    'aoaRoot (deg)': optimal_params[2],
                    'aoaTip (deg)': optimal_params[3]
                })
                print(f"    Success! Optimal design found with J = {max_j_score:.4f}")

    # --- 6. Save All Optimal Designs to a Single CSV ---
    if optimal_results:
        df_optimal = pd.DataFrame(optimal_results)
        # Reorder columns for clarity in the final CSV
        col_order = [
            'w_hover', 'material', 'AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)',
            'predicted_hover_eff', 'predicted_cruise_LD', 
            'improvement_hover', 'improvement_cruise', 'objective_score_J'
        ]
        df_optimal = df_optimal[col_order].sort_values(by=['material', 'w_hover'])
        
        os.makedirs(P["outputs_tables"], exist_ok=True)
        output_path = os.path.join(P["outputs_tables"], "optimal_designs.csv")
        df_optimal.to_csv(output_path, index=False, float_format="%.4f")
        print(f"\nSaved all optimal designs to: {output_path}")
        print("\n--- Optimal Designs Sweep ---")
        print(df_optimal.to_string(index=False))

if __name__ == "__main__":
    main()