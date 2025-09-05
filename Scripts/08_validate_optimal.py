# Scripts/07_validate_optimal.py
# Processes the experimental data for the optimal propeller and compares
# its measured performance to the surrogate model's prediction.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from path_utils import load_cfg

# --- USER: Set the filename of your new experimental test ---
OPTIMAL_CSV_FILENAME = "raw/LS_051_01.csv"

# --- HELPER FUNCTIONS (Copied from 05_process_data.py to solve import error) ---
def _find_col(df: pd.DataFrame, preferred, fallbacks=()):
    """Finds the first matching column in a dataframe."""
    all_cols = list(df.columns)
    candidates = list(preferred) + list(fallbacks)
    for cand in candidates:
        if cand in all_cols:
            return cand
    return None

def _resolve_rpm(df: pd.DataFrame):
    """Resolves the RPM column, prioritizing electrical RPM."""
    rpm_col = _find_col(df, ["Motor Electrical Speed (RPM)", "rpm"])
    if rpm_col:
        return pd.to_numeric(df[rpm_col], errors="coerce")
    rps_col = _find_col(df, ["rot (1/s)", "rotation (1/s)", "rps"])
    if rps_col:
        return pd.to_numeric(df[rps_col], errors="coerce") * 60.0
    return pd.Series(np.nan, index=df.index)

def _calculate_binned_efficiency(rpm, T, Pm, rho, disk_A, window, n_bins):
    """Calculates actuator disk efficiency, binned over an RPM window."""
    Pi = np.sqrt(np.maximum(T, 0.0)**3 / (2.0 * rho * disk_A))
    eta_row = np.where(Pm > 1e-9, Pi / np.maximum(Pm, 1e-9), np.nan)
    eta_row = np.clip(eta_row, 0.0, 1.0)
    lo, hi = (float(window[0]), float(window[1])) if window else (rpm.min(), rpm.max())
    valid_mask = (rpm >= lo) & (rpm <= hi) & np.isfinite(rpm) & np.isfinite(eta_row)
    if not valid_mask.any(): return []
    edges = np.linspace(lo, hi, n_bins + 1) if n_bins > 1 else [lo, hi]
    bin_centers = (edges[:-1] + edges[1:]) / 2
    binned_rows = []
    for i, center in enumerate(bin_centers):
        bin_mask = valid_mask & (rpm >= edges[i]) & (rpm <= edges[i+1])
        if bin_mask.any():
            binned_rows.append({
                "rpm_bin_center": center,
                "prop_efficiency_mean": float(np.nanmean(eta_row[bin_mask])),
            })
    return binned_rows
# --- END HELPER FUNCTIONS ---

def main():
    C = load_cfg()
    P = C["paths"]
    PROC_CFG = C.get("process", {})

    # --- 1. Load Trained Model and Optimal Design Parameters ---
    model_path = os.path.join(P["outputs_dir"], "models", "xgboost_model.json")
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(model_path)
    print("Successfully loaded trained XGBoost model.")

    # --- THIS IS WHERE THE GEOMETRY IS LOADED ---
    # It reads the results from your optimization, so no need to define it again.
    optimal_designs_path = os.path.join(P["outputs_tables_dir"], "optimal_designs.csv")
    df_optimal = pd.read_csv(optimal_designs_path)
    
    # We will validate against the first optimal design found (e.g., for PLA)
    optimal_geo = df_optimal.iloc[0]
    print("\nValidating against optimal geometry loaded from CSV:")
    print(optimal_geo)

    # --- 2. Process the New Experimental Data ---
    exp_data_path = os.path.join(P["data_bench_dir"], OPTIMAL_CSV_FILENAME)
    if not os.path.exists(exp_data_path):
        raise SystemExit(f"Error: Experimental data not found at '{exp_data_path}'")
    
    df_raw = pd.read_csv(exp_data_path, low_memory=False)
    
    rpm = _resolve_rpm(df_raw)
    thrust = pd.to_numeric(df_raw[_find_col(df_raw, ["Thrust (N)"])], errors="coerce")
    mech_power = pd.to_numeric(df_raw[_find_col(df_raw, ["Mechanical Power (W)"])], errors="coerce")
    
    r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
    disk_A = np.pi * r_tip**2
    
    measured_points = _calculate_binned_efficiency(
        rpm, thrust, mech_power, C["fluids"]["rho"], disk_A, 
        PROC_CFG.get("rpm_window"), int(PROC_CFG.get("rpm_n_bins"))
    )
    df_measured = pd.DataFrame(measured_points)
    print("\nProcessed Measured Data:")
    print(df_measured)

    # --- 3. Predict Efficiency with the Surrogate Model ---
    X_pred_raw = pd.DataFrame({
        'AR': optimal_geo['AR'], 'lambda': optimal_geo['lambda'],
        'aoaRoot (deg)': optimal_geo['aoaRoot (deg)'], 'aoaTip (deg)': optimal_geo['aoaTip (deg)'],
        'rpm_bin_center': df_measured['rpm_bin_center'], 'material': optimal_geo['material']
    })
    
    X_pred = pd.get_dummies(X_pred_raw, columns=['material'])
    for col in xgb_model.get_booster().feature_names:
        if col not in X_pred.columns: X_pred[col] = 0
    X_pred = X_pred[xgb_model.get_booster().feature_names]
    
    predicted_efficiency = xgb_model.predict(X_pred)
    df_measured['predicted_efficiency'] = predicted_efficiency
    
    # --- 4. Quantify Error and Plot ---
    mae = mean_absolute_error(df_measured['prop_efficiency_mean'], df_measured['predicted_efficiency'])
    print(f"\nFinal Validation MAE: {mae:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_measured['rpm_bin_center'], df_measured['predicted_efficiency'], 
            label=f'Surrogate Prediction (MAE={mae:.3f})', color='dodgerblue', marker='None', linestyle='--', zorder=2)
    ax.plot(df_measured['rpm_bin_center'], df_measured['prop_efficiency_mean'], 
            label='Measured Performance', color='crimson', marker='o', markersize=8, linestyle='None', zorder=3)

    ax.set_xlabel("RPM"); ax.set_ylabel("Propeller Efficiency")
    ax.set_title("Validation of Optimal Propeller Design"); ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.set_ylim(bottom=0)
    fig.tight_layout()
    
    plot_path = os.path.join(P["outputs_plots_dir"], "optimal_design_validation.pdf")
    fig.savefig(plot_path); plt.close(fig)
    print(f"\nSaved validation plot to: {plot_path}")

if __name__ == "__main__":
    main()