# 05_process_data.py
# Final version for ML data prep. Uses RPM window/binning for a uniform dataset.

import os
import glob
import re
import numpy as np
import pandas as pd
from path_utils import load_cfg

# --- Global Constants ---
GEO_COLS = ["AR", "lambda", "aoaRoot (deg)", "aoaTip (deg)"]
TARE_PATTERN = re.compile(r"^(tare|mount|baseline)\b", re.IGNORECASE)

# --- Helper Functions ---
def _find_col(df: pd.DataFrame, preferred, fallbacks=()):
    """Finds the first matching column in a dataframe."""
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    for cand in list(preferred) + list(fallbacks):
        cand_low = cand.lower()
        for i, col_low in enumerate(low):
            if cand_low == col_low:
                return cols[i]
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

def _create_tare_interpolator_from_raw(tare_dir):
    """Creates a high-resolution tare interpolator directly from raw tare CSVs."""
    tare_files = glob.glob(os.path.join(tare_dir, "*.csv"))
    if not tare_files:
        print("Warning: No raw tare files found. No tare correction will be applied.")
        return lambda rpm: (np.zeros_like(rpm), np.zeros_like(rpm))

    df_list = [pd.read_csv(f, low_memory=False) for f in tare_files]
    df_raw_tare = pd.concat(df_list, ignore_index=True)

    rpm = _resolve_rpm(df_raw_tare)
    thrust = pd.to_numeric(df_raw_tare[_find_col(df_raw_tare, ["Thrust (N)"])], errors="coerce")
    torque = pd.to_numeric(df_raw_tare[_find_col(df_raw_tare, ["Torque (N·m)"])], errors="coerce")

    df_clean_tare = pd.DataFrame({'rpm': rpm, 'thrust': thrust, 'torque': torque}).dropna().sort_values('rpm')

    if len(df_clean_tare) < 2:
        print("Warning: Insufficient valid data in tare files. No correction applied.")
        return lambda rpm: (np.zeros_like(rpm), np.zeros_like(rpm))

    def interpolator(rpm_values):
        rpm_values = np.asarray(rpm_values)
        thrust_tare = np.interp(rpm_values, df_clean_tare["rpm"], df_clean_tare["thrust"])
        torque_tare = np.interp(rpm_values, df_clean_tare["rpm"], df_clean_tare["torque"])
        return thrust_tare, torque_tare
    
    print(f"Successfully created tare interpolator from {len(df_clean_tare)} raw data points.")
    return interpolator

def calculate_binned_performance(rpm, T_corr, Torque_corr, rho, disk_A, window, n_bins):
    """Averages raw signals within bins first, then calculates a single efficiency value."""
    lo, hi = (float(window[0]), float(window[1])) if window else (rpm.min(), rpm.max())
    valid_mask = (rpm >= lo) & (rpm <= hi) & np.isfinite(rpm) & np.isfinite(T_corr) & np.isfinite(Torque_corr)

    if not valid_mask.any():
        return []

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    
    binned_rows = []
    for i, center in enumerate(bin_centers):
        bin_mask = valid_mask & (rpm >= edges[i]) & (rpm <= edges[i+1])
        if bin_mask.any():
            T_avg = np.nanmean(T_corr[bin_mask])
            Torque_avg = np.nanmean(Torque_corr[bin_mask])
            rpm_avg = np.nanmean(rpm[bin_mask])

            omega_avg = 2 * np.pi * rpm_avg / 60.0
            Pm_avg = Torque_avg * omega_avg
            Pi_avg = np.sqrt(np.maximum(T_avg, 0.0)**3 / (2.0 * rho * disk_A))
            efficiency = Pi_avg / Pm_avg if Pm_avg > 1e-9 else 0.0
            
            binned_rows.append({
                "rpm_bin_center": center,
                "prop_efficiency": np.clip(efficiency, 0.0, 1.0)
            })
    return binned_rows

def main():
    C = load_cfg()
    P = C["paths"]
    PROC_CFG = C.get("process", {})
    rho = C["fluids"]["rho"]

    # --- 1. Load Config and Create Tare Interpolator ---
    rpm_window = PROC_CFG.get("rpm_window")
    
    # --- NEW: Calculate n_bins from rpm_bin_width ---
    rpm_bin_width = PROC_CFG.get("rpm_bin_width")
    if rpm_window and rpm_bin_width:
        window_range = rpm_window[1] - rpm_window[0]
        n_bins = max(1, round(window_range / rpm_bin_width))
        print(f"RPM bin width of {rpm_bin_width} in a {window_range} RPM window results in {n_bins} bins.")
    else:
        # Fallback to the old method if rpm_bin_width isn't defined
        n_bins = int(PROC_CFG.get("rpm_n_bins", 1))

    tare_dir = P.get("data_tare_dir", os.path.join(P["data_bench_dir"], "tare"))
    
    tare_interpolator = _create_tare_interpolator_from_raw(tare_dir)

    # --- 2. Load DOE File ---
    doe_df = pd.read_csv(P["doe_csv"])
    if "filename" not in doe_df.columns:
        raise SystemExit("Error: 'filename' column is required in the DOE CSV.")
    
    # --- 3. Process Raw Data Files ---
    all_files = glob.glob(os.path.join(P["data_bench_dir"], "**", "*.csv"), recursive=True)
    processed_data = []

    for fpath in all_files:
        basename = os.path.basename(fpath)
        if TARE_PATTERN.match(basename):
            continue

        df_raw = pd.read_csv(fpath, low_memory=False)
        
        rpm = _resolve_rpm(df_raw)
        thrust_raw = pd.to_numeric(df_raw[_find_col(df_raw, ["Thrust (N)"])], errors="coerce")
        torque_raw = pd.to_numeric(df_raw[_find_col(df_raw, ["Torque (N·m)"])], errors="coerce")
        
        thrust_tare, torque_tare = tare_interpolator(rpm)
        thrust_corr = thrust_raw - thrust_tare
        torque_corr = torque_raw - torque_tare

        r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
        disk_A = np.pi * r_tip**2
        
        binned_results = calculate_binned_performance(
            rpm, thrust_corr, torque_corr, rho, disk_A, rpm_window, n_bins
        )
        
        for row in binned_results:
            row["filename"] = basename
            processed_data.append(row)

    if not processed_data:
        raise SystemExit("No processable propeller data was found. Aborting.")

    # --- 4. Combine with DOE and Save ---
    df_processed = pd.DataFrame(processed_data)
    df_final = pd.merge(df_processed, doe_df, on="filename", how="inner")
    
    final_cols = ["filename", *GEO_COLS, "rpm_bin_center", "prop_efficiency"]
    df_final = df_final[[c for c in final_cols if c in df_final.columns]]

    df_final.to_parquet(P["master_parquet"], index=False)
    print(f"\nSuccessfully created master dataset with {len(df_final)} rows from {df_final['filename'].nunique()} propellers.")
    print(f"Output saved to: '{P['master_parquet']}'")
    
    # --- 5. Print Final Performance Summary to Console ---
    print("\n--- Final Binned Propeller Performance for ML ---")
    summary_df = df_final.rename(
        columns={"filename": "Propeller", "rpm_bin_center": "RPM Bin", "prop_efficiency": "Efficiency"}
    ).sort_values(by=["Propeller", "RPM Bin"])
    
    with pd.option_context('display.max_rows', None, 'display.precision', 3, 'display.float_format', '{:.3f}'.format):
        print(summary_df)

if __name__ == "__main__":
    main()