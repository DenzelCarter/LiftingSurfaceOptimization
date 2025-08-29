# 00_calculate_snr.py
# Calculates performance and accurate SNR by grouping data by steady-state ESC signal.

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

def main():
    C = load_cfg()
    P = C["paths"]
    # FIXED: Load rho (air density) from the configuration here
    rho = C["fluids"]["rho"]
    
    # --- 1. Create Tare Interpolator ---
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
        
        # Find the columns for the raw signals
        rpm_col = _find_col(df_raw, ["Motor Electrical Speed (RPM)"])
        thrust_col = _find_col(df_raw, ["Thrust (N)"])
        torque_col = _find_col(df_raw, ["Torque (N·m)"])
        esc_col = _find_col(df_raw, ["ESC signal (µs)"])

        if not all([rpm_col, thrust_col, torque_col, esc_col]):
            print(f"Warning: Skipping {basename} due to missing required columns.")
            continue

        # Prepare raw signals as numeric arrays
        rpm = pd.to_numeric(df_raw[rpm_col], errors='coerce')
        thrust_raw = pd.to_numeric(df_raw[thrust_col], errors='coerce')
        torque_raw = pd.to_numeric(df_raw[torque_col], errors='coerce')
        esc_signal = pd.to_numeric(df_raw[esc_col], errors='coerce')

        # Apply tare correction
        thrust_tare, torque_tare = tare_interpolator(rpm)
        thrust_corr = thrust_raw - thrust_tare
        torque_corr = torque_raw - torque_tare

        # Group by each steady-state ESC signal
        df_raw['rpm'] = rpm
        df_raw['thrust_corr'] = thrust_corr
        df_raw['torque_corr'] = torque_corr
        
        for esc_step, group in df_raw.groupby(esc_col):
            # Calculate mean values for this step
            T_avg = group['thrust_corr'].mean()
            Torque_avg = group['torque_corr'].mean()
            rpm_avg = group['rpm'].mean()
            
            # Calculate standard deviations for this step
            T_std = group['thrust_corr'].std()
            Torque_std = group['torque_corr'].std()
            rpm_std = group['rpm'].std()
            
            # Calculate true SNR
            T_snr = T_avg / T_std if T_std > 1e-9 else np.inf
            Torque_snr = Torque_avg / Torque_std if Torque_std > 1e-9 else np.inf
            rpm_snr = rpm_avg / rpm_std if rpm_std > 1e-9 else np.inf
            
            # Calculate efficiency from the stable averages
            r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
            disk_A = np.pi * r_tip**2
            omega_avg = 2 * np.pi * rpm_avg / 60.0
            Pm_avg = Torque_avg * omega_avg
            Pi_avg = np.sqrt(np.maximum(T_avg, 0.0)**3 / (2.0 * rho * disk_A))
            efficiency = Pi_avg / Pm_avg if Pm_avg > 1e-9 else 0.0

            processed_data.append({
                "filename": basename,
                "esc_signal_us": esc_step,
                "rpm_mean": rpm_avg,
                "prop_efficiency": np.clip(efficiency, 0.0, 1.0),
                "thrust_snr": T_snr,
                "torque_snr": Torque_snr,
                "rpm_snr": rpm_snr,
            })

    if not processed_data:
        raise SystemExit("No processable propeller data was found. Aborting.")

    # --- 4. Combine with DOE and Save ---
    df_processed = pd.DataFrame(processed_data)
    df_final = pd.merge(df_processed, doe_df, on="filename", how="inner")
    
    final_cols = ["filename", *GEO_COLS, "esc_signal_us", "rpm_mean", "prop_efficiency", "thrust_snr", "torque_snr", "rpm_snr"]
    df_final = df_final[[c for c in final_cols if c in df_final.columns]]

    df_final.to_parquet(P["master_parquet"], index=False)
    print(f"\nSuccessfully created master dataset with {len(df_final)} rows from {df_final['filename'].nunique()} propellers.")
    print(f"Output saved to: '{P['master_parquet']}'")
    
    # --- 5. Print Detailed Performance Summary to Console ---
    print("\n--- Propeller Performance & Stability per ESC Step ---")
    summary_df = df_final.rename(columns={
        "filename": "Propeller", "esc_signal_us": "ESC Signal", "rpm_mean": "RPM",
        "prop_efficiency": "Efficiency", "thrust_snr": "Thrust SNR", "torque_snr": "Torque SNR", "rpm_snr": "RPM SNR"
    }).sort_values(by=["Propeller", "ESC Signal"])
    
    with pd.option_context('display.max_rows', None, 'display.precision', 3, 'display.float_format', '{:.2f}'.format):
        print(summary_df[[
            "Propeller", "ESC Signal", "RPM", "Efficiency", "Thrust SNR", "Torque SNR", "RPM SNR"
        ]])

if __name__ == "__main__":
    main()