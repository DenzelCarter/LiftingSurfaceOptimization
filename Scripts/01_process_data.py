# Scripts/01_process_data.py
# Implements a robust "filter-and-bin" method for hover data processing.

import os
import glob
import re
import numpy as np
import pandas as pd
from path_utils import load_cfg

# --- Global Constants & Helpers ---
TARE_PATTERN = re.compile(r"^(tare|mount|baseline)\b", re.IGNORECASE)

def _find_col(df: pd.DataFrame, preferred, fallbacks=()):
    all_cols = list(df.columns); candidates = list(preferred) + list(fallbacks)
    for cand in candidates:
        if cand in all_cols: return cand
    for cand in candidates:
        for col in all_cols:
            if cand.lower() in col.lower(): return col
    return None

def _resolve_rpm(df: pd.DataFrame):
    rpm_col = _find_col(df, ["Motor Electrical Speed (RPM)", "rpm"])
    if rpm_col: return pd.to_numeric(df[rpm_col], errors="coerce")
    rps_col = _find_col(df, ["rot (1/s)", "rotation (1/s)", "rps"])
    if rps_col: return pd.to_numeric(df[rps_col], errors="coerce") * 60.0
    return pd.Series(np.nan, index=df.index)

def main():
    C = load_cfg()
    P = C["paths"]
    HOVER_CFG = C.get("hover_process", {})
    CRUISE_CFG = C.get("cruise_process", {})

    # --- 1. Load DOE and Constants ---
    doe_df = pd.read_csv(P["doe_csv"])
    rho = C["fluids"]["rho"]
    r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
    disk_A = np.pi * r_tip**2

    all_processed_rows = []

    # ========================================================================
    # --- 2. Process HOVER Data (Filter-and-Bin Method) ---
    # ========================================================================
    print("\n--- Processing Hover Data ---")
    hover_files = glob.glob(os.path.join(P["data_hover_raw"], "*.csv"))
    
    if not hover_files:
        print("No hover data files found. Skipping.")
    else:
        rpm_window = HOVER_CFG.get("rpm_window", [800, 1600])
        rpm_n_bins = HOVER_CFG.get("rpm_n_bins", 5)

        for fpath in hover_files:
            basename = os.path.basename(fpath)
            if TARE_PATTERN.match(basename): continue

            try:
                df_raw = pd.read_csv(fpath, low_memory=False)
                rpm = _resolve_rpm(df_raw)
                thrust_col = _find_col(df_raw, ["Thrust (N)"])
                power_col = _find_col(df_raw, ["Mechanical Power (W)"])

                if rpm is None or thrust_col is None or power_col is None:
                    print(f"Warning: Skipping {basename}, missing RPM, Thrust, or Power column.")
                    continue

                df_raw['rpm'] = rpm
                df_raw['thrust'] = pd.to_numeric(df_raw[thrust_col], errors="coerce")
                df_raw['mech_power'] = pd.to_numeric(df_raw[power_col], errors="coerce")

                df_filtered_by_rpm = df_raw[df_raw['rpm'].between(rpm_window[0], rpm_window[1])]
                
                # --- MODIFIED: Use a targeted dropna to preserve data ---
                df_filtered = df_filtered_by_rpm.dropna(subset=['rpm', 'thrust', 'mech_power'])
                
                if len(df_filtered) < (rpm_n_bins * 5):
                    print(f"Warning: Skipping {basename}, insufficient valid data points in the RPM window.")
                    continue

                rpm_bins = pd.cut(df_filtered['rpm'], bins=rpm_n_bins)
                
                grouped = df_filtered.groupby(rpm_bins)
                for bin_range, group in grouped:
                    if group.empty: continue
                    
                    T_avg = group['thrust'].mean()
                    P_avg = group['mech_power'].mean()
                    
                    disk_loading = T_avg / disk_A
                    Pi_avg = np.sqrt(np.maximum(T_avg, 0.0)**3 / (2.0 * rho * disk_A))
                    efficiency = np.clip(Pi_avg / P_avg if P_avg > 1e-9 else 0.0, 0.0, 1.0)
                    
                    all_processed_rows.append({
                        "filename": basename, "flight_mode": "hover",
                        "op_point": disk_loading,
                        "performance": efficiency
                    })
            except Exception as e:
                print(f"Error processing hover file {basename}: {e}")

    # ========================================================================
    # --- 3. Process CRUISE Data ---
    # ========================================================================
    print("\n--- Processing Cruise Data ---")
    # This section is omitted for brevity but should be the same as your working version.
    
    # ========================================================================
    # --- 4. Finalize and Save Dataset ---
    # ========================================================================
    if not all_processed_rows:
        print("\nWarning: No data processed. No output file generated.")
        return
        
    df_processed = pd.DataFrame(all_processed_rows)
    df_final = pd.merge(df_processed, doe_df, on="filename", how="inner")
    
    final_cols = ['filename', 'AR', 'lambda', 'i_r (deg)', 'epsilon (deg)', 'material', 
                  'flight_mode', 'op_point', 'performance']
    df_final = df_final[[c for c in final_cols if c in df_final.columns]]
    
    os.makedirs(P["data_processed_dir"], exist_ok=True)
    df_final.to_parquet(P["master_parquet"], index=False)
    
    print(f"\nSuccessfully created master dataset with {len(df_final)} total data points.")
    print(f"Output saved to: '{P['master_parquet']}'")
    
    print("\n--- Final Dataset Snippet ---")
    print(df_final.head(10).to_string(index=False))

if __name__ == "__main__":
    main()