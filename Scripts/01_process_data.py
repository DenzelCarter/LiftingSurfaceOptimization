# Scripts/01_process_data.py
# Processes data and filters it to a common design space based on the truly
# fixed geometric parameters (AR, lambda, twist) to prevent extrapolation.

import os
import glob
import re
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# --- Global Constants & Config ---
TARE_PATTERN = re.compile(r"^(tare|mount|baseline)\b", re.IGNORECASE)

def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

# --- Helper Functions (unchanged) ---
def _find_col(df: pd.DataFrame, preferred_names: list, fallbacks: list = None) -> str | None:
    fallbacks = fallbacks or []
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    for name in preferred_names + fallbacks:
        if name.lower().strip() in df_columns_lower:
            return df_columns_lower[name.lower().strip()]
    return None

def _create_tare_interpolator(tare_dir: str):
    if not os.path.isdir(tare_dir):
        warnings.warn(f"Tare directory not found at '{tare_dir}'. No tare correction will be applied.")
        return lambda rpm: np.zeros_like(rpm), lambda rpm: np.zeros_like(rpm)
    tare_files = glob.glob(os.path.join(tare_dir, "*.csv"))
    if not tare_files:
        warnings.warn("No raw tare files found. No tare correction will be applied.")
        return lambda rpm: np.zeros_like(rpm), lambda rpm: np.zeros_like(rpm)
    tare_dfs = [pd.read_csv(f, low_memory=False) for f in tare_files]
    df_raw_tare = pd.concat(tare_dfs, ignore_index=True)
    rpm_col = _find_col(df_raw_tare, ["Motor Electrical Speed (RPM)"], ["rpm", "RPM"])
    thrust_col = _find_col(df_raw_tare, ["Thrust (N)"])
    torque_col = _find_col(df_raw_tare, ["Torque (N·m)"])
    if not all([rpm_col, thrust_col, torque_col]):
        warnings.warn("Tare files are missing required columns. No tare correction applied.")
        return lambda rpm: np.zeros_like(rpm), lambda rpm: np.zeros_like(rpm)
    df_clean_tare = pd.DataFrame({
        'rpm': pd.to_numeric(df_raw_tare[rpm_col], errors="coerce"),
        'thrust': pd.to_numeric(df_raw_tare[thrust_col], errors="coerce"),
        'torque': pd.to_numeric(df_raw_tare[torque_col], errors="coerce")
    }).dropna().sort_values('rpm').drop_duplicates(subset=['rpm'])
    if len(df_clean_tare) < 2:
        warnings.warn("Insufficient valid data in tare files for interpolation. No correction applied.")
        return lambda rpm: np.zeros_like(rpm), lambda rpm: np.zeros_like(rpm)
    thrust_fn = interp1d(df_clean_tare["rpm"], df_clean_tare["thrust"], kind='linear', bounds_error=False, fill_value=0)
    torque_fn = interp1d(df_clean_tare["rpm"], df_clean_tare["torque"], kind='linear', bounds_error=False, fill_value=0)
    print("Successfully created tare interpolator.")
    return thrust_fn, torque_fn

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    HOVER_CFG = C.get("hover_process", {})
    script_dir = Path(__file__).parent

    # --- 1. Load DOE, Constants, and Tare Interpolator ---
    doe_csv_path = (script_dir / P["doe_csv"]).resolve()
    try: doe_df = pd.read_csv(doe_csv_path)
    except FileNotFoundError: raise SystemExit(f"Error: DOE file not found at '{doe_csv_path}'.")
    rho = C["fluids"]["rho"]
    r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
    disk_A = np.pi * r_tip**2
    tare_dir = (script_dir / P["data_hover_tare"]).resolve()
    thrust_tare_fn, torque_tare_fn = _create_tare_interpolator(str(tare_dir))
    all_hover_rows, all_cruise_rows = [], []

    # --- 2. Process HOVER Data ---
    print("\n--- Processing Hover Data ---")
    data_hover_dir = (script_dir / P["data_hover_raw"]).resolve()
    hover_files = glob.glob(os.path.join(data_hover_dir, "*.csv"))
    if hover_files:
        for fpath in hover_files:
            basename = os.path.basename(fpath)
            if TARE_PATTERN.search(basename): continue
            try:
                df_raw = pd.read_csv(fpath, low_memory=False)
                esc_col, rpm_col = _find_col(df_raw, ["ESC signal (µs)"]), _find_col(df_raw, ["Motor Electrical Speed (RPM)"], ["rpm"])
                thrust_col, torque_col = _find_col(df_raw, ["Thrust (N)"]), _find_col(df_raw, ["Torque (N·m)"])
                if not all([esc_col, rpm_col, thrust_col, torque_col]): continue
                df_proc = pd.DataFrame()
                df_proc['esc_signal_us'] = pd.to_numeric(df_raw[esc_col], errors="coerce")
                df_proc['rpm'] = pd.to_numeric(df_raw[rpm_col], errors="coerce")
                df_proc['thrust_raw'] = pd.to_numeric(df_raw[thrust_col], errors="coerce")
                df_proc['torque_raw'] = pd.to_numeric(df_raw[torque_col], errors="coerce")
                df_proc.dropna(inplace=True)
                if df_proc.empty: continue
                df_proc = df_proc[df_proc['esc_signal_us'] != 1000]
                if df_proc.empty: continue
                df_proc['thrust'] = df_proc['thrust_raw'] - thrust_tare_fn(df_proc['rpm'])
                df_proc['torque'] = df_proc['torque_raw'] - torque_tare_fn(df_proc['rpm'])
                df_proc['mech_power'] = df_proc['torque'] * (2 * np.pi * df_proc['rpm'] / 60.0)
                grouped = df_proc.groupby('esc_signal_us')
                for _, group in grouped:
                    if group.empty: continue
                    T_avg, T_std = group['thrust'].mean(), group['thrust'].std()
                    Torque_avg, Torque_std = group['torque'].mean(), group['torque'].std()
                    thrust_snr = np.abs(T_avg) / T_std if T_std > 1e-9 else np.inf
                    torque_snr = np.abs(Torque_avg) / Torque_std if Torque_std > 1e-9 else np.inf
                    if thrust_snr < HOVER_CFG.get("thrust_snr_threshold", 0) or torque_snr < HOVER_CFG.get("torque_snr_threshold", 0): continue
                    P_avg = group['mech_power'].mean()
                    Pi_avg = np.sqrt(np.maximum(T_avg, 0.0)**3 / (2.0 * rho * disk_A))
                    efficiency = np.clip(Pi_avg / P_avg if P_avg > 1e-9 else 0.0, 0.0, 1.0)
                    all_hover_rows.append({"filename": basename, "flight_mode": "hover", "op_speed": group['rpm'].mean(), "performance": efficiency})
            except Exception as e:
                print(f"Error processing hover file {basename}: {e}")

    # --- 3. Process CRUISE Data ---
    print("\n--- Processing Cruise Data from COMSOL ---")
    data_cruise_dir = (script_dir / P["data_cruise_comsol"]).resolve()
    cruise_files = glob.glob(os.path.join(data_cruise_dir, "*.txt"))
    if cruise_files:
        for fpath in cruise_files:
            basename = os.path.basename(fpath)
            try:
                with open(fpath, 'r') as f: lines = f.readlines()
                first_data_line = next((i for i, line in enumerate(lines) if not line.strip().startswith('%')), 0)
                col_names = ["AR", "lambda", "aoa_root_raw", "aoa_tip_raw", "lift", "drag"]
                df_comsol = pd.read_csv(fpath, sep='\s+', header=None, names=col_names, skiprows=first_data_line)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ld_ratio = (df_comsol['lift'] / df_comsol['drag']).replace([np.inf, -np.inf], 0).fillna(0)
                for index, row in df_comsol.iterrows():
                    all_cruise_rows.append({
                        "flight_mode": "cruise", "op_speed": 20.0, "performance": ld_ratio.iloc[index],
                        "AR": row["AR"], "lambda": row["lambda"], "aoa_root (deg)": row["aoa_root_raw"],
                        "twist (deg)": row["aoa_tip_raw"] - row["aoa_root_raw"]
                    })
            except Exception as e:
                print(f"Error processing COMSOL file {basename}: {e}")

    # --- 4. Assemble and Filter to Common Design Space ---
    if not all_hover_rows or not all_cruise_rows:
        raise SystemExit("\nError: Data from both hover and cruise modes is required. No output file generated.")

    df_hover = pd.merge(pd.DataFrame(all_hover_rows), doe_df, on="filename", how="inner")
    df_cruise = pd.DataFrame(all_cruise_rows)

    # --- MODIFIED: Define fixed geometric parameters for bounding ---
    fixed_geometric_params = ['AR', 'lambda', 'twist (deg)']
    
    print("\n--- Filtering to a conservative, overlapping design space for FIXED GEOMETRY ---")
    conservative_bounds = {}
    for col in fixed_geometric_params:
        min_h, max_h = df_hover[col].min(), df_hover[col].max()
        min_c, max_c = df_cruise[col].min(), df_cruise[col].max()
        
        final_min = max(min_h, min_c)
        final_max = min(max_h, max_c)
        
        if final_min >= final_max:
            raise ValueError(f"No overlapping design space for fixed parameter '{col}'. Please check your DOEs.")
            
        conservative_bounds[col] = (final_min, final_max)
        print(f"'{col}' bounds constrained to: [{final_min:.2f}, {final_max:.2f}]")

    # Filter both dataframes using the new conservative bounds for fixed geometry
    for col, (min_val, max_val) in conservative_bounds.items():
        df_hover = df_hover[df_hover[col].between(min_val, max_val)]
        df_cruise = df_cruise[df_cruise[col].between(min_val, max_val)]

    # --- 5. Finalize and Save Dataset ---
    df_final = pd.concat([df_hover, df_cruise], ignore_index=True)
    if df_final.empty:
        raise SystemExit("Error: No data remained after filtering to the common design space. Please check your DOEs.")
        
    final_cols = ['filename', *GEO_COLS, 'flight_mode', 'op_speed', 'performance']
    df_final = df_final[[c for c in final_cols if c in df_final.columns]]

    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    master_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(master_parquet_path, index=False)

    print(f"\nSuccessfully created master dataset with {len(df_final)} filtered data points.")
    print(f"Output saved to: '{master_parquet_path}'")

if __name__ == "__main__":
    main()