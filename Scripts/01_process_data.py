# Scripts/01_process_data.py
# Processes experimental hover data and CFD cruise data to create a unified dataset.

import os
import glob
import re
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Global Constants ---
TARE_PATTERN = re.compile(r"^(tare|mount|baseline)\b", re.IGNORECASE)

# --- Configuration Loading ---
def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

# --- Helper Functions ---
def _find_col(df: pd.DataFrame, preferred_names: list, fallbacks: list = None) -> str | None:
    """Finds the first matching column in a DataFrame, case-insensitively."""
    fallbacks = fallbacks or []
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    for name in preferred_names + fallbacks:
        if name.lower().strip() in df_columns_lower:
            return df_columns_lower[name.lower().strip()]
    return None

def _create_tare_interpolator(tare_dir: str):
    """Creates high-resolution tare interpolation functions from all raw tare CSVs."""
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

    print(f"Successfully created tare interpolator from {len(df_clean_tare)} unique data points.")
    return thrust_fn, torque_fn

def main():
    C = load_config()
    P = C["paths"]
    GEO_COLS = C["geometry_cols"]
    HOVER_CFG = C.get("hover_process", {})
    script_dir = Path(__file__).parent

    # --- 1. Load DOE, Constants, and Create Tare Interpolator ---
    doe_csv_path = (script_dir / P["doe_csv"]).resolve()
    try:
        doe_df = pd.read_csv(doe_csv_path)
    except FileNotFoundError:
        raise SystemExit(f"Error: DOE file not found at '{doe_csv_path}'.")

    rho = C["fluids"]["rho"]
    r_tip = C["geometry"]["r_hub_m"] + C["geometry"]["span_blade_m"]
    disk_A = np.pi * r_tip**2

    tare_dir = (script_dir / P["data_hover_tare"]).resolve()
    thrust_tare_fn, torque_tare_fn = _create_tare_interpolator(str(tare_dir))

    all_hover_rows = []
    all_cruise_rows = []

    # ========================================================================
    # --- 2. Process HOVER Data (Group-by-Setpoint Method) ---
    # ========================================================================
    print("\n--- Processing Hover Data ---")
    data_hover_dir = (script_dir / P["data_hover_raw"]).resolve()
    hover_files = glob.glob(os.path.join(data_hover_dir, "*.csv"))
    
    thrust_snr_threshold = HOVER_CFG.get("thrust_snr_threshold", 0)
    torque_snr_threshold = HOVER_CFG.get("torque_snr_threshold", 0)

    if not hover_files:
        print("No hover data files found. Skipping.")
    else:
        for fpath in hover_files:
            basename = os.path.basename(fpath)
            if TARE_PATTERN.search(basename): continue
            try:
                # ... (Hover processing logic is unchanged) ...
                df_raw = pd.read_csv(fpath, low_memory=False)
                esc_col = _find_col(df_raw, ["ESC signal (µs)"])
                rpm_col = _find_col(df_raw, ["Motor Electrical Speed (RPM)"], ["rpm"])
                thrust_col = _find_col(df_raw, ["Thrust (N)"])
                torque_col = _find_col(df_raw, ["Torque (N·m)"])

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
                
                for esc_val, group in grouped:
                    if group.empty: continue
                    
                    n_points = len(group)
                    T_avg, T_std = group['thrust'].mean(), group['thrust'].std()
                    Torque_avg, Torque_std = group['torque'].mean(), group['torque'].std()
                    P_avg = group['mech_power'].mean()
                    
                    thrust_snr = np.abs(T_avg) / T_std if T_std > 1e-9 else np.inf
                    torque_snr = np.abs(Torque_avg) / Torque_std if Torque_std > 1e-9 else np.inf
                    
                    if thrust_snr < thrust_snr_threshold or torque_snr < torque_snr_threshold:
                        continue

                    disk_loading = T_avg / disk_A
                    Pi_avg = np.sqrt(np.maximum(T_avg, 0.0)**3 / (2.0 * rho * disk_A))
                    efficiency = np.clip(Pi_avg / P_avg if P_avg > 1e-9 else 0.0, 0.0, 1.0)
                    
                    all_hover_rows.append({
                        "filename": basename, "flight_mode": "hover",
                        "op_point": disk_loading,
                        "performance": efficiency,
                        "esc_signal_us": esc_val,
                        "rpm_mean": group['rpm'].mean(),
                        "n_points": n_points,
                        "thrust_snr": thrust_snr,
                        "torque_snr": torque_snr
                    })
            except Exception as e:
                print(f"Error processing hover file {basename}: {e}")

    # ========================================================================
    # --- 3. Process CRUISE Data from COMSOL ---
    # ========================================================================
    print("\n--- Processing Cruise Data from COMSOL ---")
    data_cruise_dir = (script_dir / P["data_cruise_comsol"]).resolve()
    cruise_files = glob.glob(os.path.join(data_cruise_dir, "*.txt"))

    if not cruise_files:
        print("No COMSOL cruise data files found. Skipping.")
    else:
        for fpath in cruise_files:
            try:
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                
                first_data_line_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('%'):
                        first_data_line_index = i + 1
                
                col_names = ["AR", "lambda", "aoa_root", "aoa_tip", "lift", "drag"]
                df_comsol = pd.read_csv(fpath, sep='\s+', header=None, names=col_names, skiprows=first_data_line_index)
                
                df_comsol["i_r (deg)"] = df_comsol["aoa_root"]
                df_comsol["epsilon (deg)"] = df_comsol["aoa_tip"] - df_comsol["aoa_root"]

                with np.errstate(divide='ignore', invalid='ignore'):
                    ld_ratio = df_comsol['lift'] / df_comsol['drag']
                ld_ratio.replace([np.inf, -np.inf], 0, inplace=True)
                df_comsol['performance'] = ld_ratio.fillna(0)
                
                for index, row in df_comsol.iterrows():
                    all_cruise_rows.append({
                        "flight_mode": "cruise",
                        "op_point": row["aoa_root"],
                        "performance": row["performance"],
                        "AR": row["AR"],
                        "lambda": row["lambda"],
                        "i_r (deg)": row["i_r (deg)"],
                        "epsilon (deg)": row["epsilon (deg)"]
                    })
            except Exception as e:
                print(f"Error processing COMSOL file {os.path.basename(fpath)}: {e}")

    # ========================================================================
    # --- 4. Finalize and Save Dataset ---
    # ========================================================================
    if not all_hover_rows and not all_cruise_rows:
        print("\nWarning: No data processed. No output file generated.")
        exit()

    # --- MODIFIED: Separate processing paths for hover and cruise ---
    df_hover_processed = pd.DataFrame(all_hover_rows)
    df_cruise_processed = pd.DataFrame(all_cruise_rows)

    # Merge hover data with DOE to get geometry
    df_hover_final = pd.merge(df_hover_processed, doe_df, on="filename", how="inner")
    
    # Concatenate the two complete dataframes
    df_final = pd.concat([df_hover_final, df_cruise_processed], ignore_index=True)
    
    ordered_cols = ['filename', *GEO_COLS, 'flight_mode', 'op_point', 'performance', 
                    'esc_signal_us', 'rpm_mean', 'n_points', 'thrust_snr', 'torque_snr']
    df_final = df_final[[c for c in ordered_cols if c in df_final.columns]]

    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    master_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(master_parquet_path, index=False)

    print(f"\nSuccessfully created master dataset with {len(df_final)} total data points.")
    print(f"Output saved to: '{master_parquet_path}'")

    print("\n--- Final Dataset Snippet ---")
    print(df_final.head(10).to_string(index=False))

    # ========================================================================
    # --- 5. Generate and Save Hover SNR Plots ---
    # ========================================================================
    print("\n--- Generating Hover SNR Plots ---")
    plots_dir = (script_dir / P["outputs_plots"] / "snr").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_hover_plots = df_final[df_final['flight_mode'] == 'hover']
    for filename, prop_df in df_hover_plots.groupby("filename"):
        prop_df = prop_df.sort_values("esc_signal_us")
        if prop_df.empty: continue
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # ... (Plotting logic is unchanged) ...
        fig.suptitle(f'Data Quality Analysis for: {filename}', fontsize=16)
        ax1.plot(prop_df['esc_signal_us'], prop_df['thrust_snr'], 'o-', color='royalblue', label='Thrust SNR')
        ax1.axhline(y=thrust_snr_threshold, color='r', linestyle='--', label=f'Thrust Threshold ({thrust_snr_threshold})')
        ax1.set_ylabel('Thrust SNR (mean/std)')
        ax1.set_title('Thrust Signal-to-Noise Ratio vs. ESC Signal')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        ax2.plot(prop_df['esc_signal_us'], prop_df['torque_snr'], 'o-', color='seagreen', label='Torque SNR')
        ax2.axhline(y=torque_snr_threshold, color='r', linestyle='--', label=f'Torque Threshold ({torque_snr_threshold})')
        ax2.set_ylabel('Torque SNR (mean/std)')
        ax2.set_title('Torque Signal-to-Noise Ratio vs. ESC Signal')
        ax2.set_xlabel('ESC Signal (µs)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        for index, row in prop_df.iterrows():
            ax1.text(row['esc_signal_us'], row['thrust_snr'], f" n={row['n_points']}", ha='left', va='center', fontsize=8, color='gray')
            ax2.text(row['esc_signal_us'], row['torque_snr'], f" n={row['n_points']}", ha='left', va='center', fontsize=8, color='gray')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = f"01_SNR_{os.path.splitext(filename)[0]}.pdf"
        save_path = plots_dir / plot_filename
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Successfully generated {df_hover_plots['filename'].nunique()} SNR plots in '{plots_dir}'")

if __name__ == "__main__":
    main()