import pandas as pd
import numpy as np
import os

# ===================================================================
# --- 1. SETUP: Define Constants and Parameters ---
# ===================================================================
# --- File Paths ---
DOE_FILE_PATH = 'Experiment/tools/doe_test_plan_01.csv'
RAW_DATA_DIR = 'Experiment/data'
TARE_LOOKUP_FILE = 'Experiment/tools/tare_lookup_01.csv'
BEMT_PREDICTIONS_PATH = 'Experiment/tools/bemt_predictions_final.csv' # <-- ADDED
OUTPUT_FILE_PATH = 'Experiment/tools/master_dataset.parquet'

# --- Physical Constants ---
SPAN = 0.184
R_HUB = 0.046
RHO_AIR = 1.225
MU_AIR = 1.81e-5

# --- Data Cleaning Parameters ---
MIN_THRUST_THRESHOLD = 0.1
MIN_RPM_THRESHOLD = 300
MAX_VIBRATION_THRESHOLD = 3.0
Z_THRESHOLD = 3.0

# --- Column Names ---
RPM_COL = 'Motor Electrical Speed (RPM)'
THRUST_COL = 'Thrust (N)'
TORQUE_COL = 'Torque (N·m)'
VIB_COL = 'Vibration (g)'
ELEC_POWER_COL = 'Electrical Power (W)'
MECH_POWER_COL = 'Mechanical Power (W)'


# ===================================================================
# --- 2. Load and Combine All Available Data ---
# ===================================================================
print(f"Loading DOE plan from '{DOE_FILE_PATH}'...")
try:
    doe_df = pd.read_csv(DOE_FILE_PATH)
except FileNotFoundError:
    print(f"Error: DOE file not found at '{DOE_FILE_PATH}'")
    exit()

all_data_list = []
print("Processing each propeller defined in the plan...")
for index, prop_row in doe_df.iterrows():
    prop_filename = prop_row['filename']
    raw_file_path = os.path.join(RAW_DATA_DIR, prop_filename)

    if not os.path.exists(raw_file_path):
        print(f"   - Skipping '{prop_filename}': Raw data file not found.")
        continue

    try:
        df_single = pd.read_csv(raw_file_path)
        for col_name in doe_df.columns:
            df_single[col_name] = prop_row[col_name]
        all_data_list.append(df_single)
        print(f"   + Loaded '{prop_filename}'")
    except Exception as e:
        print(f"   - Skipping '{prop_filename}': Could not read file. Error: {e}")

if not all_data_list:
    print("No data files were loaded. Exiting.")
    exit()

master_df = pd.concat(all_data_list, ignore_index=True)
print(f"\nSuccessfully combined {len(all_data_list)} files.")


# ===================================================================
# --- 3. Data Cleaning and Tare Correction ---
# ===================================================================
print("\nStarting data cleaning and processing...")
required_cols = [RPM_COL, THRUST_COL, TORQUE_COL, VIB_COL, ELEC_POWER_COL, MECH_POWER_COL]
for col in required_cols:
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
master_df.dropna(subset=required_cols, inplace=True)

print(f"Applying spinning tare correction using '{TARE_LOOKUP_FILE}'...")
try:
    tare_df = pd.read_csv(TARE_LOOKUP_FILE)
    master_df['rpm_bin'] = (master_df[RPM_COL] / 50).round() * 50
    master_df = pd.merge(master_df, tare_df, on='rpm_bin', how='left')
    master_df[['thrust_tare', 'torque_tare']] = master_df[['thrust_tare', 'torque_tare']].fillna(0)
    master_df[THRUST_COL] = master_df[THRUST_COL] - master_df['thrust_tare']
    master_df[TORQUE_COL] = master_df[TORQUE_COL] - master_df['torque_tare']
    print("   - Tare correction applied successfully.")
except FileNotFoundError:
    print(f"   - Warning: Tare lookup file not found. Proceeding without correction.")
except Exception as e:
    print(f"   - Warning: Could not apply tare correction. Error: {e}")

initial_rows = len(master_df)
print("Applying physical threshold filters...")
master_df = master_df[
    (master_df[THRUST_COL] > MIN_THRUST_THRESHOLD) &
    (master_df[RPM_COL] > MIN_RPM_THRESHOLD) &
    (master_df[VIB_COL] < MAX_VIBRATION_THRESHOLD)
].copy()
print(f"   - Removed {initial_rows - len(master_df)} rows due to threshold filters.")


# ===================================================================
# --- 4. Calculate Efficiencies and Engineer Features ---
# ===================================================================
print("Calculating efficiencies...")
master_df['D'] = 2 * (R_HUB + SPAN)
master_df['A'] = np.pi * (master_df['D'] / 2)**2
master_df['ideal_power'] = np.sqrt(master_df[THRUST_COL]**3 / (2 * RHO_AIR * master_df['A']))

master_df['prop_efficiency'] = master_df['ideal_power'] / master_df[MECH_POWER_COL]
master_df['motor_efficiency'] = master_df[MECH_POWER_COL] / master_df[ELEC_POWER_COL]
master_df['system_efficiency'] = master_df['ideal_power'] / master_df[ELEC_POWER_COL]

initial_rows = len(master_df)
master_df = master_df[
    (master_df['prop_efficiency'] > 0) & (master_df['prop_efficiency'] <= 1.0) &
    (master_df['motor_efficiency'] > 0) & (master_df['motor_efficiency'] <= 1.0)
].copy()
master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
master_df.dropna(subset=['prop_efficiency', 'motor_efficiency', 'system_efficiency'], inplace=True)
print(f"   - Removed {initial_rows - len(master_df)} rows with invalid or impossible efficiencies.")

print("Calculating Reynolds Number...")
root_chord = (2 * SPAN) / (master_df['AR'] * (1 + master_df['lambda']))
tip_chord = master_df['lambda'] * root_chord
chord_at_75 = root_chord - 0.75 * (root_chord - tip_chord)
radius_at_75 = R_HUB + (0.75 * SPAN)
velocity_at_75 = (master_df[RPM_COL] * (2 * np.pi / 60)) * radius_at_75
master_df['reynolds_number'] = (RHO_AIR * velocity_at_75 * chord_at_75) / MU_AIR

print("Calculating average blade thickness for NACA 0012 airfoil...")
THICKNESS_CHORD_RATIO = 0.12
average_chord = (root_chord + tip_chord) / 2
master_df['avg_thickness_m'] = THICKNESS_CHORD_RATIO * average_chord


# --- ADD THIS NEW BLOCK TO MERGE BEMT PREDICTIONS ---
print("Merging BEMT predictions...")
try:
    bemt_df = pd.read_csv(BEMT_PREDICTIONS_PATH)
    # Create a merge key by rounding experimental RPM to the nearest 100
    master_df['rpm_bemt'] = (master_df[RPM_COL] / 100).round() * 100
    # Merge the BEMT predictions into the master dataframe
    master_df = pd.merge(master_df, bemt_df, on=['filename', 'rpm_bemt'], how='left')
    # Clean up any rows that didn't have a matching BEMT prediction
    master_df.dropna(subset=['prop_eff_bemt'], inplace=True)
    print("   - BEMT predictions merged successfully.")
except FileNotFoundError:
    print(f"   - Warning: BEMT predictions file not found at '{BEMT_PREDICTIONS_PATH}'.")
except Exception as e:
    print(f"   - Warning: Could not merge BEMT predictions. Error: {e}")
# --- END OF NEW BLOCK ---


# ===================================================================
# --- 5. Remove Statistical Outliers ---
# ===================================================================
print(f"\nRemoving statistical outliers with a Z-score threshold of {Z_THRESHOLD}...")
master_df['z_score_prop'] = master_df.groupby('filename')['prop_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_motor'] = master_df.groupby('filename')['motor_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_sys'] = master_df.groupby('filename')['system_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_vib'] = master_df.groupby('filename')[VIB_COL].transform(lambda x: np.abs((x - x.mean()) / x.std()))

initial_rows = len(master_df)
df_cleaned = master_df[
    (master_df['z_score_prop'] < Z_THRESHOLD) &
    (master_df['z_score_motor'] < Z_THRESHOLD) &
    (master_df['z_score_sys'] < Z_THRESHOLD) &
    (master_df['z_score_vib'] < Z_THRESHOLD)
].copy()
print(f"   - Removed {initial_rows - len(df_cleaned)} statistical outlier data points.")


# ===================================================================
# --- 6. Save Final Cleaned Dataset ---
# ===================================================================
df_cleaned.to_parquet(OUTPUT_FILE_PATH, index=False)
print(f"\nProcess complete. Cleaned master dataset with {len(df_cleaned)} rows saved to '{OUTPUT_FILE_PATH}'")