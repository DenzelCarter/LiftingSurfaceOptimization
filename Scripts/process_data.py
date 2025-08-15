import pandas as pd
import numpy as np
import os

# ===================================================================
# --- 1. SETUP: Define Constants and Parameters ---
# ===================================================================
# --- File Paths ---
DOE_FILE_PATH = 'Experiment/doe_test_plan_01.csv'
RAW_DATA_DIR = 'Experiment/raw_data'
TARE_LOOKUP_FILE = 'Experiment/tare_lookup.csv'
OUTPUT_FILE_PATH = 'Experiment/master_dataset.parquet'

# --- Physical Constants ---
R_HUB = 0.040
RHO = 1.225

# --- Data Cleaning Parameters ---
MIN_THRUST_THRESHOLD = 0.1      # N
MIN_RPM_THRESHOLD = 500         # RPM
MAX_VIBRATION_THRESHOLD = 3.0   # g
Z_THRESHOLD = 3.0               # Z-score for statistical outlier removal

# --- Column Names (ensure these match your CSV headers) ---
RPM_COL = 'Motor Electrical Speed (RPM)' # Using optical as ground truth
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
        print(f"  - Skipping '{prop_filename}': Raw data file not found.")
        continue

    try:
        df_single = pd.read_csv(raw_file_path)
        for col_name in doe_df.columns:
            df_single[col_name] = prop_row[col_name]
        all_data_list.append(df_single)
        print(f"  + Loaded '{prop_filename}'")
    except Exception as e:
        print(f"  - Skipping '{prop_filename}': Could not read file. Error: {e}")

if not all_data_list:
    print("No data files were loaded. Exiting.")
    exit()

master_df = pd.concat(all_data_list, ignore_index=True)
print(f"\nSuccessfully combined {len(all_data_list)} files.")


# ===================================================================
# --- 3. Data Cleaning and Tare Correction ---
# ===================================================================
print("\nStarting data cleaning and processing...")
# --- Initial Conversion and Cleanup ---
required_cols = [RPM_COL, THRUST_COL, TORQUE_COL, VIB_COL, ELEC_POWER_COL, MECH_POWER_COL]
for col in required_cols:
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
master_df.dropna(subset=required_cols, inplace=True)

# --- Apply Spinning Tare Correction ---
print(f"Applying spinning tare correction using '{TARE_LOOKUP_FILE}'...")
try:
    tare_df = pd.read_csv(TARE_LOOKUP_FILE)
    master_df['rpm_bin'] = (master_df[RPM_COL] / 50).round() * 50
    master_df = pd.merge(master_df, tare_df, on='rpm_bin', how='left')
    master_df[['thrust_tare', 'torque_tare']] = master_df[['thrust_tare', 'torque_tare']].fillna(0)
    master_df[THRUST_COL] = master_df[THRUST_COL] - master_df['thrust_tare']
    master_df[TORQUE_COL] = master_df[TORQUE_COL] - master_df['torque_tare']
    print("  - Tare correction applied successfully.")
except FileNotFoundError:
    print(f"  - Warning: Tare lookup file not found. Proceeding without correction.")
except Exception as e:
    print(f"  - Warning: Could not apply tare correction. Error: {e}")

# --- Apply Physical and Threshold-Based Filters ---
initial_rows = len(master_df)
print("Applying physical threshold filters...")
master_df = master_df[
    (master_df[THRUST_COL] > MIN_THRUST_THRESHOLD) &
    (master_df[RPM_COL] > MIN_RPM_THRESHOLD) &
    (master_df[VIB_COL] < MAX_VIBRATION_THRESHOLD) # Added absolute vibration limit
].copy()
print(f"  - Removed {initial_rows - len(master_df)} rows due to threshold filters.")


# ===================================================================
# --- 4. Calculate Efficiencies ---
# ===================================================================
print("Calculating efficiencies...")
# --- Calculate Ideal Power and Core Efficiencies ---
master_df['D'] = 2 * (R_HUB + master_df['span (m)'])
master_df['A'] = np.pi * (master_df['D'] / 2)**2
master_df['ideal_power'] = np.sqrt(master_df[THRUST_COL]**3 / (2 * RHO * master_df['A']))

master_df['prop_efficiency'] = master_df['ideal_power'] / master_df[MECH_POWER_COL]
master_df['motor_efficiency'] = master_df[MECH_POWER_COL] / master_df[ELEC_POWER_COL]
master_df['system_efficiency'] = master_df['ideal_power'] / master_df[ELEC_POWER_COL]

# --- Filter out physically impossible results ---
initial_rows = len(master_df)
master_df = master_df[
    (master_df['prop_efficiency'] > 0) & (master_df['prop_efficiency'] <= 1.0) &
    (master_df['motor_efficiency'] > 0) & (master_df['motor_efficiency'] <= 1.0)
].copy()
master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
master_df.dropna(subset=['prop_efficiency', 'motor_efficiency', 'system_efficiency'], inplace=True)
print(f"  - Removed {initial_rows - len(master_df)} rows with invalid or impossible efficiencies.")


# ===================================================================
# --- 5. Remove Statistical Outliers ---
# ===================================================================
print(f"Removing statistical outliers with a Z-score threshold of {Z_THRESHOLD}...")
# Calculate Z-scores for each key metric within each propeller group
master_df['z_score_prop'] = master_df.groupby('filename')['prop_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_motor'] = master_df.groupby('filename')['motor_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_sys'] = master_df.groupby('filename')['system_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_vib'] = master_df.groupby('filename')[VIB_COL].transform(lambda x: np.abs((x - x.mean()) / x.std()))

# Filter out rows where any of the Z-scores are too high
initial_rows = len(master_df)
df_cleaned = master_df[
    (master_df['z_score_prop'] < Z_THRESHOLD) &
    (master_df['z_score_motor'] < Z_THRESHOLD) &
    (master_df['z_score_sys'] < Z_THRESHOLD) &
    (master_df['z_score_vib'] < Z_THRESHOLD)
].copy()
print(f"  - Removed {initial_rows - len(df_cleaned)} statistical outlier data points.")


# ===================================================================
# --- 6. Save Final Cleaned Dataset ---
# ===================================================================
# Save the CLEANED DataFrame to the Parquet file
df_cleaned.to_parquet(OUTPUT_FILE_PATH, index=False)
print(f"\nProcess complete. Cleaned master dataset with {len(df_cleaned)} rows saved to '{OUTPUT_FILE_PATH}'")