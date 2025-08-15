import pandas as pd
import numpy as np
import os

# ===================================================================
# --- 1. SETUP: Define Constants and Parameters ---
# ===================================================================
# --- File Paths ---
DOE_FILE_PATH = 'Experiment/doe_test_plan_01.csv'
RAW_DATA_DIR = 'Experiment/raw_data_PLA'
OUTPUT_FILE_PATH = 'Experiment/master_dataset.parquet'

# --- Physical Constants ---
R_HUB = 0.040
RHO = 1.225

# --- Processing Parameters ---
Z_THRESHOLD = 3.0
MIN_THRUST_THRESHOLD = 0.1  # N
MIN_RPM_THRESHOLD = 500     # RPM

# --- Column Names ---
RPM_COL = 'Motor Electrical Speed (RPM)'
THRUST_COL = 'Thrust (N)'
TORQUE_COL = 'Torque (N·m)'
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
# --- 3. Clean Data and Calculate Efficiencies ---
# ===================================================================
# In process_data.py, at the start of Section 3

# --- ADD THIS NEW TARE CORRECTION BLOCK ---
TARE_LOOKUP_FILE = 'Experiment/tare_lookup.csv'
print(f"Applying spinning tare correction using '{TARE_LOOKUP_FILE}'...")

try:
    # Load the pre-processed tare data
    tare_df = pd.read_csv(TARE_LOOKUP_FILE)

    # Create RPM bins (rounded to nearest 50 RPM) in both dataframes
    # This allows us to match each data point to its corresponding tare value
    master_df['rpm_bin'] = (master_df[RPM_COL] / 50).round() * 50
    
    # Merge the tare data into the master dataframe based on the RPM bin
    # 'how=left' ensures we keep all the original data
    master_df = pd.merge(master_df, tare_df, on='rpm_bin', how='left')

    # Replace any non-matches with 0 to avoid errors
    master_df[['thrust_tare', 'torque_tare']] = master_df[['thrust_tare', 'torque_tare']].fillna(0)

    # Perform the subtraction to get the corrected values
    master_df[THRUST_COL] = master_df[THRUST_COL] - master_df['thrust_tare']
    master_df[TORQUE_COL] = master_df[TORQUE_COL] - master_df['torque_tare']
    
    print("  - Tare correction applied successfully.")

except FileNotFoundError:
    print(f"  - Warning: Tare lookup file not found. Proceeding without correction.")
except Exception as e:
    print(f"  - Warning: Could not apply tare correction. Error: {e}")

# --- END OF NEW BLOCK ---


# ...The rest of your data cleaning and efficiency calculations continue here...
# IMPORTANT: Ensure all subsequent calculations for Mechanical Power, Ideal Power,
# and efficiencies use the newly corrected Thrust and Torque columns.

print("Cleaning data and calculating efficiencies...")

# --- Define minimum thresholds based on spec sheet ---
MIN_THRUST_N = 0.05
MIN_TORQUE_NM = 0.005
MIN_RPM = 500

# --- Initial Conversion and Cleanup ---
for col in [THRUST_COL, ELEC_POWER_COL, MECH_POWER_COL, RPM_COL, TORQUE_COL]: 
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
master_df.replace(0, np.nan, inplace=True)
master_df.dropna(subset=[THRUST_COL, ELEC_POWER_COL, RPM_COL, TORQUE_COL], inplace=True)
initial_rows = len(master_df)

# --- Apply Filters Based on Sensor Limits ---
print(f"Applying filters: Thrust > {MIN_THRUST_N}N, Torque > {MIN_TORQUE_NM}Nm, RPM > {MIN_RPM}")
master_df = master_df[
    (master_df[THRUST_COL] > MIN_THRUST_N) &
    (master_df[TORQUE_COL] > MIN_TORQUE_NM) &
    (master_df[RPM_COL] > MIN_RPM)
].copy()
print(f"  - Removed {initial_rows - len(master_df)} rows that were below sensor thresholds.")
initial_rows = len(master_df)

# --- Calculate Ideal Power and Core Efficiencies ---
master_df['D'] = 2 * (R_HUB + master_df['span (m)'])
master_df['A'] = np.pi * (master_df['D'] / 2)**2
master_df['ideal_power'] = np.sqrt(master_df[THRUST_COL]**3 / (2 * RHO * master_df['A']))

master_df['prop_efficiency'] = master_df['ideal_power'] / master_df[MECH_POWER_COL]
master_df['motor_efficiency'] = master_df[MECH_POWER_COL] / master_df[ELEC_POWER_COL]
master_df['system_efficiency'] = master_df['ideal_power'] / master_df[ELEC_POWER_COL]

# --- Filter out physically impossible results ---
master_df = master_df[(master_df['prop_efficiency'] <= 1.0) & (master_df['motor_efficiency'] <= 1.0)]
print(f"  - Removed {initial_rows - len(master_df)} rows with calculated efficiencies > 100%.")

# ... rest of the script (outlier removal, etc.) continues here ...

# In process_data.py

# ===================================================================
# --- 4. Remove Statistical Outliers ---
# ===================================================================
print(f"Removing statistical outliers with a Z-score threshold of {Z_THRESHOLD}...")

# Calculate Z-scores for each key efficiency within each propeller group
master_df['z_score_prop'] = master_df.groupby('filename')['prop_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_motor'] = master_df.groupby('filename')['motor_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))
master_df['z_score_sys'] = master_df.groupby('filename')['system_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))

# --- ADD THIS LINE ---
# Add a Z-score calculation for vibration to detect desync spikes
master_df['z_score_vib'] = master_df.groupby('filename')['Vibration (g)'].transform(lambda x: np.abs((x - x.mean()) / x.std()))


# --- UPDATE THIS FILTER ---
# Filter out rows where any of the Z-scores are too high
initial_rows = len(master_df)
df_cleaned = master_df[
    (master_df['z_score_prop'] < Z_THRESHOLD) &
    (master_df['z_score_motor'] < Z_THRESHOLD) &
    (master_df['z_score_sys'] < Z_THRESHOLD) &
    (master_df['z_score_vib'] < Z_THRESHOLD)  # Add the vibration check here
].copy()
print(f"Removed {initial_rows - len(df_cleaned)} statistical outlier data points.")

# ===================================================================
# --- 5. Save Final Cleaned Dataset ---
# ===================================================================
# Save the CLEANED DataFrame to the Parquet file
df_cleaned.to_parquet(OUTPUT_FILE_PATH)
print(f"\nProcess complete. Cleaned master dataset with {len(df_cleaned)} rows saved to '{OUTPUT_FILE_PATH}'")