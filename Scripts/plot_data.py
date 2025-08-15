import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# ===================================================================
# --- 1. SETUP: Define Constants and Parameters ---
# ===================================================================
# --- File Paths ---
DOE_FILE_PATH = 'Experiment/doe_test_plan_01.csv'
RAW_DATA_DIR = 'Experiment/raw_data'
PLOTS_FOLDER = 'Plots'

# --- Physical Constants ---
R_HUB = 0.040
RHO = 1.225

# --- Processing Parameters ---
NUM_BINS = 20      # Number of RPM bins to average data into
Z_THRESHOLD = 3.0  # Z-score for outlier removal (3 is standard)

# --- Column Names ---
RPM_COL = 'Motor Electrical Speed (RPM)'
THRUST_COL = 'Thrust (N)'
ELEC_POWER_COL = 'Electrical Power (W)'

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

# ===================================================================
# --- 3. Clean Data and Calculate System Efficiency ---
# ===================================================================
print("\nCleaning data and calculating system efficiency...")
# Convert key columns to numeric
for col in [THRUST_COL, ELEC_POWER_COL, RPM_COL]:
    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
master_df.replace(0, np.nan, inplace=True)
master_df.dropna(subset=[THRUST_COL, ELEC_POWER_COL, RPM_COL], inplace=True)

# --- Calculate Ideal Power and System Efficiency ---
master_df['D'] = 2 * (R_HUB + master_df['span (m)'])
master_df['A'] = np.pi * (master_df['D'] / 2)**2
master_df['ideal_power'] = np.sqrt(master_df[THRUST_COL]**3 / (2 * RHO * master_df['A']))
master_df['system_efficiency'] = master_df['ideal_power'] / master_df[ELEC_POWER_COL]

master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
master_df.dropna(subset=['system_efficiency'], inplace=True)

# ===================================================================
# --- 4. Remove Outliers ---
# ===================================================================
print(f"Removing outliers with a Z-score threshold of {Z_THRESHOLD}...")
# Calculate Z-scores for system efficiency within each propeller group
master_df['z_score_sys_eff'] = master_df.groupby('filename')['system_efficiency'].transform(lambda x: np.abs((x - x.mean()) / x.std()))

# Filter out the outliers
df_cleaned = master_df[master_df['z_score_sys_eff'] < Z_THRESHOLD].copy()
print(f"Removed {len(master_df) - len(df_cleaned)} outlier data points.")

# ===================================================================
# --- 5. Bin Data to Reduce Noise ---
# ===================================================================
print(f"Binning data into {NUM_BINS} RPM buckets...")
value_cols = [RPM_COL, 'system_efficiency'] # Only need to bin these two columns now

def bin_group(group):
    if len(group) > 1:
        group['rpm_bin'] = pd.cut(group[RPM_COL], bins=NUM_BINS, labels=False)
        return group.groupby('rpm_bin')[value_cols].mean()
    return None

df_binned = df_cleaned.groupby('filename').apply(bin_group).reset_index()

# ===================================================================
# --- 6. Generate and Save Plot ---
# ===================================================================
# --- Initialize Plot ---
fig, ax = plt.subplots(figsize=(14, 9))
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"\nGenerating plot. File will be saved to '{PLOTS_FOLDER}/'")

# Get a unique color for each propeller
palette = sns.color_palette("husl", n_colors=df_binned['filename'].nunique())
color_map = {prop: color for prop, color in zip(df_binned['filename'].unique(), palette)}

# --- Loop through final data and plot ---
for prop_name, prop_data in df_binned.groupby('filename'):
    prop_label = prop_name.replace('.csv', '')
    color = color_map[prop_name]
    ax.plot(prop_data[RPM_COL], prop_data['system_efficiency'], marker='o', linestyle='-', markersize=4, label=prop_label, color=color)

# --- Finalize the Plot ---
ax.set_title('System Efficiency vs. RPM for All Propeller Designs', fontsize=16)
ax.set_xlabel('RPM', fontsize=12)
ax.set_ylabel('System Efficiency', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Propeller")
save_path = os.path.join(PLOTS_FOLDER, 'System_Efficiency_vs_RPM_All_Props.pdf')
fig.savefig(save_path, bbox_inches='tight')
print(f"  - System Efficiency plot saved to '{save_path}'")

# --- Display Plot ---
print("Displaying plot...")
plt.show()