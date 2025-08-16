import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# ===================================================================
# --- 1. SETUP: Define Constants and Parameters ---
# ===================================================================
# --- File Paths ---
MASTER_DATASET_PATH = 'Experiment/master_dataset.parquet'
PLOTS_FOLDER = 'Plots'

# --- Processing Parameters ---
NUM_BINS = 20      # Number of RPM bins to average data into

# --- Column Names ---
RPM_COL = 'Motor Electrical Speed (RPM)'

# ===================================================================
# --- 2. Load the Pre-Processed Master Dataset ---
# ===================================================================
print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
if not os.path.exists(MASTER_DATASET_PATH):
    print(f"Error: Master dataset not found at '{MASTER_DATASET_PATH}'")
    print("Please run the 'process_data.py' script first.")
    exit()

df_cleaned = pd.read_parquet(MASTER_DATASET_PATH)
print(f"Successfully loaded {len(df_cleaned)} cleaned data points.")

# ===================================================================
# --- 3. Bin Data to Reduce Noise for Plotting ---
# ===================================================================
print(f"Binning data into {NUM_BINS} RPM buckets for plotting...")

# Define all columns that need to be averaged in the bins
value_cols = [
    RPM_COL,
    'system_efficiency',
    'prop_efficiency',
    'motor_efficiency'
]

# Use a more explicit loop for binning to ensure stability and silence warnings
binned_data_list = []
for name, group in df_cleaned.groupby('filename'):
    if len(group) > NUM_BINS:
        # Create bins based on the RPM range for this specific prop
        group['rpm_bin'] = pd.cut(group[RPM_COL], bins=NUM_BINS, labels=False, duplicates='drop')
        
        # Group by the new bins and calculate the mean of all our value columns
        binned_group = group.groupby('rpm_bin')[value_cols].mean()
        
        # Add the filename back in as a column
        binned_group['filename'] = name
        binned_data_list.append(binned_group)

# Concatenate all the binned dataframes back into one
df_binned = pd.concat(binned_data_list).reset_index()

# ===================================================================
# --- 4. Generate and Save Plots ---
# ===================================================================
# --- Create a plotting function to avoid repeating code ---
def create_and_save_plot(data, y_col, title, y_label, file_name):
    """Generic function to create and save an efficiency plot."""
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get a unique color for each propeller
    palette = sns.color_palette("husl", n_colors=data['filename'].nunique())
    color_map = {prop: color for prop, color in zip(data['filename'].unique(), palette)}

    # Loop through final data and plot
    for prop_name, prop_data in data.groupby('filename'):
        prop_label = prop_name.replace('.csv', '')
        color = color_map[prop_name]
        ax.plot(prop_data[RPM_COL], prop_data[y_col], marker='o', linestyle='-', markersize=4, label=prop_label, color=color)

    # Finalize the Plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('RPM', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Propeller")
    
    # Save the figure
    save_path = os.path.join(PLOTS_FOLDER, file_name)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"  - Plot saved to '{save_path}'")
    plt.close(fig) # Close the figure to free up memory

# --- Generate all three plots ---
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"\nGenerating plots. Files will be saved to '{PLOTS_FOLDER}/'")

create_and_save_plot(
    data=df_binned,
    y_col='system_efficiency',
    title='System Efficiency vs. RPM for All Propeller Designs',
    y_label='System Efficiency (Ideal Power / Electrical Power)',
    file_name='System_Efficiency_vs_RPM.pdf'
)

create_and_save_plot(
    data=df_binned,
    y_col='prop_efficiency',
    title='Propeller Efficiency vs. RPM for All Propeller Designs',
    y_label='Propeller Efficiency (Ideal Power / Mechanical Power)',
    file_name='Propeller_Efficiency_vs_RPM.pdf'
)

create_and_save_plot(
    data=df_binned,
    y_col='motor_efficiency',
    title='Motor Efficiency vs. RPM for All Propeller Designs',
    y_label='Motor Efficiency (Mechanical Power / Electrical Power)',
    file_name='Motor_Efficiency_vs_RPM.pdf'
)