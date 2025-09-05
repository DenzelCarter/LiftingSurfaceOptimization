import pandas as pd
import numpy as np
import os

# ===================================================================
# --- 1. SETUP: Define File Path and Parameters ---
# ===================================================================
# --- IMPORTANT: Change this to the specific raw data file you want to analyze ---
RAW_FILE_PATH = 'Experiment/data/Prop_18_PLA_01.csv' 

# The specific ESC signal value to isolate for the stable condition
STABLE_ESC_SIGNAL = 1343 # in microseconds (µs)

# The list of columns you want to analyze for SNR
TARGET_COLUMNS = [
    'Thrust (N)',
    'Torque (N·m)',
    'Motor Electrical Speed (RPM)',
    'Mechanical Power (W)',
    'Voltage (V)',
    'Current (A)',
    'Electrical Power (W)'
]

# ===================================================================
# --- 2. Load Data and Isolate Stable Condition ---
# ===================================================================
if not os.path.exists(RAW_FILE_PATH):
    print(f"Error: Data file not found at '{RAW_FILE_PATH}'")
    exit()

print(f"Loading data from '{RAW_FILE_PATH}'...")
df = pd.read_csv(RAW_FILE_PATH)

# --- Calculate derived power columns if they don't exist ---
if 'Electrical Power (W)' not in df.columns:
    df['Electrical Power (W)'] = df['Voltage (V)'] * df['Current (A)']
if 'Mechanical Power (W)' not in df.columns:
    df['Mechanical Power (W)'] = df['Torque (N·m)'] * df['Motor Electrical Speed (RPM)'] * (2 * np.pi / 60)


print(f"Isolating data points where 'ESC signal (µs)' == {STABLE_ESC_SIGNAL}...")
df_stable = df[df['ESC signal (µs)'] == STABLE_ESC_SIGNAL].copy()

# ===================================================================
# --- 3. Calculate and Print SNR for Each Column ---
# ===================================================================
if len(df_stable) > 1:
    print("\n--- SNR Calculation Results ---")
    print(f"Number of data points at {STABLE_ESC_SIGNAL}µs: {len(df_stable)}")

    for column in TARGET_COLUMNS:
        if column in df_stable.columns:
            readings = df_stable[column]
            
            signal = readings.mean()
            noise = readings.std()
            
            if noise > 0:
                snr = abs(signal / noise) # Use absolute value for SNR
                snr_db = 20 * np.log10(snr)
            else:
                snr = float('inf')
                snr_db = float('inf')

            print(f"\n--- {column} ---")
            print(f"  Signal (Mean): {signal:.4f}")
            print(f"  Noise (Std Dev): {noise:.4f}")
            print(f"  Signal-to-Noise Ratio (SNR): {snr:.2f}")

        else:
            print(f"\n--- {column} ---")
            print(f"  Column not found in the data file.")

else:
    print(f"\nCould not calculate SNR.")
    print(f"Found {len(df_stable)} data points for ESC signal {STABLE_ESC_SIGNAL}µs.")
    print("Need at least two data points to calculate standard deviation.")