import pandas as pd
import os

# ===================================================================
# --- 1. SETUP: Define Constants and File Path ---
# ===================================================================
DOE_FILE_PATH = 'Experiment/doe/doe_test_plan_03.csv'

# --- Fixed Geometric Parameters ---
SPAN_M = 0.184  # The fixed span of a single blade in meters

# ===================================================================
# --- 2. Load the DOE Plan ---
# ===================================================================
print(f"Loading DOE plan from '{DOE_FILE_PATH}'...")
try:
    doe_df = pd.read_csv(DOE_FILE_PATH)
except FileNotFoundError:
    print(f"Error: DOE file not found at '{DOE_FILE_PATH}'")
    exit()

# ===================================================================
# --- 3. Calculate Chord Lengths ---
# ===================================================================
print("Calculating root and tip chords...")

# Use the provided formulas to calculate chord lengths

# Calculate root chord
doe_df['root_chord_mm'] = (2 * SPAN_M) / (doe_df['AR'] * (1 + doe_df['lambda'])) * 1000

# Calculate tip chord
doe_df['tip_chord_mm'] = doe_df['lambda'] * doe_df['root_chord_mm']

# ===================================================================
# --- 4. Display the Results Table ---
# ===================================================================
# Select and reorder columns for a clean output
output_cols = [
    'filename',
    'AR',
    'lambda',
    'root_chord_mm',
    'tip_chord_mm'
]
results_df = doe_df[output_cols]

# Set display options for pandas to show all rows and format numbers
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("\n--- Calculated Propeller Chord Lengths (for Span = 0.19m) ---")
print(results_df)