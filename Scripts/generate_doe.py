import pandas as pd
from scipy.stats import qmc
import os

# ===================================================================
# --- 1. DEFINE YOUR DESIGN SPACE ---
# ===================================================================

# --- File and Sample Settings ---
NUM_SAMPLES = 25
OUTPUT_FILENAME = 'Experiment/tools/doe_test_plan_02.csv'

# --- Fixed Parameters (Constants for all designs in this plan) ---
DEFAULT_FLEX_MOD_GPA = 2.75 # Default to PLA Basic, can be changed in the CSV

# --- Variable Parameters (for Latin Hypercube Sampling) ---
# Define the lower and upper bounds for each variable parameter
# The order is: [AR, lambda, aoaRoot, aoaTip]
LOWER_BOUNDS = [6.0, 0.5, 10.0, 3.0]
UPPER_BOUNDS = [10.0, 1.0, 20.0, 8.0]
COLUMN_NAMES = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)']

# ===================================================================
# --- 2. GENERATE THE LATIN HYPERCUBE SAMPLE ---
# ===================================================================

print("Generating Latin Hypercube Sample...")
# Create the sampler object for our 4 variable dimensions
sampler = qmc.LatinHypercube(d=len(LOWER_BOUNDS))

# Get a sample in the unit hypercube [0, 1]
sample = sampler.random(n=NUM_SAMPLES)

# Scale the sample from the unit hypercube to your real engineering units
scaled_sample = qmc.scale(sample, LOWER_BOUNDS, UPPER_BOUNDS)
print("Sample generation complete.")

# ===================================================================
# --- 3. CREATE AND SAVE THE TEST PLAN ---
# ===================================================================

# Create a pandas DataFrame from the scaled sample
doe_df = pd.DataFrame(scaled_sample, columns=COLUMN_NAMES)

# --- Add the fixed and placeholder columns ---
# Insert the filename column at the beginning
doe_df.insert(0, 'filename', [f'Prop_{i+26:03d}_MAT_01.csv' for i in range(NUM_SAMPLES)])

# Add the other fixed parameters
doe_df['flexMod (GPA)'] = DEFAULT_FLEX_MOD_GPA

# Reorder columns for clarity
final_columns_order = [
    'filename',
    'AR',
    'lambda',
    'aoaRoot (deg)',
    'aoaTip (deg)',
    'flexMod (GPA)'
]
doe_df = doe_df[final_columns_order]

# --- Save the final test plan to a CSV file ---
# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
doe_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\nSuccessfully created DOE test plan with {NUM_SAMPLES} new designs.")
print(f"Test plan saved to '{OUTPUT_FILENAME}'")
print("\nFirst 5 designs of the new plan:")
print(doe_df.head())