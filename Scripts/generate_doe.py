import pandas as pd
from scipy.stats import qmc

# ===================================================================
# --- 1. DEFINE YOUR DESIGN SPACE ---
# ===================================================================

# Define the number of unique propellers you want to test
NUM_SAMPLES = 25
# Define the number of geometric parameters
NUM_DIMENSIONS = 4

# Define the lower and upper bounds for each parameter IN ORDER
# [AR, lambda, span, aoaRoot, aoaTip]
lower_bounds = [4.0, 0.4, 8.0, 2.0]
upper_bounds = [12.0, 1.0, 20.0, 8.0]

# Define the column names for the output file
column_names = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)']

# Define the output filename
output_filename = 'Experiment/doe_test_plan.csv'

# ===================================================================
# --- 2. GENERATE THE LATIN HYPERCUBE SAMPLE ---
# ===================================================================

print("Generating Latin Hypercube Sample...")
# Create the sampler object
sampler = qmc.LatinHypercube(d=NUM_DIMENSIONS)

# Get a sample in the unit hypercube [0, 1]
sample = sampler.random(n=NUM_SAMPLES)

# Scale the sample from the unit hypercube to your real engineering units
scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
print("Sample generation complete.")

# ===================================================================
# --- 3. CREATE AND SAVE THE TEST PLAN ---
# ===================================================================

# Create a pandas DataFrame from the scaled sample
doe_df = pd.DataFrame(scaled_sample, columns=column_names)

# Optional: Add a column for a unique test ID
doe_df.insert(0, 'Test_ID', [f'Prop_{i+1:02d}' for i in range(NUM_SAMPLES)])

# Save the test plan to a CSV file
doe_df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created DOE test plan with {NUM_SAMPLES} designs.")
print(f"Test plan saved to '{output_filename}'")
print("\nFirst 5 designs to test:")
print(doe_df.head())