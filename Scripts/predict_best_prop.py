import pandas as pd
import numpy as np
import joblib
import os

# ===================================================================
# --- 1. SETUP ---
# ===================================================================
DOE_FILE_PATH = 'Experiment/doe_test_plan_01.csv'
MODEL_FILE_PATH = 'Trained_Models/prop_efficiency_model.joblib'

# --- Physical Constants ---
SPAN = 0.184
R_HUB = 0.046
RHO_AIR = 1.225
MU_AIR = 1.81e-5  # Dynamic viscosity of air in Pa·s
THICKNESS_CHORD_RATIO = 0.12 # For NACA 0012 airfoil

# Define the RPM range you want to evaluate for each prop
RPM_RANGE_TO_TEST = np.linspace(600, 2000, 15)

# *** IMPORTANT: Update this list with all the props you have tested so far ***
TESTED_PROPS = [
    'Prop_01_T2K_01.csv','Prop_13_PLA_01.csv','Prop_14_T2K_01.csv'
]

# ===================================================================
# --- 2. Load Model and Data ---
# ===================================================================
print(f"Loading model from '{MODEL_FILE_PATH}'...")
try:
    model = joblib.load(MODEL_FILE_PATH)
except FileNotFoundError:
    print(f"Model file not found. Please run train_model.py first.")
    exit()

print(f"Loading DOE plan from '{DOE_FILE_PATH}'...")
doe_df = pd.read_csv(DOE_FILE_PATH)

# ===================================================================
# --- 3. Create Prediction Set for Untested Props ---
# ===================================================================
untested_df = doe_df[~doe_df['filename'].isin(TESTED_PROPS)].copy()

if untested_df.empty:
    print("Congratulations, all propellers have been tested!")
    exit()

print(f"Found {len(untested_df)} untested designs. Generating test points for prediction...")

prediction_points = []
for index, row in untested_df.iterrows():
    for rpm in RPM_RANGE_TO_TEST:
        new_row = row.copy()
        new_row['Motor Electrical Speed (RPM)'] = rpm
        prediction_points.append(new_row)

X_to_predict = pd.DataFrame(prediction_points)

# ===================================================================
# --- 4. Engineer New Features for the Prediction Set ---
# ===================================================================
print("Engineering new features (thickness, Reynolds number)...")

# --- Calculate Average Thickness ---
root_chord = (2 * SPAN) / (X_to_predict['AR'] * (1 + X_to_predict['lambda']))
tip_chord = X_to_predict['lambda'] * root_chord
average_chord = (root_chord + tip_chord) / 2
X_to_predict['avg_thickness_m'] = THICKNESS_CHORD_RATIO * average_chord

# --- Calculate Reynolds Number ---
chord_at_75 = root_chord - 0.75 * (root_chord - tip_chord)
radius_at_75 = R_HUB + (0.75 * SPAN)
velocity_at_75 = (X_to_predict['Motor Electrical Speed (RPM)'] * (2 * np.pi / 60)) * radius_at_75
X_to_predict['reynolds_number'] = (RHO_AIR * velocity_at_75 * chord_at_75) / MU_AIR

# ===================================================================
# --- 5. Make Predictions and Identify Best Candidate ---
# ===================================================================
print("Making predictions on untested designs...")

# Ensure the columns are in the exact order the model expects
X_for_prediction = X_to_predict[model.feature_names_in_]

# Use the reordered dataframe for prediction
predicted_values = model.predict(X_for_prediction)

# Add the predictions back to our dataframe
X_to_predict['predicted_efficiency'] = predicted_values

# Now, find the max predicted efficiency for each propeller design
best_performance_per_prop = X_to_predict.groupby('filename')['predicted_efficiency'].max()

# Sort the results to find the best propeller overall
ranked_results = best_performance_per_prop.sort_values(ascending=False) # Higher efficiency is better

print("\n--- Top 5 Recommended Propellers to Test Next (Based on Max Predicted Propeller Efficiency) ---")
print(ranked_results.head())

if not ranked_results.empty:
    best_prop_name = ranked_results.index[0]
    print(f"\nModel prediction complete. The single best propeller to test next is: {best_prop_name}")
else:
    print("\nNo predictions could be made on the remaining untested props.")