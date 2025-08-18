import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================================
# --- 1. SETUP: Define Paths, Features, and Holdout Set ---
# ===================================================================
# --- File Paths ---
MASTER_DATASET_PATH = 'Experiment/tools/master_dataset.parquet'
MODEL_SAVE_PATH = 'Trained_Models/system_efficiency_model.joblib'
PLOTS_FOLDER = 'Plots'

# --- Feature and Target Definitions ---
# Define the features for the system model
X_FEATURES = [
    'AR',
    'lambda',
    'aoaRoot (deg)',
    'aoaTip (deg)',
    'Motor Electrical Speed (RPM)',
    'flexMod (GPA)',
    'avg_thickness_m',
    'reynolds_number',
    'Voltage (V)'
]
# Define the target variable
Y_TARGET = 'system_efficiency'

# --- Holdout Test Set Definition ---
# *** IMPORTANT: Choose one or more props you have tested to hold out for the final test ***
TEST_PROP_FILENAMES = ['Prop_14_T2K_01.csv'] # Example: using prop 14 data as the test set

# ===================================================================
# --- 2. Load and Prepare Data ---
# ===================================================================
print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
if not os.path.exists(MASTER_DATASET_PATH):
    print(f"Error: Master dataset not found. Please run process_data.py first.")
    exit()

master_df = pd.read_parquet(MASTER_DATASET_PATH)

# Drop any rows with missing values in the columns we need
master_df.dropna(subset=X_FEATURES + [Y_TARGET], inplace=True)
print(f"Loaded {len(master_df)} clean data points.")

# ===================================================================
# --- 3. Split Data into Training and Test Sets ---
# ===================================================================
print(f"Holding out data from the following props for the test set: {TEST_PROP_FILENAMES}")
train_df = master_df[~master_df['filename'].isin(TEST_PROP_FILENAMES)]
test_df = master_df[master_df['filename'].isin(TEST_PROP_FILENAMES)]

if train_df.empty or test_df.empty:
    print("Error: Could not create both a training and a test set.")
    print("Please check your TEST_PROP_FILENAMES list and ensure you have tested multiple props.")
    exit()

print(f"Training set size: {len(train_df)} data points")
print(f"Test set size: {len(test_df)} data points")

# Define features (X) and target (y) for both sets
X_train = train_df[X_FEATURES]
y_train = train_df[Y_TARGET]
X_test = test_df[X_FEATURES]
y_test = test_df[Y_TARGET]

# ===================================================================
# --- 4. Train the Random Forest Model ---
# ===================================================================
print("\nTraining the Random Forest model...")
model = RandomForestRegressor(
    n_estimators=500,
    max_features='sqrt',
    oob_score=True,  # Use Out-of-Bag score for a cross-validation estimate
    random_state=42,
    n_jobs=-1        # Use all available CPU cores
)
model.fit(X_train, y_train)
print("Model training complete.")

# ===================================================================
# --- 5. Evaluate Model Performance ---
# ===================================================================
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")

# --- Predicted vs. Actual Plot ---
os.makedirs(PLOTS_FOLDER, exist_ok=True)
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Model Predictions')

# Set plot limits to be equal and add a perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2, label="Perfect Prediction")
plt.xlim(min_val * 0.95, max_val * 1.05)
plt.ylim(min_val * 0.95, max_val * 1.05)

plt.xlabel("Actual System Efficiency")
plt.ylabel("Predicted System Efficiency")
plt.title(f"Predicted vs. Actual Performance (R² = {r2:.3f})")
plt.legend()
plt.grid(True)
plt.tight_layout()

perf_save_path = os.path.join(PLOTS_FOLDER, 'system_model_performance_predicted_vs_actual.pdf')
plt.savefig(perf_save_path, bbox_inches='tight')
print(f"Saved performance plot to '{perf_save_path}'")
plt.close()

# ===================================================================
# --- 6. Analyze Feature Importance ---
# ===================================================================
feature_importance_df = pd.DataFrame({
    'feature': X_FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
# Corrected barplot call to handle future Seaborn versions
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', hue='feature', legend=False)
plt.title("Feature Importance for System Efficiency Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

imp_save_path = os.path.join(PLOTS_FOLDER, 'system_model_feature_importance.pdf')
plt.savefig(imp_save_path, bbox_inches='tight')
print(f"Saved importance plot to '{imp_save_path}'")
plt.close()

# ===================================================================
# --- 7. Save the Trained Model ---
# ===================================================================
joblib.dump(model, MODEL_SAVE_PATH, compress=3)
print(f"\nTrained model saved to '{MODEL_SAVE_PATH}'")
print("\nScript finished on this Monday afternoon in Cambridge.")