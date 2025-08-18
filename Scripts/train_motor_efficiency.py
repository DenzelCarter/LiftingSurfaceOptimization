import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================================
# --- 1. SETUP: Define File Paths and Column Names ---
# ===================================================================
# --- File Paths ---
MASTER_DATASET_PATH = 'Experiment/master_dataset.parquet'
MODEL_SAVE_PATH = 'Trained_Models/motor_efficiency_model.joblib'
PLOTS_FOLDER = 'Plots'

# --- Column Names ---
# Define the features the model will be trained on
MECH_COL = 'Mechanical Power (W)'
ELEC_COL = 'Electrical Power (W)'
RPM_COL = 'Motor Electrical Speed (RPM)'

# Define the target variable the model will predict
TARGET_COL = 'motor_efficiency' # <-- UPDATED to our new target

# ===================================================================
# --- 2. Load and Prepare Data ---
# ===================================================================
print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
if not os.path.exists(MASTER_DATASET_PATH):
    print(f"Error: Master dataset not found. Please run process_data.py first.")
    exit()
df = pd.read_parquet(MASTER_DATASET_PATH)
print("Dataset loaded successfully.")

# --- Define the list of columns we need for this model ---
features = [MECH_COL, ELEC_COL, RPM_COL]
features_and_target = features + [TARGET_COL]

# --- Final cleaning step: drop any rows with missing values in our selected columns ---
df.dropna(subset=features_and_target, inplace=True)

print(f"Data prepared. Using {len(df)} clean data points for training.")
if df.empty:
    print(f"Error: No data remains after cleaning. Cannot train model.")
    exit()

# ===================================================================
# --- 3. Define Features (X) and Target (y) ---
# ===================================================================
X = df[features]
y = df[TARGET_COL]

# ===================================================================
# --- 4. Split Data for Training and Testing ---
# ===================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# ===================================================================
# --- 5. Train the Random Forest Model ---
# ===================================================================
print("Training the Random Forest model...")
# Using the parameters found from tuning
model = RandomForestRegressor(
    n_estimators=500,
    max_features='sqrt',
    max_depth=None,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model training complete.")

# ===================================================================
# --- 6. Evaluate Model Performance ---
# ===================================================================
print("Evaluating model performance...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")

# --- Predicted vs. Actual Plot ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, label='Model Predictions')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
padding = (max_val - min_val) * 0.1
plot_lim_min = min_val - padding
plot_lim_max = max_val + padding

plt.xlim(plot_lim_min, plot_lim_max)
plt.ylim(plot_lim_min, plot_lim_max)
plt.plot([plot_lim_min, plot_lim_max], [plot_lim_min, plot_lim_max], '--r', lw=2, label="Perfect Prediction")

# These three lines have been corrected
plt.xlabel("Actual Motor Efficiency")
plt.ylabel("Predicted Motor Efficiency")
plt.title(f"Predicted vs. Actual Motor Efficiency (R² = {r2:.3f})")

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
perf_save_path = os.path.join(PLOTS_FOLDER, 'motor_model_performance_predicted_vs_actual.pdf')
plt.savefig(perf_save_path, bbox_inches='tight')
print(f"Saved performance plot to '{perf_save_path}'")

# ===================================================================
# --- 7. Analyze Feature Importance ---
# ===================================================================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

imp_save_path = os.path.join(PLOTS_FOLDER, 'motor_model_feature_importance.pdf')
plt.savefig(imp_save_path, bbox_inches='tight')
print(f"Saved importance plot to '{imp_save_path}'")

# ===================================================================
# --- 8. Save the Trained Model & Show Plots ---
# ===================================================================
joblib.dump(model, MODEL_SAVE_PATH,compress=3)
print(f"\nTrained model saved to '{MODEL_SAVE_PATH}'")

# print("Displaying plots...")
# plt.show()