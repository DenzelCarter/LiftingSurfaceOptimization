import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

# ===================================================================
# --- 1. SETUP: Define Paths and Features ---
# ===================================================================
MASTER_DATASET_PATH = 'Experiment/tools/master_dataset.parquet'
MODEL_SAVE_PATH = 'Trained_Models/motor_efficiency_model.joblib'
PLOTS_FOLDER = 'Plots'

X_FEATURES = ['Voltage (V)', 'Motor Electrical Speed (RPM)']
Y_TARGET = 'motor_efficiency'

# ===================================================================
# --- 2. Load and Prepare Data ---
# ===================================================================
print("\n--- Training Motor Efficiency Model ---")
if not os.path.exists(MASTER_DATASET_PATH):
    print(f"Error: Master dataset not found. Please run process_data.py first.")
    exit()

master_df = pd.read_parquet(MASTER_DATASET_PATH)
master_df.dropna(subset=X_FEATURES + [Y_TARGET], inplace=True)
print(f"Loaded {len(master_df)} clean data points.")

X = master_df[X_FEATURES]
y = master_df[Y_TARGET]
groups = master_df['filename']

# ===================================================================
# --- 3. Evaluate Model with Cross-Validation ---
# ===================================================================
logo = LeaveOneGroupOut()
model = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1)

print(f"\nPerforming Leave-One-Group-Out Cross-Validation on {len(np.unique(groups))} test runs...")
scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='r2')

print("\n--- Cross-Validation Results ---")
print(f"R² scores for each holdout run: {np.round(scores, 3)}")
print(f"Average R² score: {scores.mean():.4f}")
print(f"Standard Deviation of scores: {scores.std():.4f}")

# ===================================================================
# --- 4. Train Final Model and Analyze Features ---
# ===================================================================
print("\nTraining final model on ALL available data...")
model.fit(X, y)
print("Final model training complete.")

feature_importance_df = pd.DataFrame({
    'feature': X_FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', hue='feature', legend=False)
plt.title("Feature Importance for Motor Efficiency Model")
plt.tight_layout()
os.makedirs(PLOTS_FOLDER, exist_ok=True)
imp_save_path = os.path.join(PLOTS_FOLDER, 'motor_model_feature_importance.pdf')
plt.savefig(imp_save_path, bbox_inches='tight')
print(f"Saved importance plot to '{imp_save_path}'")
plt.close()

# ===================================================================
# --- 5. Save the Final Model ---
# ===================================================================
joblib.dump(model, MODEL_SAVE_PATH, compress=3)
print(f"\nFinal trained model saved to '{MODEL_SAVE_PATH}'")