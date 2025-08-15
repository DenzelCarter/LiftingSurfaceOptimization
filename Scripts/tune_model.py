import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# ===================================================================
# --- 1. SETUP: Define File Paths and Column Names ---
# ===================================================================
MASTER_DATASET_PATH = 'Experiment/master_dataset.parquet'

# --- Define the features the model should be trained on ---
# This ensures we don't leak other data (like prop_efficiency) into the model
FEATURES = [
    'span (m)',
    'AR',
    'lambda',
    'aoaRoot (deg)',
    'aoaTip (deg)',
    'Motor Electrical Speed (RPM)'
]
TARGET_COL = 'system_efficiency'

# ===================================================================
# --- 2. Load and Prepare Data ---
# ===================================================================
print("Loading data...")
master_df = pd.read_parquet(MASTER_DATASET_PATH)

# --- Ensure we only use the specified features and target, and drop any NaNs ---
required_cols = FEATURES + [TARGET_COL]
master_df.dropna(subset=required_cols, inplace=True)

# --- Define Features (X) and Target (y) using our explicit list ---
X = master_df[FEATURES]
y = master_df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data loaded and prepared with {len(FEATURES)} features.")

# ===================================================================
# --- 3. Define the Parameter Grid ---
# ===================================================================
# These are the parameters and values the Grid Search will test.
# Your original grid is well-defined and a great starting point.
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': [2, 3, 4, 'sqrt'],
    'max_depth': [10, 20, None]
}

# ===================================================================
# --- 4. Set up and Run the Grid Search ---
# ===================================================================
print("Starting Grid Search... this may take a while.")
# Initialize the model
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Set up the grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# ===================================================================
# --- 5. Print the Best Results ---
# ===================================================================
print("\n--- Grid Search Complete ---")
print(f"Best R-squared score found: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)