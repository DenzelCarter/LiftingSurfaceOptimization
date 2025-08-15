import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# ===================================================================
# --- 1. SETUP ---
# ===================================================================
INPUT_CSV_PATH = 'Experiment/model_comparison.csv'
PLOTS_FOLDER = 'Plots'

# --- Define the different efficiency metrics to be compared ---
# This dictionary now controls the entire script's logic.
METRICS_TO_COMPARE = [
    {
        'name': 'System Efficiency',
        'prefix': 'se',
        'model_path': 'Trained_Models/system_efficiency_model.joblib',
        'features': ['span (m)', 'AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'Motor Electrical Speed (RPM)']
    },
    {
        'name': 'Propeller Efficiency',
        'prefix': 'pe',
        'model_path': 'Trained_Models/prop_efficiency_model.joblib',
        'features': ['span (m)', 'AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'Motor Electrical Speed (RPM)']
    },
    {
        'name': 'Motor Efficiency',
        'prefix': 'me',
        'model_path': 'Trained_Models/motor_efficiency_model.joblib',
        'features': ['Mechanical Power (W)', 'Motor Electrical Speed (RPM)']
    }
]

# ===================================================================
# --- 2. Load Input Data ---
# ===================================================================
print(f"Loading data from '{INPUT_CSV_PATH}'...")
try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_CSV_PATH}' not found.")
    exit()

# ===================================================================
# --- 3. Run ML Predictions for Each Metric ---
# ===================================================================
print("\nRunning ML model predictions for all efficiency types...")

for metric in METRICS_TO_COMPARE:
    metric_name = metric['name']
    prefix = metric['prefix']
    model_path = metric['model_path']
    features = metric['features']
    
    print(f"  - Predicting for: {metric_name}")
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"    Warning: Model file not found at '{model_path}'. Skipping ML comparison for this metric.")
        continue

    if not all(feature in df.columns for feature in features):
        print(f"    Warning: Input CSV is missing columns required for this model. Skipping.")
        continue
    
    # Predict and store in a new column (e.g., 'se_ml_model')
    X_for_prediction = df[features]
    df[f'{prefix}_ml_model'] = model.predict(X_for_prediction)

print("All ML predictions complete.")

# ===================================================================
# --- 4. Generate, Save, and Display Comparison Plots ---
# ===================================================================
os.makedirs(PLOTS_FOLDER, exist_ok=True)
print(f"\nGenerating comparison plots. Files will be saved to '{PLOTS_FOLDER}/'")

for metric in METRICS_TO_COMPARE:
    metric_name = metric['name']
    prefix = metric['prefix']
    
    print(f"\n--- Plotting: {metric_name} ---")

    # --- Build the comparison DataFrame for the current metric ---
    cols_to_compare = [f'{prefix}_analytical', f'{prefix}_comsol']
    ml_col_name = f'{prefix}_ml_model'
    
    # Only include the ML model column if its predictions were successfully generated
    if ml_col_name in df.columns:
        cols_to_compare.append(ml_col_name)
    else:
        print("    (Skipping ML model in plots as it was not loaded/run)")
        
    comparison_df = df[cols_to_compare].copy()
    
    # Rename columns for cleaner plot labels
    rename_dict = {f'{prefix}_analytical': 'Analytical', f'{prefix}_comsol': 'COMSOL'}
    if ml_col_name in df.columns:
        rename_dict[ml_col_name] = 'ML Model'
    comparison_df.rename(columns=rename_dict, inplace=True)

    # --- Calculate and Print MAPE Metric ---
    print("  - Comparison Metrics (vs COMSOL):")
    mape_ana = mean_absolute_percentage_error(comparison_df['COMSOL'], comparison_df['Analytical'])
    print(f"    - MAPE (Analytical): {mape_ana:.2%}")
    
    if 'ML Model' in comparison_df.columns:
        mape_ml = mean_absolute_percentage_error(comparison_df['COMSOL'], comparison_df['ML Model'])
        print(f"    - MAPE (ML Model): {mape_ml:.2%}")
    
    # --- Plot 1: Correlation Heatmap ---
    plt.figure(figsize=(8, 6))
    corr_matrix = comparison_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'Correlation Matrix of {metric_name} Predictions', fontsize=16)
    
    heatmap_filename = f'{metric_name.replace(" ", "_")}_correlation_heatmap.pdf'
    heatmap_save_path = os.path.join(PLOTS_FOLDER, heatmap_filename)
    plt.savefig(heatmap_save_path, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Paired Scatter Plots ---
    g = sns.pairplot(comparison_df, corner=True, diag_kind='kde')
    g.fig.suptitle(f'Comparison of {metric_name} Predictions', y=1.02, fontsize=16)

    for i, j in zip(*np.tril_indices_from(g.axes, k=-1)):
        g.axes[i, j].plot(g.axes[i, j].get_xlim(), g.axes[i, j].get_xlim(), '--r', lw=2)

    pairplot_filename = f'{metric_name.replace(" ", "_")}_pairplot_comparison.pdf'
    pairplot_save_path = os.path.join(PLOTS_FOLDER, pairplot_filename)
    g.savefig(pairplot_save_path, bbox_inches='tight')
    plt.close()

print("\nAll plots generated successfully.")