# Scripts/09_validate_optimal_design.py
# This script provides the final validation for the framework. It loads the
# pre-trained GPR model and compares its predictions directly against the
# newly measured experimental data for a single, optimized design.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
from sklearn.metrics import mean_squared_error
import path_utils

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # --- 1. Load the Pre-Trained GPR Model and Full Dataset ---
    print("--- Loading pre-trained GPR model and full dataset... ---")
    models_dir = Path(P["outputs_models"])
    master_parquet_path = Path(P["master_parquet"])
    
    try:
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        df_full = pd.read_parquet(master_parquet_path)
    except FileNotFoundError as e:
        raise SystemExit(f"Error: A required file was not found. Please run previous scripts. Details: {e}")

    # --- 2. Identify and Isolate the New Validation Design ---
    # We now identify the validation design by its unique filename.
    validation_filename = 'LS_052_01'
    df_hover = df_full[df_full['flight_mode'] == 'hover']
    
    # The 'filename' column in the master dataset does not include the '.csv'
    df_validation = df_hover[df_hover['filename'] == validation_filename].copy()

    if df_validation.empty:
        raise SystemExit(f"Error: Could not find validation data for filename '{validation_filename}' in the master dataset. Please ensure 01_process_data.py has been run.")
    
    print(f"Successfully isolated validation data for: {validation_filename}")

    # --- 3. Predict Performance vs. Measured Performance ---
    features = gpr_hover.feature_names_in_
    X_validation = df_validation[features]
    y_validation_measured = df_validation['performance']

    # Get GPR predictions and uncertainty (standard deviation)
    y_validation_predicted, std_dev = gpr_hover.predict(X_validation, return_std=True)
    
    # Sort values by op_speed for clean plotting
    sort_indices = X_validation['op_speed'].argsort()
    X_validation_sorted = X_validation.iloc[sort_indices]
    y_measured_sorted = y_validation_measured.iloc[sort_indices]
    y_predicted_sorted = y_validation_predicted[sort_indices]
    std_dev_sorted = std_dev[sort_indices]
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_measured_sorted, y_predicted_sorted))
    
    # --- 4. Generate the Validation Plot ---
    print("--- Generating validation plot... ---")
    plots_dir = Path(P["outputs_plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "09_validation_plot.pdf"

    with plt.rc_context({'font.size': 12}):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(f'Validation of Optimized Design ({validation_filename})', fontsize=16)
        ax.set_xlabel('Operational Speed (RPM)', fontsize=12)
        ax.set_ylabel('Hover Performance (η)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Plot the measured data points
        ax.plot(X_validation_sorted['op_speed'], y_measured_sorted, 
                'o', color='red', markersize=8, label='Measured Performance')

        # Plot the GPR prediction line
        ax.plot(X_validation_sorted['op_speed'], y_predicted_sorted, 
                '-', color='blue', linewidth=2, label='GPR Mean Prediction')

        # Plot the uncertainty bands (95% confidence interval)
        ax.fill_between(X_validation_sorted['op_speed'], 
                        y_predicted_sorted - 1.96 * std_dev_sorted, 
                        y_predicted_sorted + 1.96 * std_dev_sorted, 
                        color='blue', alpha=0.2, label='95% Confidence Interval')

        ax.legend()
        
        # Add the RMSE value to the plot for a quantitative summary
        ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

    print(f"\nSuccessfully generated validation plot: {plot_path}")
    print(f"Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()