# Scripts/08_analyze_uncertainty.py
# This script quantifies the overall exploration of the design space by
# calculating the mean uncertainty of the GPR models over a large
# random sample of the design space.

import pandas as pd
import numpy as np
from scipy.stats.qmc import LatinHypercube
from joblib import load
from pathlib import Path
from tqdm import tqdm
import path_utils

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # --- 1. Load Data and Models ---
    print("--- Loading data and trained GPR models... ---")
    models_dir = Path(P["outputs_models"])
    master_parquet_path = Path(P["master_parquet"])
    
    try:
        gpr_hover = load(models_dir / "gpr_hover_model.joblib")
        gpr_cruise = load(models_dir / "gpr_cruise_model.joblib")
        df_full = pd.read_parquet(master_parquet_path)
    except FileNotFoundError as e:
        raise SystemExit(f"Error: A required file was not found. Please run previous scripts. Details: {e}")

    df_hover_data = df_full[df_full['flight_mode'] == 'hover']
    df_cruise_data = df_full[df_full['flight_mode'] == 'cruise']

    # --- 2. Define Design Space Bounds ---
    print("--- Defining the full design space bounds... ---")
    # Use the same feature names the model was trained on
    hover_features = gpr_hover.feature_names_in_
    cruise_features = gpr_cruise.feature_names_in_

    bounds_hover = np.array([
        (df_hover_data[feat].min(), df_hover_data[feat].max()) for feat in hover_features
    ])
    bounds_cruise = np.array([
        (df_cruise_data[feat].min(), df_cruise_data[feat].max()) for feat in cruise_features
    ])

    # --- 3. Generate a Large Random Sample of the Design Space ---
    n_samples = 10000
    print(f"--- Generating {n_samples} random samples using Latin Hypercube Sampling... ---")
    
    # Use LHS for efficient, non-collapsing coverage of the space
    sampler_hover = LatinHypercube(d=len(hover_features), seed=42)
    sample_hover_norm = sampler_hover.random(n=n_samples)
    sample_hover = bounds_hover[:, 0] + sample_hover_norm * (bounds_hover[:, 1] - bounds_hover[:, 0])
    df_sample_hover = pd.DataFrame(sample_hover, columns=hover_features)

    sampler_cruise = LatinHypercube(d=len(cruise_features), seed=42)
    sample_cruise_norm = sampler_cruise.random(n=n_samples)
    sample_cruise = bounds_cruise[:, 0] + sample_cruise_norm * (bounds_cruise[:, 1] - bounds_cruise[:, 0])
    df_sample_cruise = pd.DataFrame(sample_cruise, columns=cruise_features)

    # --- 4. Predict Uncertainty at Each Sample Point ---
    print("--- Predicting model uncertainty across the design space (this may take a moment)... ---")
    
    _, std_hover = gpr_hover.predict(df_sample_hover, return_std=True)
    _, std_cruise = gpr_cruise.predict(df_sample_cruise, return_std=True)

    # --- 5. Calculate and Report the Final Metrics ---
    # We normalize by the median performance to get an intuitive percentage
    median_hover_perf = df_hover_data['performance'].median()
    median_cruise_perf = df_cruise_data['performance'].median()

    mean_uncertainty_hover = np.mean(std_hover)
    mean_uncertainty_cruise = np.mean(std_cruise)

    normalized_uncertainty_hover = (mean_uncertainty_hover / median_hover_perf) * 100
    normalized_uncertainty_cruise = (mean_uncertainty_cruise / median_cruise_perf) * 100

    print("\n" + "="*50)
    print("      Design Space Exploration Analysis")
    print("="*50)
    print(f"\nHOVER MODEL:")
    print(f"  - Median Hover Performance (η): {median_hover_perf:.4f}")
    print(f"  - Mean Predicted Uncertainty (σ_η): {mean_uncertainty_hover:.4f}")
    print(f"  - Mean Normalized Uncertainty: {normalized_uncertainty_hover:.2f}%")

    print(f"\nCRUISE MODEL:")
    print(f"  - Median Cruise Performance (L/D): {median_cruise_perf:.4f}")
    print(f"  - Mean Predicted Uncertainty (σ_L/D): {mean_uncertainty_cruise:.4f}")
    print(f"  - Mean Normalized Uncertainty: {normalized_uncertainty_cruise:.2f}%")
    print("\n" + "="*50)
    print("Interpretation: A value < 5% suggests good exploration.")
    print("A value > 15% suggests significant unexplored regions remain.")
    print("="*50)

if __name__ == "__main__":
    main()