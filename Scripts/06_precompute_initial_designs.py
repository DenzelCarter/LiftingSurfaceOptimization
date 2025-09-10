# Scripts/06_precompute_initial_designs.py
# This script performs the computationally intensive task of finding the
# optimal operational parameters for every unique initial geometry.
# The results are saved to a CSV file for fast access by the plotting script.

import pandas as pd
from scipy.optimize import differential_evolution
from joblib import load
from pathlib import Path
from tqdm import tqdm # For a progress bar
import path_utils

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # --- 1. Load Data and Models ---
    print("--- Loading all data and trained models... ---")
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
    
    geometry_cols = ['AR', 'lambda', 'twist (deg)']
    unique_geometries = df_full[geometry_cols].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_geometries)} unique initial geometries to evaluate.")

    # --- 2. Optimize Operational Parameters for EACH Initial Geometry ---
    print("\n--- Optimizing operational conditions for each initial design (this may take a while)... ---")

    op_bounds = [
        (df_hover_data['aoa_root (deg)'].min(), df_hover_data['aoa_root (deg)'].max()),
        (df_hover_data['op_speed'].min(), df_hover_data['op_speed'].max()),
        (df_cruise_data['aoa_root (deg)'].min(), df_cruise_data['aoa_root (deg)'].max()),
        (df_cruise_data['op_speed'].min(), df_cruise_data['op_speed'].max())
    ]

    def op_objective(X_op, fixed_geom):
        AR, lam, twist = fixed_geom
        aoa_h, rpm_h, aoa_c, v_c = X_op
        
        df_h = pd.DataFrame([[AR, lam, aoa_h, twist, rpm_h]], columns=gpr_hover.feature_names_in_)
        df_c = pd.DataFrame([[AR, lam, aoa_c, twist, v_c]], columns=gpr_cruise.feature_names_in_)
        
        eta = gpr_hover.predict(df_h)[0]
        ld = gpr_cruise.predict(df_c)[0]
        
        # A simple objective to find the best combined performance point
        return -(eta + ld) 

    initial_opt_results = []
    for index, geom in tqdm(unique_geometries.iterrows(), total=len(unique_geometries), desc="Optimizing Initial Designs"):
        fixed_geom_values = geom.values
        res = differential_evolution(op_objective, op_bounds, args=(fixed_geom_values,), seed=42, maxiter=50, popsize=15)
        
        AR, lam, twist = fixed_geom_values
        aoa_h, rpm_h, aoa_c, v_c = res.x
        df_h = pd.DataFrame([[AR, lam, aoa_h, twist, rpm_h]], columns=gpr_hover.feature_names_in_)
        df_c = pd.DataFrame([[AR, lam, aoa_c, twist, v_c]], columns=gpr_cruise.feature_names_in_)
        
        initial_opt_results.append({
            'eta_opt': gpr_hover.predict(df_h)[0],
            'ld_opt': gpr_cruise.predict(df_c)[0]
        })

    df_initial_optimized = pd.DataFrame(initial_opt_results)

    # --- 3. Save Pre-computed Results ---
    tables_dir = Path(P["outputs_tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "06_initial_designs_optimized.csv"
    df_initial_optimized.to_csv(output_path, index=False, float_format="%.4f")
    
    print(f"\nSuccessfully pre-computed and saved initial design performance to:\n{output_path}")

if __name__ == "__main__":
    main()