import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os

# ===================================================================
# --- 1. SETUP: Define Constants and File Paths ---
# ===================================================================
TOOLS_DIR = 'Experiment/tools'
DOE_FILE_PATH = os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv')
AIRFOIL_DATA_PATH = os.path.join(TOOLS_DIR, 'naca0012_all_re.csv')
OUTPUT_FILE_PATH = os.path.join(TOOLS_DIR, 'bemt_predictions_final.csv')

# --- Physical and Geometric Constants ---
RHO = 1.225
MU_AIR = 1.81e-5
R_HUB = 0.046
SPAN_BLADE = 0.184
R_TIP = R_HUB + SPAN_BLADE
NUM_BLADES = 2
NUM_ELEMENTS = 20

# ===================================================================
# --- 2. Load Airfoil Data ---
# ===================================================================
def load_airfoil_data_2d(filepath):
    """Loads combined airfoil data and finds the valid alpha and Re ranges."""
    df = pd.read_csv(filepath)
    points = df[['reynolds', 'alpha']].values
    cl_values = df['cl'].values
    cd_values = df['cd'].values
    
    alpha_min_deg = df['alpha'].min()
    alpha_max_deg = df['alpha'].max()
    re_min = df['reynolds'].min()
    re_max = df['reynolds'].max()
    
    print(f"2D airfoil data loaded. Valid alpha: {alpha_min_deg} to {alpha_max_deg} deg, Valid Re: {re_min} to {re_max}.")
    return points, cl_values, cd_values, alpha_min_deg, alpha_max_deg, re_min, re_max

# ===================================================================
# --- 3. Main BEMT Calculation (with Induced Velocity Solver) ---
# ===================================================================
def run_bemt_with_solver(prop_params, rpm, interp_points, cl_vals, cd_vals, alpha_min, alpha_max, re_min, re_max):
    """Calculates Thrust and Torque using a BEMT model with an iterative solver for induced velocity."""
    r_stations = np.linspace(R_HUB, R_TIP, NUM_ELEMENTS + 1)
    r_centers = (r_stations[:-1] + r_stations[1:]) / 2
    dr = r_stations[1] - r_stations[0]
    
    root_chord = (2 * SPAN_BLADE) / (prop_params['AR'] * (1 + prop_params['lambda']))
    tip_chord = prop_params['lambda'] * root_chord
    local_chord = root_chord + (tip_chord - root_chord) * (r_centers - R_HUB) / SPAN_BLADE
    local_twist = np.deg2rad(prop_params['aoaRoot (deg)'] + (prop_params['aoaTip (deg)'] - prop_params['aoaRoot (deg)']) * (r_centers - R_HUB) / SPAN_BLADE)
    
    omega = rpm * (2 * np.pi / 60)
    V_tan = omega * r_centers
    
    # --- Iterative Solver for Induced Velocity ---
    v_induced = np.zeros_like(r_centers) # Initial guess for induced velocity is 0
    for _ in range(100): # Iterate to find a stable solution
        V_inflow = v_induced # For static case, inflow is just induced velocity
        phi = np.arctan2(V_inflow, V_tan)
        phi[np.isnan(phi)] = 0
        
        alpha_rad = local_twist - phi
        alpha_deg = np.rad2deg(alpha_rad)
        alpha_deg_clamped = np.clip(alpha_deg, alpha_min, alpha_max)
        
        V_res = np.sqrt(V_inflow**2 + V_tan**2)
        local_re = (RHO * V_res * local_chord) / MU_AIR
        local_re_clamped = np.clip(local_re, re_min, re_max)
        
        interp_queries = np.vstack((local_re_clamped, alpha_deg_clamped)).T
        cl = griddata(interp_points, cl_vals, interp_queries, method='linear', fill_value=0)
        cd = griddata(interp_points, cd_vals, interp_queries, method='linear', fill_value=0)
        
        L = 0.5 * RHO * V_res**2 * local_chord * cl
        D = 0.5 * RHO * V_res**2 * local_chord * cd
        
        f_tip = (NUM_BLADES / 2) * (R_TIP - r_centers) / (r_centers * np.sin(phi, where=phi!=0, out=np.full_like(phi, 1e-6)))
        F = (2 / np.pi) * np.arccos(np.exp(-f_tip))
        F = np.nan_to_num(F, nan=1.0)
        
        dT_elements = (L * np.cos(phi) - D * np.sin(phi))
        
        # New induced velocity from momentum theory
        v_induced_new = (NUM_BLADES * local_chord * (cl * np.cos(phi) + cd * np.sin(phi)) * V_res) / (8 * np.pi * r_centers * F * np.sin(phi, where=phi!=0, out=np.full_like(phi, 1e-6)))
        
        # Relaxation factor to stabilize convergence
        v_induced = 0.5 * v_induced + 0.5 * v_induced_new
        v_induced = np.nan_to_num(v_induced, nan=0.0)

    # Final calculation with converged induced velocity
    dQ_elements = (L * np.sin(phi) + D * np.cos(phi)) * r_centers * dr * F
    total_thrust = np.sum(dT_elements * dr) * NUM_BLADES
    total_torque = np.sum(dQ_elements) * NUM_BLADES
    
    if total_torque < 0:
        total_torque, total_thrust = 0, 0

    return total_thrust, total_torque

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    doe_df = pd.read_csv(DOE_FILE_PATH)
    points, cl_values, cd_values, alpha_min_deg, alpha_max_deg, re_min, re_max = load_airfoil_data_2d(AIRFOIL_DATA_PATH)
    results = []
    rpm_range = np.arange(500, 2100, 100)
    
    print("\nRunning FINAL BEMT simulations for all props...")
    for index, row in doe_df.iterrows():
        for rpm in rpm_range:
            thrust, torque = run_bemt_with_solver(row, rpm, points, cl_values, cd_values, alpha_min_deg, alpha_max_deg, re_min, re_max)
            
            A = np.pi * R_TIP**2
            P_mech = torque * (rpm * 2 * np.pi / 60)
            P_ideal = np.sqrt(thrust**3 / (2 * RHO * A)) if thrust > 0 else 0
            
            prop_eff_bemt = P_ideal / P_mech if P_mech > 0 else 0

            results.append({
                'filename': row['filename'],
                'rpm_bemt': rpm,
                'prop_eff_bemt': prop_eff_bemt
            })
        print(f"  - Finished simulation for {row['filename']}")
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nBEMT predictions saved to '{OUTPUT_FILE_PATH}'")