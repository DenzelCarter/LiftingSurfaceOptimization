# process_data.py
# Clean → bin (150 rpm) → add effective stiffness features (xy/z/isotropic) →
# merge with BEMT → save master_dataset.parquet and report any BEMT gaps.

import os
import re
import numpy as np
import pandas as pd

# ---------------- paths (relative to this file) ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')
DATA_DIR   = os.path.join(PROJ_ROOT, 'Experiment', 'data')

DOE_01 = os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv')
DOE_02 = os.path.join(TOOLS_DIR, 'doe_test_plan_02.csv')
TARE_LOOKUP_FILE = os.path.join(TOOLS_DIR, 'tare_lookup_01.csv')
BEMT_PRED_PATH   = os.path.join(TOOLS_DIR, 'bemt_predictions_final.csv')
OUTPUT_PARQUET   = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
BEMT_MISSING_CSV = os.path.join(TOOLS_DIR, 'bemt_missing_bins.csv')

# ---------------- physics ----------------
SPAN = 0.184
R_HUB = 0.046
RHO_AIR = 1.225
MU_AIR  = 1.81e-5
D = 2.0*(R_HUB + SPAN)
A = np.pi*(D/2.0)**2

# ---------------- data params ----------------
BIN_RPM = 150
TARE_BIN_RPM = 50
MIN_THRUST = 0.1
MIN_RPM = 300
MAX_VIB = 3.0
MIN_COUNT_PER_BIN = 1

RPM_COL   = 'Motor Electrical Speed (RPM)'
THRUST_COL= 'Thrust (N)'
TORQUE_COL= 'Torque (N·m)'
VIB_COL   = 'Vibration (g)'
EPOW_COL  = 'Electrical Power (W)'
MPOW_COL  = 'Mechanical Power (W)'

def bin_nearest(val, w):
    v = np.asarray(val, dtype=float)
    b = np.floor((v + 0.5 * w) / w) * w
    return b.astype(int)

def robust_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _coalesce_cols(df, new_col, candidates, default=np.nan):
    """Create new_col from the first existing candidate, else default."""
    for c in candidates:
        if c in df.columns:
            df[new_col] = df[c]
            return df
    df[new_col] = default
    return df

def main():
    # ---- load DOE ----
    frames = []
    if os.path.exists(DOE_01): frames.append(pd.read_csv(DOE_01))
    if os.path.exists(DOE_02): frames.append(pd.read_csv(DOE_02))
    if not frames:
        raise SystemExit("DOE files not found.")
    doe = pd.concat(frames, ignore_index=True)

    # Back-compat: ensure stiffness metadata columns exist
    # Expected (if provided): 'flexMod_xy (GPA)', 'flexMod_z (GPA)', optional 'flexMod_iso (GPA)'
    # If only 'flexMod (GPA)' exists, use it as xy and iso fallback
    if 'flexMod_xy (GPA)' not in doe.columns:
        doe = _coalesce_cols(doe, 'flexMod_xy (GPA)', ['flexMod (GPA)'], default=np.nan)
    if 'flexMod_z (GPA)' not in doe.columns:
        # if z not given, assume 0.5× xy as a mild anisotropy fallback
        doe['flexMod_z (GPA)'] = np.where(np.isfinite(doe['flexMod_xy (GPA)']), 0.5*doe['flexMod_xy (GPA)'], np.nan)
    if 'flexMod_iso (GPA)' not in doe.columns:
        doe = _coalesce_cols(doe, 'flexMod_iso (GPA)', ['flexMod (GPA)', 'flexMod_xy (GPA)'], default=np.nan)

    # Standardize metadata naming (optional but nice to have)
    # Expected: 'process' in {'FDM','SLA','Resin',...}, 'orientation' in {'span_strong','span_weak','isotropic'}, 'material' string
    if 'process' not in doe.columns:  doe['process'] = 'FDM'
    if 'orientation' not in doe.columns: doe['orientation'] = np.where(doe['process'].isin(['SLA','Resin']), 'isotropic', 'span_strong')
    if 'material' not in doe.columns:
        # try to infer from filename like 'Prop_013_PLA_01.csv' → 'PLA'
        def infer_mat(fn):
            m = re.search(r'Prop_\d+_([A-Za-z0-9\-]+)_\d+', str(fn))
            return m.group(1) if m else 'UNKNOWN'
        doe['material'] = doe.get('filename', pd.Series(dtype=str)).apply(infer_mat)

    # Force isotropic orientation for SLA/Resin even if CSV says otherwise
    doe.loc[doe['process'].isin(['SLA','Resin']), 'orientation'] = 'isotropic'

    # ---- load raw data per prop ----
    all_list = []
    for _, prow in doe.iterrows():
        fname = prow['filename']
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            # attach DOE metadata columns to every row
            for c in doe.columns:
                df[c] = prow[c]
            all_list.append(df)
            print(f"   + Loaded '{fname}'")
        except Exception as e:
            print(f"   - Skipping '{fname}': {e}")
    if not all_list:
        raise SystemExit("No raw data files found that match DOE filenames.")
    master = pd.concat(all_list, ignore_index=True)

    # ---- clean & tare ----
    num_cols = [RPM_COL, THRUST_COL, TORQUE_COL, VIB_COL, EPOW_COL, MPOW_COL]
    master = robust_numeric(master, num_cols)
    master.dropna(subset=num_cols, inplace=True)

    try:
        tare = pd.read_csv(TARE_LOOKUP_FILE)
        if 'rpm_bin_50' not in tare.columns:
            if 'rpm_bin' in tare.columns:
                tare = tare.rename(columns={'rpm_bin': 'rpm_bin_50'})
            elif 'rpm' in tare.columns:
                tare['rpm_bin_50'] = bin_nearest(tare['rpm'].to_numpy(), TARE_BIN_RPM)
            else:
                raise KeyError("tare file has no 'rpm_bin_50', 'rpm_bin', or 'rpm' column.")
        if 'thrust_tare' not in tare.columns: tare['thrust_tare'] = 0.0
        if 'torque_tare' not in tare.columns: tare['torque_tare'] = 0.0

        master['rpm_bin_50'] = bin_nearest(master[RPM_COL].to_numpy(), TARE_BIN_RPM)
        master = master.merge(tare[['rpm_bin_50','thrust_tare','torque_tare']], on='rpm_bin_50', how='left')
        master[['thrust_tare','torque_tare']] = master[['thrust_tare','torque_tare']].fillna(0.0)
        master[THRUST_COL] -= master['thrust_tare']
        master[TORQUE_COL] -= master['torque_tare']
    except Exception as e:
        print(f"   - Warning: tare correction skipped: {e}")

    master = master[(master[THRUST_COL] > MIN_THRUST) &
                    (master[RPM_COL]    > MIN_RPM) &
                    (master[VIB_COL]    < MAX_VIB)].copy()

    # ---- aerodynamic features & efficiencies ----
    master['A'] = A
    master['ideal_power']    = np.sqrt(np.clip(master[THRUST_COL], 0, None)**3 / (2 * RHO_AIR * A))
    master['prop_efficiency']= master['ideal_power'] / master[MPOW_COL]
    master['motor_efficiency']= master[MPOW_COL] / master[EPOW_COL]
    master['system_efficiency']= master['ideal_power'] / master[EPOW_COL]

    master = master[(master['prop_efficiency']  > 0) & (master['prop_efficiency'] <= 1.0) &
                    (master['motor_efficiency'] > 0) & (master['motor_efficiency'] <= 1.0)].copy()

    root_chord = (2*SPAN) / (master['AR'] * (1 + master['lambda']))
    tip_chord  = master['lambda'] * root_chord
    chord_75   = root_chord + 0.75*(tip_chord - root_chord)
    r_75       = R_HUB + 0.75*SPAN
    omega      = master[RPM_COL] * (2*np.pi/60)
    v_tan_75   = omega * r_75
    master['reynolds_number'] = (RHO_AIR * v_tan_75 * chord_75) / MU_AIR
    master['log_reynolds_number'] = np.log(master['reynolds_number'].clip(lower=1.0))
    avg_chord  = 0.5*(root_chord + tip_chord)
    master['avg_thickness_m'] = 0.12 * avg_chord

    # ---- effective flexural modulus based on orientation/process ----
    # If isotropic (resin/SLA or orientation=='isotropic'): use ISO if present, else xy
    # For FDM with explicit orientation:
    #   - 'span_strong' → use xy
    #   - 'span_weak'   → use z
    def compute_E_eff(row):
        orient = str(row.get('orientation', 'span_strong')).lower()
        proc   = str(row.get('process', 'FDM'))
        Exy = row.get('flexMod_xy (GPA)', np.nan)
        Ez  = row.get('flexMod_z (GPA)',  np.nan)
        Eiso= row.get('flexMod_iso (GPA)', np.nan)
        if proc in ['SLA','Resin'] or orient == 'isotropic':
            return Eiso if np.isfinite(Eiso) else (Exy if np.isfinite(Exy) else np.nan)
        if orient == 'span_weak':
            return Ez if np.isfinite(Ez) else (0.5*Exy if np.isfinite(Exy) else np.nan)
        # default span_strong
        return Exy if np.isfinite(Exy) else (Eiso if np.isfinite(Eiso) else np.nan)

    master['E_eff_GPa'] = master.apply(compute_E_eff, axis=1)
    master['logE_eff']  = np.log(np.clip(master['E_eff_GPa']*1e9, 2e8, 5e10))  # stabilize extremes
    # simple deflection proxy (per-row), later averaged per (prop, rpm_bin)
    omega_row = omega
    master['C_deflect'] = (omega_row**2) * (master['avg_thickness_m']**3) * (SPAN**4) / np.clip(master['E_eff_GPa']*1e9, 1e6, None)

    # ---- averaging in 150-rpm bins ----
    master['rpm_bin'] = bin_nearest(master[RPM_COL].to_numpy(), BIN_RPM)

    if MIN_COUNT_PER_BIN > 1:
        counts = master.groupby(['filename','rpm_bin'])[THRUST_COL].transform('count')
        master = master[counts >= MIN_COUNT_PER_BIN].copy()

    # aggregate numeric means per (prop, rpm_bin)
    numeric_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    for drop_key in ['rpm_bin', 'rpm_bin_50']:
        if drop_key in numeric_cols:
            numeric_cols.remove(drop_key)

    # carry id/metadata (include new stiffness + process/material/orientation)
    id_cols = ['filename','AR','lambda','aoaRoot (deg)','aoaTip (deg)',
               'flexMod_xy (GPA)','flexMod_z (GPA)','flexMod_iso (GPA)',
               'E_eff_GPa','logE_eff','process','orientation','material']

    agg = (master
           .groupby(['filename','rpm_bin'])[numeric_cols].mean().reset_index()
           .merge(master[id_cols].drop_duplicates('filename'),
                  on='filename', how='left', suffixes=('','_y')))
    agg.drop(columns=[c for c in agg.columns if c.endswith('_y')], inplace=True)

    # rename a few means for clarity
    if 'prop_efficiency' in agg.columns:
        agg.rename(columns={'prop_efficiency':'prop_efficiency_mean'}, inplace=True)
    if 'reynolds_number' in agg.columns:
        agg.rename(columns={'reynolds_number':'reynolds_number_mean'}, inplace=True)
    if 'log_reynolds_number' in agg.columns:
        agg.rename(columns={'log_reynolds_number':'log_reynolds_number_mean'}, inplace=True)

    # ---- merge BEMT on (filename, rpm_bin) ----
    bemt = pd.read_csv(BEMT_PRED_PATH)
    agg['rpm_bin'] = agg['rpm_bin'].astype(int)
    if 'rpm_bemt' not in bemt.columns:
        if 'rpm' in bemt.columns:
            bemt['rpm_bemt'] = bin_nearest(bemt['rpm'].to_numpy(), BIN_RPM)
        else:
            raise KeyError("bemt_predictions_final.csv has neither 'rpm_bemt' nor 'rpm'.")
    bemt['rpm_bemt'] = bemt['rpm_bemt'].astype(int)

    df = agg.merge(bemt, left_on=['filename','rpm_bin'], right_on=['filename','rpm_bemt'], how='left')

    # clamp BEMT efficiency defensively to [0,1]
    if 'prop_eff_bemt' in df.columns:
        df['prop_eff_bemt'] = df['prop_eff_bemt'].clip(lower=0.0, upper=1.0)

    # ---- report any missing BEMT rows ----
    miss = df['prop_eff_bemt'].isna()
    if miss.any():
        r_by_prop = bemt.groupby('filename')['rpm_bemt'].agg(['min','max']).reset_index()
        dfm = df.loc[miss, ['filename','rpm_bin']].merge(r_by_prop, on='filename', how='left')
        dfm['reason'] = np.where(dfm['rpm_bin'] < dfm['min'], 'below_bemt_min',
                          np.where(dfm['rpm_bin'] > dfm['max'], 'above_bemt_max', 'no_match'))
        dfm.to_csv(BEMT_MISSING_CSV, index=False)
        print(f"Warning: {miss.sum()} bins have no BEMT match. See {BEMT_MISSING_CSV}")
    else:
        print("BEMT merge coverage: 100% (no missing bins).")

    os.makedirs(TOOLS_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved master dataset with {len(df)} rows to '{OUTPUT_PARQUET}'")
    print(f"Binning: {BIN_RPM} rpm; per-prop bins present: median={int(df.groupby('filename').size().median())}")

if __name__ == "__main__":
    main()
