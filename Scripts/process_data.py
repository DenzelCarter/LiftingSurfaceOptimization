# process_data.py
# Clean → bin (150 rpm, robust nearest) → merge with BEMT → save master_dataset.parquet
# Also logs any BEMT gaps to bemt_missing_bins.csv.
# Fixes:
#   - no walrus operator in DOE_02 check
#   - avoids double-inserting rpm_bin during aggregation
#   - clamps merged prop_eff_bemt to [0,1] defensively

import os
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

def main():
    # ---- load DOE ----
    frames = []
    if os.path.exists(DOE_01): frames.append(pd.read_csv(DOE_01))
    if os.path.exists(DOE_02): frames.append(pd.read_csv(DOE_02))  # <= fixed (no walrus)
    if not frames:
        raise SystemExit("DOE files not found.")
    doe = pd.concat(frames, ignore_index=True)

    # ---- load raw data per prop ----
    all_list = []
    for _, prow in doe.iterrows():
        fname = prow['filename']
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p): 
            continue
        try:
            df = pd.read_csv(p)
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

    # ---- features & efficiencies ----
    master['A'] = A
    master['ideal_power'] = np.sqrt(np.clip(master[THRUST_COL], 0, None)**3 / (2 * RHO_AIR * A))
    master['prop_efficiency'] = master['ideal_power'] / master[MPOW_COL]
    master['motor_efficiency'] = master[MPOW_COL] / master[EPOW_COL]
    master['system_efficiency'] = master['ideal_power'] / master[EPOW_COL]

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

    id_cols = ['filename','AR','lambda','aoaRoot (deg)','aoaTip (deg)','flexMod (GPA)']
    agg = master.groupby(['filename','rpm_bin'])[numeric_cols].mean().reset_index()
    agg = agg.merge(master[id_cols].drop_duplicates('filename'), on='filename', how='left', suffixes=('','_y'))
    agg.drop(columns=[c for c in agg.columns if c.endswith('_y')], inplace=True)

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

    # ---- report any missing BEMT rows (should be none) ----
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
