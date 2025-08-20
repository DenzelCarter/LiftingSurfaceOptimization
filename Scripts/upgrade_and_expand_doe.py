# upgrade_and_expand_doe.py
# - Rewrites DOE CSVs sorted by prop number.
# - Normalizes PAHT-CF -> PCF and filenames to 'Prop_###_PCF_##.csv'.
# - Ensures PLA/span_strong and PCF/span_strong exist for every base prop.
# - Fills PCF XY=4.23 GPa, Z=1.82 GPa (others unchanged).
# - No material DB used.

import os, re, shutil
import numpy as np
import pandas as pd

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

DOE_FILES = [
    os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv'),
    os.path.join(TOOLS_DIR, 'doe_test_plan_02.csv'),
]

GEOM_BASE = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']

# desired variants to guarantee exist
DESIRED_CONFIGS = [
    {'material': 'PLA', 'orientation': 'span_strong', 'process': 'FDM'},
    {'material': 'PCF', 'orientation': 'span_strong', 'process': 'FDM'},  # PCF = PAHT-CF
]

# PCF mechanicals (GPa)
PCF_XY = 4.23
PCF_Z  = 1.82

# ---------- helpers ----------
def ensure_cols(df):
    for c in ['process','material','orientation','flexMod (GPA)','flexMod_xy (GPA)','flexMod_z (GPA)','flexMod_iso (GPA)']:
        if c not in df.columns: df[c] = np.nan
    return df

def base_prop_id(fn: str) -> str:
    m = re.search(r'(Prop|prop)_(\d+)', str(fn))
    return f"Prop_{int(m.group(2)):03d}" if m else str(fn).split('.')[0]

def base_prop_num(fn: str) -> int:
    m = re.search(r'(Prop|prop)_(\d+)', str(fn))
    return int(m.group(2)) if m else 10**9  # push non-matching to end

def next_index_for(base_num: int, material_token: str, df: pd.DataFrame) -> int:
    # e.g., material_token = 'PLA' or 'PCF'; case-insensitive match
    pat = re.compile(rf'^(?:Prop|prop)_{base_num:03d}_{material_token}_(\d+)\.csv$', re.IGNORECASE)
    idx = 0
    for f in df['filename'].astype(str):
        m = pat.match(f)
        if m:
            idx = max(idx, int(m.group(1)))
    return idx + 1

def normalize_material(val: str) -> str:
    v = (val or "").strip().upper()
    if v in ['PAHT-CF','PAHTCF','PCF']: return 'PCF'
    return v if v else 'PLA'

def infer_material_from_filename(fn: str) -> str:
    m = re.search(r'(?:Prop|prop)_\d+_([A-Za-z0-9\-]+)_\d+\.csv', str(fn))
    return normalize_material(m.group(1)) if m else np.nan

def compute_e_eff_row(row):
    orient = str(row.get('orientation','')).lower()
    proc   = str(row.get('process','')).lower()
    Exy = row.get('flexMod_xy (GPA)', np.nan)
    Ez  = row.get('flexMod_z (GPA)',  np.nan)
    Eiso= row.get('flexMod_iso (GPA)',np.nan)
    E   = row.get('flexMod (GPA)',    np.nan)
    if proc in ['sla','resin'] or orient == 'isotropic':
        return Eiso if np.isfinite(Eiso) else (Exy if np.isfinite(Exy) else E)
    if orient == 'span_weak':
        return Ez if np.isfinite(Ez) else (0.5*Exy if np.isfinite(Exy) else E)
    # default span_strong
    return Exy if np.isfinite(Exy) else (Eiso if np.isfinite(Eiso) else E)

def pcf_filename(base_num: int, idx: int) -> str:
    return f"Prop_{base_num:03d}_PCF_{idx:02d}.csv"  # UPPERCASE token to match others

def expand_dataframe(df):
    df = df.copy()
    df = ensure_cols(df)

    # normalize material column & infer when missing
    df['material'] = df['material'].fillna(df['filename'].apply(infer_material_from_filename))
    df['material'] = df['material'].apply(normalize_material)

    # default process/orientation
    df['process'] = df['process'].fillna('FDM')
    df['orientation'] = df.apply(
        lambda r: 'isotropic' if str(r.get('process','')).upper() in ['SLA','RESIN'] else (r.get('orientation') or 'span_strong'),
        axis=1
    )

    # set PCF XY/Z where missing
    is_pcf = df['material'].astype(str).str.upper().eq('PCF')
    df.loc[is_pcf & df['flexMod_xy (GPA)'].isna(), 'flexMod_xy (GPA)'] = PCF_XY
    df.loc[is_pcf & df['flexMod_z (GPA)'].isna(),  'flexMod_z (GPA)']  = PCF_Z

    # effective modulus + logE
    df['E_eff_GPa'] = df.apply(compute_e_eff_row, axis=1)
    df['logE_eff']  = np.log(np.clip(df['E_eff_GPa']*1e9, 2e8, 5e10))

    # build base table
    df['base_num'] = df['filename'].apply(base_prop_num)
    bases = df.drop_duplicates(subset=['base_num'])[ ['base_num'] + GEOM_BASE ]

    new_rows = []
    for _, brow in bases.iterrows():
        base_num = int(brow['base_num'])
        geom_vals = {c: brow[c] for c in GEOM_BASE}

        existing = df[df['filename'].str.contains(rf'^(?:Prop|prop)_{base_num:03d}_', regex=True, na=False)]

        for cfg in DESIRED_CONFIGS:
            mat = cfg['material']      # 'PLA' or 'PCF'
            orient = cfg['orientation']# 'span_strong'
            proc = cfg['process']      # 'FDM'

            # already exists?
            already = existing[
                (existing['material'].astype(str).str.upper() == mat) &
                (existing['orientation'].astype(str) == orient)
            ]
            if not already.empty:
                continue

            # filename + index style
            if mat == 'PCF':
                nn = next_index_for(base_num, 'PCF', df)
                fname = pcf_filename(base_num, nn)
            else:  # PLA
                nn = next_index_for(base_num, 'PLA', df)
                fname = f"Prop_{base_num:03d}_PLA_{nn:02d}.csv"

            row = {'filename': fname, **geom_vals,
                   'process': proc, 'material': mat, 'orientation': orient}

            # carry a fallback flexMod if any row for this base had one
            base_any = existing.head(1)
            fallback_E = float(base_any['flexMod (GPA)'].iloc[0]) if (not base_any.empty and 'flexMod (GPA)' in base_any.columns) else np.nan
            row['flexMod (GPA)'] = fallback_E

            # set PCF XY/Z numbers (PLA left as-is)
            if mat == 'PCF':
                row['flexMod_xy (GPA)'] = PCF_XY
                row['flexMod_z (GPA)']  = PCF_Z
                row['flexMod_iso (GPA)'] = np.nan
            else:
                row['flexMod_xy (GPA)'] = np.nan
                row['flexMod_z (GPA)']  = np.nan
                row['flexMod_iso (GPA)']= np.nan

            tmp = pd.Series(row)
            row['E_eff_GPa'] = compute_e_eff_row(tmp)
            row['logE_eff']  = np.log(np.clip(row['E_eff_GPa']*1e9, 2e8, 5e10)) if pd.notna(row['E_eff_GPa']) else np.nan
            new_rows.append(row)

    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # normalize *existing* PCF-like names to 'Prop_###_PCF_##.csv'
    def normalize_pcf_filename(fn: str) -> str:
        # capture number + index from any old PCF variant (pcf/PAHT-CF/pahtcf)
        m = re.search(r'(?:Prop|prop)_(\d+)_([A-Za-z\-]+)_(\d+)\.csv', str(fn))
        if not m: return fn
        num = int(m.group(1))
        mat_tok = m.group(2).upper()
        idx = int(m.group(3))
        if mat_tok in ['PCF','PAHT-CF','PAHTCF']:
            return pcf_filename(num, idx)
        return fn

    df_out['filename'] = df_out['filename'].apply(normalize_pcf_filename)

    # sort by numeric prop number, then filename; drop duplicates on filename
    df_out['base_num'] = df_out['filename'].apply(base_prop_num)
    df_out = (df_out
              .sort_values(['base_num','filename'])
              .drop(columns=['base_num'])
              .drop_duplicates(subset=['filename'])
              .reset_index(drop=True))

    return df_out, new_rows

def main():
    for path in DOE_FILES:
        if not os.path.exists(path):
            print(f"Skip (missing): {path}")
            continue
        print(f"\nUpgrading & expanding (sorted): {path}")
        df = pd.read_csv(path)
        df_out, new_rows = expand_dataframe(df)

        # backup then write
        backup = path + ".bak"
        if not os.path.exists(backup):
            shutil.copyfile(path, backup)
            print(f"  Backup written: {backup}")
        df_out.to_csv(path, index=False)
        print(f"  Wrote {len(df_out)} rows (added {len(new_rows)}). Sorted by prop number.")
        if new_rows:
            print("  New examples:")
            print(pd.DataFrame(new_rows).head(min(5, len(new_rows))).to_string(index=False))

if __name__ == "__main__":
    main()
