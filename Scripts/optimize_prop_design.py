# optimize_prop_design.py
# Find the model-optimal prop design within geometric bounds and allowed materials/orientations.
# Uses same feature construction as select_next_props/train_prop_eff.

import os, numpy as np, pandas as pd, re
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- paths ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

MASTER_PARQUET = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
OUTPUT_TOPK    = os.path.join(TOOLS_DIR, 'optimal_designs.csv')

# ---------------- config ----------------
# Geometric bounds (same as your plan_02 box)
BOUNDS = {
    'AR':            (6.0, 10.0),
    'lambda':        (0.5, 1.0),
    'aoaRoot (deg)': (10.0, 20.0),
    'aoaTip (deg)':  (3.0, 8.0),
}

# Materials/orientations the optimizer is allowed to choose from
# (PCF = your PAHT-CF; both span_strong FDM)
ALLOWED_CONFIGS = [
    {'material':'PCF', 'process':'FDM', 'orientation':'span_strong'},
    {'material':'PLA', 'process':'FDM', 'orientation':'span_strong'},
]

# TDS flexural moduli (GPa) used to build logE_eff for candidates
TDS = {
    'PLA': {'flexMod_xy (GPA)': 2.36, 'flexMod_z (GPA)': 1.77, 'flexMod_iso (GPA)': np.nan},
    'PCF': {'flexMod_xy (GPA)': 4.23, 'flexMod_z (GPA)': 1.82, 'flexMod_iso (GPA)': np.nan},
    # add resins if you want later, e.g. 'Rigid10K': {'flexMod_iso (GPA)': 10.0}
}

# Search resolution
N_SAMPLES_STAGE1 = 40000   # coarse global random
N_SAMPLES_STAGE2 = 8000    # fine local random around top seeds
N_SEEDS_STAGE2   = 20      # how many top seeds to refine
TOPK_SAVE        = 50      # how many best designs to write to CSV

GEOM_COLS    = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']
CAT_COLS     = ['process','material','orientation']

# ---------------- helpers ----------------
def compute_e_eff_like_process_data(row):
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

def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[GEOM_COLS + ['logE_eff']].astype(float).copy()
    # one-hots
    X_cat = pd.get_dummies(df[CAT_COLS].astype(str), prefix=CAT_COLS, drop_first=False)
    # interactions: logE × each cat dummy
    logE = df['logE_eff'].to_numpy().reshape(-1,1)
    inter_list, inter_cols = [], []
    for col in X_cat.columns:
        v = X_cat[[col]].to_numpy()
        inter_list.append((logE * v).ravel())
        inter_cols.append(f"logE_eff__x__{col}")
    X_inter = (pd.DataFrame(np.column_stack(inter_list), columns=inter_cols, index=df.index)
               if inter_list else pd.DataFrame(index=df.index))
    X = pd.concat([X, X_cat, X_inter], axis=1)
    return X

def fit_ridge_on_tested(master_path):
    if not os.path.exists(master_path):
        raise SystemExit(f"master dataset not found: {master_path}")
    m = pd.read_parquet(master_path)
    need_cols = ['filename','prop_efficiency_mean','logE_eff'] + GEOM_COLS
    for c in need_cols:
        if c not in m.columns:
            raise SystemExit(f"master dataset missing column: {c}")

    # avg η̄ per prop
    g = m.dropna(subset=['prop_efficiency_mean']).groupby('filename')
    table = g['prop_efficiency_mean'].mean().to_frame('eta_mean').reset_index()
    # pull per-prop metadata (first occurrence)
    meta_cols = ['filename','logE_eff'] + GEOM_COLS + [c for c in CAT_COLS if c in m.columns]
    meta = m[meta_cols].drop_duplicates('filename')
    table = table.merge(meta, on='filename', how='left').dropna(subset=['logE_eff']+GEOM_COLS)
    y = table['eta_mean'].clip(0,1).to_numpy(float)

    X_df = build_features_df(table.rename(columns={}))
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  RidgeCV(alphas=np.logspace(-6, 3, 40), cv=min(5, max(3, len(table)//3 if len(table)>=6 else 3)),
                           scoring='neg_mean_absolute_error'))
    ])
    model.fit(X_df.to_numpy(float), y)
    return model, X_df.columns.tolist()

def sample_uniform(n):
    """Uniform random samples inside bounds."""
    out = {}
    for c, (lo, hi) in BOUNDS.items():
        out[c] = np.random.uniform(lo, hi, size=n)
    return pd.DataFrame(out)

def clamp(v, lo, hi): return np.minimum(np.maximum(v, lo), hi)

def refine_around(seeds_df, n_each=400, scale=0.08):
    """Local jitter around seeds; gaussian in each dim, clamped to bounds."""
    rows = []
    for _, s in seeds_df.iterrows():
        B = {}
        for c, (lo, hi) in BOUNDS.items():
            span = hi - lo
            mu   = float(s[c])
            std  = span*scale
            B[c] = clamp(np.random.normal(mu, std, size=n_each), lo, hi)
        rows.append(pd.DataFrame(B))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=GEOM_COLS)

def make_candidates(n1, n2, n_seeds):
    # Stage 1: global
    g1 = sample_uniform(n1)

    # Build cartesian with allowed (process,material,orientation)
    cand_all = []
    for cfg in ALLOWED_CONFIGS:
        df = g1.copy()
        df['process']     = cfg['process']
        df['material']    = cfg['material']
        df['orientation'] = cfg['orientation']
        # supply TDS for E fields
        t = TDS.get(cfg['material'], {})
        df['flexMod (GPA)']     = np.nan
        df['flexMod_xy (GPA)']  = t.get('flexMod_xy (GPA)', np.nan)
        df['flexMod_z (GPA)']   = t.get('flexMod_z (GPA)',  np.nan)
        df['flexMod_iso (GPA)'] = t.get('flexMod_iso (GPA)',np.nan)
        # effective E + logE
        df['E_eff_GPa'] = df.apply(compute_e_eff_like_process_data, axis=1)
        df['logE_eff']  = np.log(np.clip(df['E_eff_GPa']*1e9, 2e8, 5e10))
        cand_all.append(df)
    C1 = pd.concat(cand_all, ignore_index=True)

    return C1, n2, n_seeds

def main():
    np.random.seed(42)

    # 1) Fit exploitation model on tested props
    model, train_cols = fit_ridge_on_tested(MASTER_PARQUET)

    # 2) Stage 1: global sampling + score
    C1, n2, n_seeds = make_candidates(N_SAMPLES_STAGE1, N_SAMPLES_STAGE2, N_SEEDS_STAGE2)
    X1 = build_features_df(C1)
    X1 = X1.reindex(columns=train_cols, fill_value=0.0)
    C1['pred_eta'] = model.predict(X1.to_numpy(float))

    # 3) Stage 2: refine around best seeds
    seeds = (C1.sort_values('pred_eta', ascending=False)
                .groupby(['material','orientation'], as_index=False)
                .head(n_seeds)
                .reset_index(drop=True))
    R_list = []
    for _, s in seeds.iterrows():
        R = refine_around(s[GEOM_COLS].to_frame().T, n_each=max(1, n2//(len(seeds) or 1)))
        R['process']     = s['process']
        R['material']    = s['material']
        R['orientation'] = s['orientation']
        t = TDS.get(s['material'], {})
        R['flexMod (GPA)']     = np.nan
        R['flexMod_xy (GPA)']  = t.get('flexMod_xy (GPA)', np.nan)
        R['flexMod_z (GPA)']   = t.get('flexMod_z (GPA)',  np.nan)
        R['flexMod_iso (GPA)'] = t.get('flexMod_iso (GPA)',np.nan)
        R['E_eff_GPa'] = R.apply(compute_e_eff_like_process_data, axis=1)
        R['logE_eff']  = np.log(np.clip(R['E_eff_GPa']*1e9, 2e8, 5e10))
        R_list.append(R)
    C2 = pd.concat(R_list, ignore_index=True) if R_list else pd.DataFrame(columns=C1.columns)

    if not C2.empty:
        X2 = build_features_df(C2).reindex(columns=train_cols, fill_value=0.0)
        C2['pred_eta'] = model.predict(X2.to_numpy(float))
        C = pd.concat([C1, C2], ignore_index=True)
    else:
        C = C1

    # 4) Pick the best and save top-K
    C = C.sort_values('pred_eta', ascending=False).reset_index(drop=True)
    top = C.iloc[0].to_dict()
    topk = C.head(TOPK_SAVE).copy()
    # cosmetic rounding
    for c in GEOM_COLS + ['E_eff_GPa','pred_eta']:
        if c in topk.columns:
            topk[c] = topk[c].astype(float).round(4)

    os.makedirs(TOOLS_DIR, exist_ok=True)
    topk.to_csv(OUTPUT_TOPK, index=False)

    # 5) Print best
    print("\n=== Model-optimal design (within bounds, allowed materials) ===")
    print(f"pred_eta: {top['pred_eta']:.4f}   material: {top['material']}   orientation: {top['orientation']}   process: {top['process']}")
    print("Geometry:")
    for c in GEOM_COLS:
        print(f"  {c:>12s} = {float(top[c]):.4f}")
    print(f"\nE_eff (GPa): {float(top['E_eff_GPa']):.4f}   logE_eff: {float(top['logE_eff']):.2f}")
    print(f"\nTop {TOPK_SAVE} candidates saved to: {OUTPUT_TOPK}")

if __name__ == "__main__":
    main()
