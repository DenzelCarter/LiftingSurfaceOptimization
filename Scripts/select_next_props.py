# select_next_props.py
# Recommend next DOE props to test (exploit + explore).
# - Exploit: ML prediction on geometry+logE (+ cats), blended with priors.
# - Priors: BEMT (avg across rpm) and CFD single-RPM via a geometry surrogate + calibration.
# - Explore: distance to tested set in standardized numeric space.

import os, re
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------- Config ----------
K_RECOMMEND = 6
W_EXPLOIT = 0.70
W_EXPLORE = 0.30

USE_BEMT_PRIOR    = True
BEMT_WEIGHT       = 0.15

USE_CFD_SINGLE    = True
CFD_SINGLE_WEIGHT = 0.25   # remainder goes to ML prediction

PREFER_PLAN_02 = True
INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS = True
PLAN_02_BOUNDS = {'AR': (6.0,10.0), 'lambda': (0.5,1.0), 'aoaRoot (deg)': (10.0,20.0), 'aoaTip (deg)': (3.0,8.0)}

ALLOWED_MATERIALS    = ['PLA','PCF']
ALLOWED_ORIENTATIONS = ['span_strong']

USE_CATEGORICALS_IN_MODEL = True
INCLUDE_TWIST_SLOPE = False

FEATURE_WEIGHTS = {'AR':1.0,'lambda':1.0,'aoaRoot (deg)':1.0,'aoaTip (deg)':1.0,'logE_eff':1.0}
MIN_DIST_BETWEEN_RECS = 0.30

# ---------- Paths ----------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

DOE_FILES = [os.path.join(TOOLS_DIR,'doe_test_plan_01.csv'),
             os.path.join(TOOLS_DIR,'doe_test_plan_02.csv')]
MASTER_PARQUET = os.path.join(TOOLS_DIR,'master_dataset.parquet')
BEMT_CSV       = os.path.join(TOOLS_DIR,'bemt_predictions_final.csv')
CFD_SINGLE_CSV = os.path.join(TOOLS_DIR,'cfd_single_prior.csv')
OUTPUT_RECS    = os.path.join(TOOLS_DIR,'next_props_recommendations.csv')

GEOM_BASE = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']
DOE_STIFF_COLS = ['flexMod (GPA)','flexMod_xy (GPA)','flexMod_z (GPA)','flexMod_iso (GPA)']
CAT_COLS = ['process','material','orientation']

# ---------- Helpers ----------
def standardize(X):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd

def z01(v):
    v = np.asarray(v, float); lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12: return np.zeros_like(v)
    return (v - lo) / (hi - lo)

def pairwise_min_dist(x, Y):
    if Y.size == 0: return float(np.linalg.norm(x, ord=2))
    d = np.sqrt(np.sum((Y - x[None,:])**2, axis=1))
    return float(np.min(d))

def greedy_maximin_indices(Xw, idx_tested, idx_pool, k, min_sep=0.0):
    selected = []; X_tested = Xw[idx_tested,:] if len(idx_tested)>0 else np.zeros((0,Xw.shape[1])); pool=list(idx_pool)
    for _ in range(k):
        if not pool: break
        best_i, best = None, -1.0
        for i in pool:
            dmin = pairwise_min_dist(Xw[i,:], X_tested)
            if dmin > best: best_i, best = i, dmin
        if best_i is None: break
        selected.append((best_i,best))
        X_tested = Xw[best_i,:][None,:] if X_tested.size==0 else np.vstack([X_tested, Xw[best_i,:]])
        pool.remove(best_i)
        if min_sep>0.0:
            pool = [j for j in pool if np.linalg.norm(Xw[j,:]-Xw[best_i,:]) >= min_sep]
    return selected

def within_bounds(row, bounds):
    for c,(lo,hi) in bounds.items():
        v = float(row[c]); 
        if not (lo <= v <= hi): return False
    return True

def compute_e_eff_like_process_data(row):
    orient = str(row.get('orientation','')).lower(); proc = str(row.get('process','')).lower()
    Exy = row.get('flexMod_xy (GPA)', np.nan); Ez = row.get('flexMod_z (GPA)', np.nan)
    Eiso= row.get('flexMod_iso (GPA)', np.nan); E = row.get('flexMod (GPA)', np.nan)
    if proc in ['sla','resin'] or orient == 'isotropic':
        return Eiso if np.isfinite(Eiso) else (Exy if np.isfinite(Exy) else E)
    if orient == 'span_weak':
        return Ez if np.isfinite(Ez) else (0.5*Exy if np.isfinite(Exy) else E)
    return Exy if np.isfinite(Exy) else (Eiso if np.isfinite(Eiso) else E)

def load_doe(paths):
    frames=[]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p); df['__plan'] = os.path.basename(p); frames.append(df)
        else:
            print(f"Warning: DOE file not found: {p}")
    if not frames: raise SystemExit("No DOE CSVs found.")
    df_all = pd.concat(frames, ignore_index=True)
    if INCLUDE_TWIST_SLOPE and 'twist_slope' not in df_all.columns:
        if all(c in df_all.columns for c in ['aoaTip (deg)','aoaRoot (deg)']):
            df_all['twist_slope'] = df_all['aoaTip (deg)'] - df_all['aoaRoot (deg)']
    if 'logE_eff' not in df_all.columns:
        for c in DOE_STIFF_COLS:
            if c not in df_all.columns: df_all[c] = np.nan
        df_all['E_eff_GPa'] = df_all.apply(compute_e_eff_like_process_data, axis=1)
        df_all['logE_eff']  = np.log(np.clip(df_all['E_eff_GPa']*1e9, 2e8, 5e10))
    return df_all

def filter_doe(df, prefer_plan_02=True, include_plan01_if_in_box=True, bounds=None):
    if prefer_plan_02:
        df2 = df[df['__plan'].str.contains('doe_test_plan_02', case=False, regex=False)].copy()
    else:
        df2 = pd.DataFrame(columns=df.columns)
    if include_plan01_if_in_box and bounds is not None:
        df1 = df[df['__plan'].str.contains('doe_test_plan_01', case=False, regex=False)].copy()
        df1 = df1[df1.apply(lambda r: within_bounds(r, bounds), axis=1)]
        df_out = pd.concat([df2, df1], ignore_index=True)
    else:
        df_out = df2 if prefer_plan_02 else df.copy()
    if ALLOWED_MATERIALS and 'material' in df_out.columns:
        df_out = df_out[df_out['material'].astype(str).isin(ALLOWED_MATERIALS)]
    if ALLOWED_ORIENTATIONS and 'orientation' in df_out.columns:
        df_out = df_out[df_out['orientation'].astype(str).isin(ALLOWED_ORIENTATIONS)]
    return df_out.drop_duplicates(subset=['filename']).reset_index(drop=True)

def load_tested_props(master_parquet):
    if not os.path.exists(master_parquet):
        print(f"Note: master parquet not found at {master_parquet}. Assuming none tested yet.")
        return set(), None
    m = pd.read_parquet(master_parquet)
    if 'filename' not in m.columns or 'prop_efficiency_mean' not in m.columns:
        print("Note: master parquet missing needed columns. Assuming none tested yet.")
        return set(), None
    g = m.dropna(subset=['prop_efficiency_mean']).groupby('filename')
    table = g['prop_efficiency_mean'].mean().to_frame('eta_mean').reset_index()
    keep_cols = ['filename'] + GEOM_BASE + ['logE_eff'] + [c for c in CAT_COLS if c in m.columns]
    geom = m[keep_cols].drop_duplicates('filename')
    table = table.merge(geom, on='filename', how='left').dropna(subset=GEOM_BASE+['logE_eff'])
    table['eta_mean'] = table['eta_mean'].clip(0,1)
    return set(map(str, table['filename'].unique())), table

def build_exploit_features_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[GEOM_BASE + ['logE_eff']].astype(float).copy()
    if USE_CATEGORICALS_IN_MODEL:
        cats_present = [c for c in CAT_COLS if c in df.columns]
        if cats_present:
            X_cat = pd.get_dummies(df[cats_present].astype(str), prefix=cats_present, drop_first=False)
            X = pd.concat([X, X_cat], axis=1)
            logE = df['logE_eff'].to_numpy().reshape(-1,1)
            for col in X_cat.columns:
                X[f'logE_eff__x__{col}'] = (logE * X_cat[[col]].to_numpy()).ravel()
    return X

def ridge_pred(Xtr, ytr, Xte):
    n = Xtr.shape[0]; cv_k = min(5, max(3, n)) if n > 2 else 3
    model = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-6,3,20), cv=cv_k, scoring='neg_mean_absolute_error'))])
    model.fit(Xtr, ytr); return model.predict(Xte)

def bemt_avg_by_prop(bemt_csv):
    if not os.path.exists(bemt_csv): return None
    b = pd.read_csv(bemt_csv)
    if 'filename' not in b.columns or 'prop_eff_bemt' not in b.columns: return None
    return b.groupby('filename')['prop_eff_bemt'].mean().clip(0,1).to_frame('eta_bemt_mean').reset_index()

def calibrate_bemt_on_tested(bemt_table, tested_table):
    merged = tested_table[['filename','eta_mean']].merge(bemt_table, on='filename', how='inner')
    if merged.empty: return None
    x = merged['eta_bemt_mean'].to_numpy(float); y = merged['eta_mean'].to_numpy(float)
    A = np.vstack([x, np.ones_like(x)]).T
    a,b = np.linalg.lstsq(A,y, rcond=None)[0]
    return float(a), float(b)

# ----- NEW: CFD single-RPM surrogate (no NaNs) -----
def load_cfd_single(path):
    if not os.path.exists(path): return None
    t = pd.read_csv(path)
    need = GEOM_BASE + ['eta_cfd_single']
    if not all(c in t.columns for c in need): return None
    for c in GEOM_BASE:
        t[c] = pd.to_numeric(t[c], errors='coerce').round(4)
    t['eta_cfd_single'] = pd.to_numeric(t['eta_cfd_single'], errors='coerce').clip(0,1)
    t = t.dropna(subset=GEOM_BASE + ['eta_cfd_single']).reset_index(drop=True)
    return t

def fit_cfd_surrogate(cfd_df):
    # simple geometry->eta surrogate (Ridge on standardized inputs)
    X = cfd_df[GEOM_BASE].to_numpy(float); y = cfd_df['eta_cfd_single'].to_numpy(float)
    cv_k = min(5, max(3, len(y))) if len(y) > 2 else 3
    model = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-6,3,20), cv=cv_k, scoring='neg_mean_absolute_error'))])
    model.fit(X, y)
    return model

def calibrate_linear(x_pred, y_true):
    # y_true ≈ a * x_pred + b
    A = np.vstack([x_pred, np.ones_like(x_pred)]).T
    a,b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    return float(a), float(b)

# ---------- Main ----------
def main():
    # 1) Load DOE & filter
    doe_all = load_doe(DOE_FILES)
    doe = filter_doe(doe_all, PREFER_PLAN_02, INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS, PLAN_02_BOUNDS)

    # 2) Tested props
    tested_set, tested_table = load_tested_props(MASTER_PARQUET)

    # 3) Exploration space (numeric)
    numeric_cols = GEOM_BASE + ['logE_eff']
    if INCLUDE_TWIST_SLOPE and 'twist_slope' in doe.columns:
        numeric_cols.append('twist_slope'); FEATURE_WEIGHTS.setdefault('twist_slope', 1.0)
    for c in numeric_cols:
        if c not in doe.columns: raise SystemExit(f"DOE missing required column: '{c}'")
    X_num = doe[numeric_cols].astype(float).to_numpy()
    Xz, mu, sd = standardize(X_num)
    w = np.array([FEATURE_WEIGHTS.get(c,1.0) for c in numeric_cols], float)
    Xw = Xz * w[None,:]

    filenames = doe['filename'].astype(str).tolist()
    idx_all = np.arange(len(filenames))
    idx_tested = [i for i in idx_all if filenames[i] in tested_set]
    idx_pool   = [i for i in idx_all if filenames[i] not in tested_set]
    if len(idx_pool) == 0:
        print("All DOE candidates (after filtering) appear to be tested. Nothing to recommend."); return

    # 4) Exploitation via ML prediction
    pred_eta = np.full(len(doe), np.nan)
    if tested_table is not None and len(tested_table) >= 3:
        Xtr_df = build_exploit_features_df(tested_table)
        Xte_df = build_exploit_features_df(doe).reindex(columns=Xtr_df.columns, fill_value=0.0)
        pred_eta[:] = ridge_pred(Xtr_df.to_numpy(float),
                                 tested_table['eta_mean'].to_numpy(float),
                                 Xte_df.to_numpy(float))
    else:
        print("Not enough tested props to train ML; will rely on priors if present.")

    # 5) BEMT prior (optional)
    eta_bemt = np.full(len(doe), np.nan)
    if USE_BEMT_PRIOR:
        bt = bemt_avg_by_prop(BEMT_CSV)
        if bt is not None:
            tmp = doe[['filename']].merge(bt, on='filename', how='left')
            eta_bemt = tmp['eta_bemt_mean'].to_numpy(float)
            if tested_table is not None and len(tested_table) >= 3:
                fit = calibrate_bemt_on_tested(bt, tested_table)
                if fit is not None:
                    a,b = fit; eta_bemt = np.clip(a*eta_bemt + b, 0.0, 1.0)

    # 6) CFD single-RPM prior via surrogate (never NaN)
    eta_cfd1 = np.full(len(doe), np.nan)
    if USE_CFD_SINGLE:
        cfd = load_cfd_single(CFD_SINGLE_CSV)
        if cfd is not None and len(cfd) >= 3:
            cfd_model = fit_cfd_surrogate(cfd)
            # predict for DOE
            eta_cfd_pred_doe = cfd_model.predict(doe[GEOM_BASE].to_numpy(float))
            # calibrate using tested geometries
            if tested_table is not None and len(tested_table) >= 3:
                eta_cfd_pred_tested = cfd_model.predict(tested_table[GEOM_BASE].to_numpy(float))
                a,b = calibrate_linear(eta_cfd_pred_tested, tested_table['eta_mean'].to_numpy(float))
                eta_cfd1 = np.clip(a*eta_cfd_pred_doe + b, 0.0, 1.0)
            else:
                eta_cfd1 = np.clip(eta_cfd_pred_doe, 0.0, 1.0)

    # 7) Blend exploit sources (renormalize weights if some missing)
    bw = BEMT_WEIGHT if USE_BEMT_PRIOR else 0.0
    sw = CFD_SINGLE_WEIGHT if USE_CFD_SINGLE else 0.0
    mw = max(0.0, 1.0 - (bw + sw))
    combo = np.zeros(len(doe), float)
    for i in range(len(doe)):
        s = 0.0; wsum = 0.0
        if np.isfinite(pred_eta[i]): s += mw * pred_eta[i]; wsum += mw
        if np.isfinite(eta_bemt[i]): s += bw * eta_bemt[i]; wsum += bw
        if np.isfinite(eta_cfd1[i]): s += sw * eta_cfd1[i]; wsum += sw
        combo[i] = s / max(wsum, 1e-6)
    exploit_raw   = combo
    exploit_score = z01(exploit_raw)

    # 8) Exploration score
    X_tested = Xw[idx_tested,:] if len(idx_tested)>0 else np.zeros((0,Xw.shape[1]))
    dist_to_tested = np.array([pairwise_min_dist(Xw[i,:], X_tested) for i in idx_all], float)
    explore_score  = z01(dist_to_tested)

    # 9) Composite and picks
    composite = W_EXPLOIT * exploit_score + W_EXPLORE * explore_score
    pool_sorted = sorted(idx_pool, key=lambda i: composite[i], reverse=True)
    chosen = []
    for i in pool_sorted:
        if len(chosen) >= K_RECOMMEND: break
        if MIN_DIST_BETWEEN_RECS <= 0.0:
            chosen.append(i); continue
        ok = True
        for j in chosen:
            if np.linalg.norm(Xw[i,:] - Xw[j,:]) < MIN_DIST_BETWEEN_RECS:
                ok = False; break
        if ok: chosen.append(i)
    if len(chosen) < K_RECOMMEND:
        remaining = [i for i in idx_pool if i not in chosen]
        chosen += [i for (i, _) in greedy_maximin_indices(Xw, idx_tested + chosen, remaining, K_RECOMMEND - len(chosen), 0.0)]

    # 10) Output table
    out_rows=[]
    for rank,i in enumerate(chosen, start=1):
        row = {
            'rank': rank,
            'filename': filenames[i],
            'source_plan': doe.loc[i,'__plan'],
            'pred_eta_mean': float(np.clip(pred_eta[i],0,1)) if np.isfinite(pred_eta[i]) else np.nan,
            'bemt_eta_mean_cal': float(eta_bemt[i]) if np.isfinite(eta_bemt[i]) else np.nan,
            'cfd_single_eta_cal': float(eta_cfd1[i]) if np.isfinite(eta_cfd1[i]) else np.nan,
            'exploit_score': float(exploit_score[i]),
            'explore_score': float(explore_score[i]),
            'composite_score': float(composite[i]),
            'min_dist_std_space': float(dist_to_tested[i]),
        }
        for c in GEOM_BASE + ['logE_eff'] + (['twist_slope'] if INCLUDE_TWIST_SLOPE and 'twist_slope' in doe.columns else []) + CAT_COLS:
            if c in doe.columns: row[c] = doe.loc[i, c]
        out_rows.append(row)
    out = pd.DataFrame(out_rows).sort_values('rank').reset_index(drop=True)

    os.makedirs(TOOLS_DIR, exist_ok=True)
    out.to_csv(OUTPUT_RECS, index=False)

    print("\nRecommended next props to test (exploit+explore):")
    view = ['rank','filename','source_plan','composite_score','exploit_score','explore_score','min_dist_std_space',
            'pred_eta_mean','bemt_eta_mean_cal','cfd_single_eta_cal'] + GEOM_BASE + ['logE_eff'] + CAT_COLS
    with pd.option_context('display.max_columns', None):
        print(out[view].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved recommendations to: {OUTPUT_RECS}")


if __name__ == "__main__":
    main()
