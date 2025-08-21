# train_prop_eff.py
# Prop-level AVG-η LOPO with optional CFD surrogate feature.
# - No rpm window; no deflection proxy.
# - Geometry + logE_eff + one-hot cats (+ logE×cats).
# - (Optional) CFD surrogate: learn from cfd_single_prior.csv and calibrate to bench η̄.
# - BEMT is NOT used in training.
# - Mean-impute + standardize; robust to NaNs.

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -------- Paths --------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

MASTER_PARQUET        = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
CFD_SINGLE_PRIOR_PATH = os.path.join(TOOLS_DIR, 'cfd_single_prior.csv')   # from integrate_cfd_single.py

OUT_METRICS = os.path.join(TOOLS_DIR, 'prop_avg_metrics.csv')
OUT_PREDS   = os.path.join(TOOLS_DIR, 'prop_avg_predictions.csv')
OUT_COEFS   = os.path.join(TOOLS_DIR, 'feature_importance_coeffs.csv')

# -------- Columns --------
GEOM_COLS  = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)']
STIFF_COLS = ['logE_eff']
CAT_COLS   = ['process', 'material', 'orientation']

# -------- Feature toggles --------
USE_CATS_DEFAULT                 = True
USE_LOGE_X_CAT_INTERACT_DEFAULT  = True
USE_CFD_SURR_FEATURE_DEFAULT     = False   # <- turn OFF if you want training 100% experimental

def _print(x): print(x, flush=True)

# ---------- Data ----------
def load_master(path):
    if not os.path.exists(path):
        raise SystemExit(f"Master dataset not found: {path}")
    m = pd.read_parquet(path)
    need = ['filename','prop_efficiency_mean'] + GEOM_COLS + STIFF_COLS
    miss = [c for c in need if c not in m.columns]
    if miss:
        raise SystemExit(f"Master parquet missing columns: {miss}")
    return m

def build_prop_avg_table(m: pd.DataFrame):
    # per-prop η̄
    g = (m.dropna(subset=['prop_efficiency_mean'])
           .groupby('filename', as_index=False)['prop_efficiency_mean'].mean()
           .rename(columns={'prop_efficiency_mean':'eta_mean'}))
    g['eta_mean'] = g['eta_mean'].clip(0,1)

    meta_cols = ['filename'] + GEOM_COLS + STIFF_COLS + [c for c in CAT_COLS if c in m.columns]
    meta = m[meta_cols].drop_duplicates('filename')

    table = g.merge(meta, on='filename', how='left')
    for c in GEOM_COLS:
        table[c] = pd.to_numeric(table[c], errors='coerce').round(4)

    table = table.dropna(subset=GEOM_COLS + STIFF_COLS).reset_index(drop=True)
    return table

# ---------- CFD surrogate (covers all rows) ----------
def load_cfd_single(path):
    if not os.path.exists(path): return None
    t = pd.read_csv(path)
    need = GEOM_COLS + ['eta_cfd_single']
    if not all(c in t.columns for c in need): return None
    for c in GEOM_COLS:
        t[c] = pd.to_numeric(t[c], errors='coerce').round(4)
    t['eta_cfd_single'] = pd.to_numeric(t['eta_cfd_single'], errors='coerce').clip(0,1)
    t = t.dropna(subset=GEOM_COLS + ['eta_cfd_single']).reset_index(drop=True)
    return t

def fit_cfd_surrogate_model(cfd_df):
    # Ridge on standardized geometry -> eta_cfd_single
    X = cfd_df[GEOM_COLS].to_numpy(float); y = cfd_df['eta_cfd_single'].to_numpy(float)
    cv_k = min(5, max(3, len(y))) if len(y) > 2 else 3
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('ridge',   RidgeCV(alphas=np.logspace(-6,3,40), cv=cv_k, scoring='neg_mean_absolute_error'))
    ])
    model.fit(X, y)
    return model

def add_cfd_surrogate_feature(table: pd.DataFrame) -> pd.DataFrame:
    cfd = load_cfd_single(CFD_SINGLE_PRIOR_PATH)
    table = table.copy()
    table['eta_cfd_surr_cal'] = np.nan
    if cfd is None or len(cfd) < 3:
        _print("Note: CFD single prior not found or too small; surrogate disabled.")
        return table

    # fit surrogate on geometry -> CFD eta
    surr = fit_cfd_surrogate_model(cfd)
    # predict CFD prior for every prop row
    eta_pred_all = surr.predict(table[GEOM_COLS].to_numpy(float))

    # calibrate surrogate to bench η̄ (linear a*x + b)
    a, b = np.linalg.lstsq(np.vstack([eta_pred_all, np.ones_like(eta_pred_all)]).T,
                           table['eta_mean'].to_numpy(float), rcond=None)[0]
    table['eta_cfd_surr_cal'] = np.clip(a*eta_pred_all + b, 0.0, 1.0)
    return table

# ---------- Features ----------
def build_features_df(df: pd.DataFrame,
                      use_cfd_surr: bool,
                      use_cats: bool,
                      use_inter: bool):
    num = GEOM_COLS + STIFF_COLS
    if use_cfd_surr and 'eta_cfd_surr_cal' in df.columns:
        num = num + ['eta_cfd_surr_cal']
    X = df[num].copy().astype(float)

    if use_cats:
        cats_present = [c for c in CAT_COLS if c in df.columns]
        if cats_present:
            X_cat = pd.get_dummies(df[cats_present].astype(str), prefix=cats_present, drop_first=False)
            X = pd.concat([X, X_cat], axis=1)
            if use_inter:
                for col in X_cat.columns:
                    X[f'logE_eff__x__{col}'] = df['logE_eff'].to_numpy() * X_cat[col].to_numpy()
    return X

# ---------- Model ----------
def fit_ridge(X: pd.DataFrame, y: np.ndarray):
    mask = ~X.isna().all(axis=0)
    X = X.loc[:, mask]
    cv_k = min(5, max(3, len(y))) if len(y) > 2 else 3
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('ridge',   RidgeCV(alphas=np.logspace(-6, 3, 40), cv=cv_k, scoring='neg_mean_absolute_error')),
    ])
    model.fit(X.to_numpy(float), y)
    return model, mask

def lopo_eval(table: pd.DataFrame,
              use_cfd_surr: bool,
              use_cats: bool,
              use_inter: bool):
    yT, yP, rows = [], [], []
    for f, grp in table.groupby('filename'):
        train = table.loc[table['filename'] != f]
        test  = table.loc[table['filename'] == f]
        if len(train) < 3: continue

        Xtr = build_features_df(train, use_cfd_surr, use_cats, use_inter)
        Xte = build_features_df(test,  use_cfd_surr, use_cats, use_inter).reindex(columns=Xtr.columns, fill_value=np.nan)

        model, mask = fit_ridge(Xtr, train['eta_mean'].to_numpy(float))
        Xte = Xte.loc[:, mask]
        yhat = model.predict(Xte.to_numpy(float))
        y = float(test['eta_mean'].iloc[0]); yh = float(np.clip(yhat[0], 0, 1))
        yT.append(y); yP.append(yh)
        rows.append({'filename': f, 'eta_true': y, 'eta_pred': yh, 'abs_err': abs(y-yh)})

    yT = np.array(yT, float); yP = np.array(yP, float)
    r2  = r2_score(yT, yP) if len(yT) >= 2 else np.nan
    mae = mean_absolute_error(yT, yP) if len(yT) >= 1 else np.nan
    if _HAS_SCIPY and len(yT) >= 2: rho, _ = spearmanr(yT, yP)
    else: rho = pd.Series(yT).corr(pd.Series(yP), method='spearman') if len(yT) >= 2 else np.nan
    return r2, mae, rho, pd.DataFrame(rows)

def fit_full_and_export_coefs(table: pd.DataFrame,
                              use_cfd_surr: bool,
                              use_cats: bool,
                              use_inter: bool,
                              out_csv: str):
    X = build_features_df(table, use_cfd_surr, use_cats, use_inter)
    y = table['eta_mean'].to_numpy(float)
    mask = ~X.isna().all(axis=0)
    X = X.loc[:, mask]
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('ridge',   RidgeCV(alphas=np.logspace(-6, 3, 40), cv=min(5, max(3, len(y))), scoring='neg_mean_absolute_error'))
    ])
    model.fit(X.to_numpy(float), y)
    coef = model.named_steps['ridge'].coef_.ravel()
    cols = X.columns.tolist()

    def group_of(c):
        if c in GEOM_COLS: return 'geometry'
        if c in STIFF_COLS: return 'stiffness'
        if c.startswith('process_'): return 'process'
        if c.startswith('material_'): return 'material'
        if c.startswith('orientation_'): return 'orientation'
        if c.startswith('logE_eff__x__process_'): return 'interaction:logE×process'
        if c.startswith('logE_eff__x__material_'): return 'interaction:logE×material'
        if c.startswith('logE_eff__x__orientation_'): return 'interaction:logE×orientation'
        if c == 'eta_cfd_surr_cal': return 'prior:cfd_surrogate'
        return 'other'

    out = (pd.DataFrame({'feature': cols,
                         'coef_std': coef,
                         'abs_coef_std': np.abs(coef),
                         'group': [group_of(c) for c in cols]})
           .sort_values('abs_coef_std', ascending=False))
    out.to_csv(out_csv, index=False)
    return out

def main():
    _print(f"Loading master dataset from '{MASTER_PARQUET}'...")
    m = load_master(MASTER_PARQUET)
    table = build_prop_avg_table(m)

    # add CFD surrogate feature (if file is present)
    table = add_cfd_surrogate_feature(table)

    # Ablation: with vs without CFD surrogate
    configs = [
        ("no_cfd_surr", False),
        ("with_cfd_surr", True),
    ]
    rows = []
    for name, use_surr in configs:
        r2, mae, rho, _ = lopo_eval(
            table,
            use_cfd_surr = use_surr,
            use_cats     = USE_CATS_DEFAULT,
            use_inter    = USE_LOGE_X_CAT_INTERACT_DEFAULT
        )
        rows.append({'setup': name, 'global_r2': r2, 'avg_mae': mae, 'spearman_rho': rho})
    ablate = pd.DataFrame(rows)
    _print("\nLOPO ablation:")
    with pd.option_context('display.max_columns', None):
        _print(ablate.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    # Final config
    USE_SURR = USE_CFD_SURR_FEATURE_DEFAULT
    r2, mae, rho, per = lopo_eval(
        table,
        use_cfd_surr = USE_SURR,
        use_cats     = USE_CATS_DEFAULT,
        use_inter    = USE_LOGE_X_CAT_INTERACT_DEFAULT
    )
    _print("\nProp-level AVG-η LOPO (chosen defaults):")
    _print(f"  Global R²: {r2:0.4f} | Avg MAE: {mae:0.4f} | Spearman ρ: {rho:0.3f}")
    _print(f"  Held-out props: {len(per)}")

    os.makedirs(TOOLS_DIR, exist_ok=True)
    pd.DataFrame([{'global_r2': r2, 'avg_mae': mae, 'spearman_rho': rho, 'n_props': len(per)}]).to_csv(OUT_METRICS, index=False)
    per.sort_values('filename').to_csv(OUT_PREDS, index=False)
    fit_full_and_export_coefs(table, USE_SURR, USE_CATS_DEFAULT, USE_LOGE_X_CAT_INTERACT_DEFAULT, OUT_COEFS)

    _print(f"\nWrote metrics to {OUT_METRICS}")
    _print(f"Wrote per-prop predictions to {OUT_PREDS}")
    _print(f"Wrote feature importances to {OUT_COEFS}")

    _print("\nFeatures used:")
    for c in GEOM_COLS:  _print(f"  - {c}")
    for c in STIFF_COLS: _print(f"  - {c}")
    if USE_SURR: _print("  - eta_cfd_surr_cal")
    if USE_CATS_DEFAULT:
        for c in CAT_COLS:
            if c in m.columns: _print(f"  - {c} (one-hot)")
        if USE_LOGE_X_CAT_INTERACT_DEFAULT:
            for c in CAT_COLS:
                if c in m.columns: _print(f"  - logE_eff__x__{c} (interactions)")

if __name__ == "__main__":
    main()
