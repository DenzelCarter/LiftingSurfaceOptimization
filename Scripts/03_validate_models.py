# 03_validate_models.py
# Honest LOOCV by k_id with per-fold retraining and case curves.
# Hover: heteroscedastic (alpha = performance_variance). Cruise: homoscedastic with small nugget.
# Cruise case selection: choose k_id with the most unique alpha (fallback: most rows).
# Outputs (paths.outputs_tables):
#   03_loocv_results_hover.csv, 03_loocv_results_cruise.csv
#   03_loocv_metrics.csv, 03_interval_calibration.csv
#   03_hover_case_curve.csv, 03_hover_case_measured.csv
#   03_cruise_case_curve.csv, 03_cruise_case_measured.csv

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

import path_utils

N_RESTARTS_HOVER  = 12
N_RESTARTS_CRUISE = 12
RANDOM_STATE = 0
N_GRID = 200

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    if "twist (deg)" in df.columns and "twist" not in df.columns:
        ren["twist (deg)"] = "twist"
    if "aoa_root (deg)" in df.columns and "alpha" not in df.columns:
        ren["aoa_root (deg)"] = "alpha"
    if "op_speed" in df.columns and "v" not in df.columns:
        ren["op_speed"] = "v"
    return df.rename(columns=ren)

def _make_logo_folds(groups):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    all_idx = np.arange(len(groups))
    for g in uniq:
        te = np.where(groups == g)[0]
        tr = np.setdiff1d(all_idx, te, assume_unique=False)
        if len(tr) == 0 or len(te) == 0:
            continue
        yield tr, te

def _clean_mode_df(df_mode: pd.DataFrame, features, need_alpha: bool) -> pd.DataFrame:
    if df_mode is None or df_mode.empty:
        return df_mode
    df = df_mode.copy()
    mask = np.isfinite(df.get('performance', np.nan).astype(float))
    for f in features:
        if f in df.columns:
            mask &= np.isfinite(df[f].astype(float))
        else:
            mask &= False
    if need_alpha:
        if 'performance_variance' in df.columns:
            a = df['performance_variance'].astype(float)
            mask &= np.isfinite(a) & (a >= 0.0)
        else:
            mask &= False
    return df.loc[mask].reset_index(drop=True)

def _build_hover_pipe(D):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-9,
            normalize_y=False,
            n_restarts_optimizer=N_RESTARTS_HOVER,
            random_state=RANDOM_STATE,
        ))
    ])

def _build_cruise_pipe(D):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=N_RESTARTS_CRUISE,
            random_state=RANDOM_STATE,
        ))
    ])

def _coverage_and_nlpd(y, mu, std, meas_var=None):
    std = np.maximum(std, 1e-12)
    if meas_var is None:
        var_tot = std**2
    else:
        var_tot = std**2 + np.maximum(meas_var, 0.0)
    r = (y - mu) / np.sqrt(var_tot)
    if r.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    cov68 = float(np.mean(np.abs(r) <= 1.0))
    cov95 = float(np.mean(np.abs(r) <= 1.96))
    nlpd = float(np.mean(0.5*np.log(2*np.pi*var_tot) + 0.5*((y - mu)**2 / var_tot)))
    std_r = float(np.std(r))
    return cov68, cov95, nlpd, std_r

def _lambda_calibration(y, mu, std, meas_var=None):
    std = np.maximum(std, 1e-12)
    tot = std if meas_var is None else np.sqrt(std**2 + np.maximum(meas_var, 0.0))
    r = np.abs((y - mu) / tot)
    if r.size == 0:
        return np.nan, np.nan
    return float(np.quantile(r, 0.68)), float(np.quantile(r, 0.95))

def _attach_pred_95ci(df_pred, lam95):
    if df_pred is None or df_pred.empty:
        return df_pred
    s = df_pred['model_uncertainty_std'].astype(float)
    mu = df_pred['predicted_performance'].astype(float)
    lam = 1.0 if (lam95 is None or not np.isfinite(lam95)) else float(lam95)
    df_pred['pred_std'] = s
    df_pred['95_conf_lower'] = mu - 1.96 * lam * s
    df_pred['95_conf_upper'] = mu + 1.96 * lam * s
    return df_pred

def _attach_meas_95ci_hover(df_meas):
    if df_meas is None or df_meas.empty or 'measurement_variance' not in df_meas.columns:
        return df_meas
    s = np.sqrt(np.maximum(df_meas['measurement_variance'].astype(float), 0.0))
    mu = df_meas['measured_performance'].astype(float)
    df_meas['meas_95_conf_lower'] = mu - 1.96 * s
    df_meas['meas_95_conf_upper'] = mu + 1.96 * s
    return df_meas

def _select_case_kid(df_mode: pd.DataFrame, mode: str):
    if df_mode is None or df_mode.empty or "k_id" not in df_mode.columns:
        return None
    if mode == "cruise" and "alpha" in df_mode.columns:
        # choose k with most unique alpha (size of sweep); tie-break by total rows
        nu = df_mode.groupby("k_id")["alpha"].nunique().sort_values(ascending=False)
        if nu.empty:
            return None
        max_n = nu.iloc[0]
        cands = nu[nu == max_n].index
        counts = df_mode[df_mode["k_id"].isin(cands)].groupby("k_id").size().sort_values(ascending=False)
        return counts.index[0]
    # default: most rows
    counts = df_mode.groupby('k_id').size().sort_values(ascending=False)
    return None if counts.empty else counts.index[0]

def _build_case_curve(df_mode, features, kid, mode, n_grid=N_GRID):
    df_g = df_mode[df_mode['k_id'] == kid].copy()
    if df_g.empty:
        return None, None
    df_tr = df_mode[df_mode['k_id'] != kid].copy()
    if df_tr.empty:
        return None, None

    X_tr = df_tr[features].copy()
    y_tr = df_tr['performance'].astype(float).copy()

    if mode == "hover" and 'performance_variance' in df_tr.columns:
        v_tr = df_tr['performance_variance'].astype(float).to_numpy()
        pipe = _build_hover_pipe(D=len(features))
        pipe.named_steps['gpr'].alpha = v_tr + 1e-12
    else:
        pipe = _build_cruise_pipe(D=len(features))

    pipe.fit(X_tr, y_tr)

    sweep_col = 'v' if mode == 'hover' else 'alpha'
    fix_col   = 'alpha' if mode == 'hover' else 'v'
    if sweep_col not in features:
        variances = {c: float(df_g[c].var()) for c in features if c in df_g.columns}
        if not variances:
            return None, None
        sweep_col = max(variances, key=variances.get)
        others = [c for c in features if c != sweep_col]
        fix_col = others[0] if others else None

    geom_cols = [c for c in ['AR','lambda','twist'] if c in features]
    fixed_vals = {gc: df_g.iloc[0][gc] for gc in geom_cols}
    if fix_col and fix_col in features:
        fixed_vals[fix_col] = float(df_g[fix_col].median())

    lo = float(df_g[sweep_col].min())
    hi = float(df_g[sweep_col].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    grid_vals = np.linspace(lo, hi, n_grid)

    grid = pd.DataFrame({sweep_col: grid_vals})
    for k, v in fixed_vals.items():
        grid[k] = v
    for f in features:
        if f not in grid.columns:
            grid[f] = df_g.iloc[0][f]
    grid = grid[features]

    mu, std = pipe.predict(grid, return_std=True)

    df_curve = grid.copy()
    df_curve['k_id'] = kid
    df_curve['predicted_performance'] = mu
    df_curve['model_uncertainty_std'] = std

    keep = list({c for c in [sweep_col, fix_col, 'performance', 'performance_variance'] if c and c in df_g.columns})
    df_meas = df_g[keep].copy()
    df_meas.rename(columns={'performance':'measured_performance',
                            'performance_variance':'measurement_variance'}, inplace=True)
    df_meas['k_id'] = kid

    return df_curve, df_meas

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]
    features = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])

    master_pq = Path(P["master_parquet"])
    tables_dir = Path(P.get("outputs_tables") or P.get("dir_processed"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not master_pq.exists():
        raise FileNotFoundError(f"Master parquet not found: {master_pq}")
    df = pd.read_parquet(master_pq)
    df = _ensure_cols(df)

    need_cols = {'flight_mode','k_id','performance'}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Master parquet missing required columns {need_cols}.")

    df_hover_raw  = df[df['flight_mode']=='hover'].copy()
    df_cruise_raw = df[df['flight_mode']=='cruise'].copy()
    df_hover  = _clean_mode_df(df_hover_raw,  features, need_alpha=True)
    df_cruise = _clean_mode_df(df_cruise_raw, features, need_alpha=False)

    metrics_rows = []
    lam68_h = lam95_h = np.nan
    lam68_c = lam95_c = np.nan

    if not df_hover.empty:
        X = df_hover[features]
        y = df_hover['performance'].astype(float)
        v = df_hover['performance_variance'].astype(float)
        groups = df_hover['k_id'].astype(str).to_numpy()

        preds = []; stds = []; truths = []; meas_vars = []
        for tr, te in _make_logo_folds(groups):
            if tr.size == 0 or te.size == 0:
                continue
            pipe = _build_hover_pipe(D=len(features))
            pipe.named_steps['gpr'].alpha = v.iloc[tr].to_numpy() + 1e-12
            tr_mask = np.isfinite(y.iloc[tr].to_numpy()) & np.all(np.isfinite(X.iloc[tr].to_numpy()), axis=1)
            if tr_mask.sum() == 0: continue
            pipe.fit(X.iloc[tr][tr_mask], y.iloc[tr][tr_mask])
            te_mask = np.isfinite(y.iloc[te].to_numpy()) & np.all(np.isfinite(X.iloc[te].to_numpy()), axis=1)
            if te_mask.sum() == 0: continue
            mu, s = pipe.predict(X.iloc[te][te_mask], return_std=True)
            preds.append(mu.ravel()); stds.append(s.ravel())
            truths.append(y.iloc[te][te_mask].to_numpy().ravel())
            meas_vars.append(v.iloc[te][te_mask].to_numpy().ravel())

        y_pred = np.concatenate(preds) if preds else np.array([])
        y_std  = np.concatenate(stds) if stds else np.array([])
        y_true = np.concatenate(truths) if truths else np.array([])
        m_var  = np.concatenate(meas_vars) if meas_vars else np.array([])

        pd.DataFrame({
            "ground_truth": y_true,
            "prediction": y_pred,
            "model_uncertainty_std": y_std,
            "measurement_variance": m_var
        }).to_csv(tables_dir / "03_loocv_results_hover.csv", index=False)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true.size else np.nan
        mae  = float(mean_absolute_error(y_true, y_pred)) if y_true.size else np.nan
        rho  = float(spearmanr(y_true, y_pred)[0]) if y_true.size else np.nan
        cov68, cov95, nlpd, std_r = _coverage_and_nlpd(y_true, y_pred, y_std, meas_var=m_var)
        metrics_rows.append({
            "mode":"hover","R2":np.nan,"RMSE":rmse,"MAE":mae,"Spearman":rho,
            "Coverage@1σ":cov68,"Coverage@1.96σ":cov95,"NLPD":nlpd,"Std(StdResid)":std_r
        })
        lam68_h, lam95_h = _lambda_calibration(y_true, y_pred, y_std, meas_var=m_var)

    if not df_cruise.empty:
        X = df_cruise[features]
        y = df_cruise['performance'].astype(float)
        groups = df_cruise['k_id'].astype(str).to_numpy()

        preds = []; stds = []; truths = []
        for tr, te in _make_logo_folds(groups):
            if tr.size == 0 or te.size == 0:
                continue
            pipe = _build_cruise_pipe(D=len(features))
            tr_mask = np.isfinite(y.iloc[tr].to_numpy()) & np.all(np.isfinite(X.iloc[tr].to_numpy()), axis=1)
            if tr_mask.sum() == 0: continue
            pipe.fit(X.iloc[tr][tr_mask], y.iloc[tr][tr_mask])
            te_mask = np.isfinite(y.iloc[te].to_numpy()) & np.all(np.isfinite(X.iloc[te].to_numpy()), axis=1)
            if te_mask.sum() == 0: continue
            mu, s = pipe.predict(X.iloc[te][te_mask], return_std=True)
            preds.append(mu.ravel()); stds.append(s.ravel())
            truths.append(y.iloc[te][te_mask].to_numpy().ravel())

        y_pred = np.concatenate(preds) if preds else np.array([])
        y_std  = np.concatenate(stds) if stds else np.array([])
        y_true = np.concatenate(truths) if truths else np.array([])

        pd.DataFrame({
            "ground_truth": y_true,
            "prediction": y_pred,
            "model_uncertainty_std": y_std
        }).to_csv(tables_dir / "03_loocv_results_cruise.csv", index=False)

        r2   = float(r2_score(y_true, y_pred)) if y_true.size else np.nan
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true.size else np.nan
        mae  = float(mean_absolute_error(y_true, y_pred)) if y_true.size else np.nan
        rho  = float(spearmanr(y_true, y_pred)[0]) if y_true.size else np.nan
        cov68, cov95, nlpd, std_r = _coverage_and_nlpd(y_true, y_pred, y_std, meas_var=None)
        metrics_rows.append({
            "mode":"cruise","R2":r2,"RMSE":rmse,"MAE":mae,"Spearman":rho,
            "Coverage@1σ":cov68,"Coverage@1.96σ":cov95,"NLPD":nlpd,"Std(StdResid)":std_r
        })
        lam68_c, lam95_c = _lambda_calibration(y_true, y_pred, y_std, meas_var=None)

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(tables_dir / "03_loocv_metrics.csv", index=False)
        cal_rows = []
        if np.isfinite(lam95_h): cal_rows.append({"mode":"hover","lambda_68":lam68_h,"lambda_95":lam95_h})
        if np.isfinite(lam95_c): cal_rows.append({"mode":"cruise","lambda_68":lam68_c,"lambda_95":lam95_c})
        if cal_rows:
            pd.DataFrame(cal_rows).to_csv(tables_dir / "03_interval_calibration.csv", index=False)

    # Case studies
    if not df_hover.empty:
        kid_h = _select_case_kid(df_hover, mode="hover")
        if kid_h is not None:
            curve_h, meas_h = _build_case_curve(df_hover, features, kid_h, mode="hover", n_grid=N_GRID)
            if curve_h is not None:
                curve_h = _attach_pred_95ci(curve_h, lam95_h)
                cols = [c for c in ['k_id','AR','lambda','twist','alpha','v',
                                    'predicted_performance','model_uncertainty_std','pred_std',
                                    '95_conf_lower','95_conf_upper'] if c in curve_h.columns]
                curve_h[cols].to_csv(tables_dir / "03_hover_case_curve.csv", index=False)
            if meas_h is not None:
                meas_h = _attach_meas_95ci_hover(meas_h)
                cols = [c for c in ['k_id','v','alpha','measured_performance','measurement_variance',
                                    'meas_95_conf_lower','meas_95_conf_upper'] if c in meas_h.columns]
                meas_h[cols].to_csv(tables_dir / "03_hover_case_measured.csv", index=False)

    if not df_cruise.empty:
        kid_c = _select_case_kid(df_cruise, mode="cruise")
        if kid_c is not None:
            curve_c, meas_c = _build_case_curve(df_cruise, features, kid_c, mode="cruise", n_grid=N_GRID)
            if curve_c is not None:
                curve_c = _attach_pred_95ci(curve_c, lam95_c)
                cols = [c for c in ['k_id','AR','lambda','twist','alpha','v',
                                    'predicted_performance','model_uncertainty_std','pred_std',
                                    '95_conf_lower','95_conf_upper'] if c in curve_c.columns]
                curve_c[cols].to_csv(tables_dir / "03_cruise_case_curve.csv", index=False)
            if meas_c is not None:
                cols = [c for c in ['k_id','alpha','v','measured_performance'] if c in meas_c.columns]
                meas_c[cols].to_csv(tables_dir / "03_cruise_case_measured.csv", index=False)

    print("03: LOGOCV + case studies complete.")

if __name__ == "__main__":
    main()
