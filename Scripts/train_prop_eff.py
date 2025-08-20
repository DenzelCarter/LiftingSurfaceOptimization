# train_prop_eff.py
# Prop-level model: predict AVERAGE prop efficiency (eta_mean) from geometry + effective stiffness.
# LOPO over props; reports GLOBAL LOPO R^2, MAE, and Spearman ρ. Saves per-prop predictions and a plot.
# Adds diagnostics: standardized coefficients, grouped L2 importances, permutation importance.

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------- paths ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')
PLOTS_DIR  = os.path.join(PROJ_ROOT, 'Plots')

MASTER_DATASET_PATH = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
METRICS_PATH        = os.path.join(TOOLS_DIR, 'prop_avg_metrics.csv')
PREDICTIONS_PATH    = os.path.join(TOOLS_DIR, 'prop_avg_predictions.csv')
PLOT_PATH           = os.path.join(PLOTS_DIR, 'prop_avg_model_vs_actual.pdf')

# Diagnostics outputs
FI_COEFFS_PATH      = os.path.join(TOOLS_DIR, 'feature_importance_coeffs.csv')
FI_GROUPS_PATH      = os.path.join(TOOLS_DIR, 'feature_importance_groups.csv')
FI_PERM_PATH        = os.path.join(TOOLS_DIR, 'feature_importance_permutation.csv')

os.makedirs(TOOLS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- config ----------------
# Use the full rpm range (your best result)
RPM_WINDOW   = None            # was (350, 2000)
THRUST_REQ_N = None            # e.g., 15.0 or None

# Keep behavior stable (you found both best off)
USE_TRIMMED_MEAN  = False      # robust average per prop (off by default)
TRIM_PCT          = 0.10
USE_SAMPLE_WEIGHTS= False      # weight props by n_bins (off by default)

# Base numeric features (geometry)
BASE_NUM_COLS = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']

# Stiffness features – your preferred setup drops the deflection proxy
# STIFF_NUM_COLS = ['logE_eff','C_deflect']
STIFF_NUM_COLS = ['logE_eff']  # <— only logE_eff

# Categorical metadata to one-hot (keep full set)
CAT_COLS = ['process', 'orientation', 'material']

TARGET_BIN_COL = 'prop_efficiency_mean'   # per (prop, rpm_bin)
GROUP_COL      = 'base_prop_id'           # group by geometry family for LOPO
RPM_BIN_COL    = 'rpm_bin'
THRUST_COL     = 'Thrust (N)'

# Diagnostics toggle
RUN_IMPORTANCE_DIAGNOSTICS = True  # writes CSVs with importances

def _extract_base_prop_id(fn: str) -> str:
    m = re.search(r'(Prop_\d+)', str(fn))
    return m.group(1) if m else str(fn).split('.')[0]

def _trimmed_mean(series: pd.Series, p: float = TRIM_PCT) -> float:
    if series.size == 0:
        return np.nan
    lo, hi = series.quantile([p, 1 - p])
    return series.clip(lower=lo, upper=hi).mean()

def build_prop_avg_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if 'base_prop_id' not in d.columns:
        d['base_prop_id'] = d['filename'].apply(_extract_base_prop_id)

    if RPM_WINDOW is not None:
        lo, hi = RPM_WINDOW
        d = d[(d[RPM_BIN_COL] >= lo) & (d[RPM_BIN_COL] <= hi)]
    if THRUST_REQ_N is not None:
        d = d[d[THRUST_COL] >= THRUST_REQ_N]

    d = d.dropna(subset=[TARGET_BIN_COL])
    if d.empty:
        raise SystemExit("No bins remain after RPM/Thrust filters.")

    g = d.groupby('filename')
    if USE_TRIMMED_MEAN:
        eta_mean_series = g[TARGET_BIN_COL].apply(lambda s: _trimmed_mean(s, TRIM_PCT))
    else:
        eta_mean_series = g[TARGET_BIN_COL].mean()

    prop_stats = pd.DataFrame({
        'filename': g.size().index,
        'eta_mean': eta_mean_series.values,
        'eta_std_bins': g[TARGET_BIN_COL].std().values,
        'n_bins': g[TARGET_BIN_COL].size().values,
        'rpm_min': g[RPM_BIN_COL].min().values,
        'rpm_max': g[RPM_BIN_COL].max().values,
    })
    prop_stats['rpm_span'] = prop_stats['rpm_max'] - prop_stats['rpm_min']

    # one set of meta per filename
    meta_cols = ['filename','base_prop_id'] + BASE_NUM_COLS + STIFF_NUM_COLS + CAT_COLS
    geom = d[meta_cols].drop_duplicates(subset=['filename'])

    table = prop_stats.merge(geom, on='filename', how='left')
    table['eta_mean'] = table['eta_mean'].clip(0.0, 1.0)

    need_cols = ['eta_mean'] + BASE_NUM_COLS + STIFF_NUM_COLS + CAT_COLS + ['base_prop_id']
    table = table.dropna(subset=need_cols).reset_index(drop=True)
    return table

def _build_design_matrix(table: pd.DataFrame):
    d = table.copy()
    for c in CAT_COLS:
        d[c] = d[c].astype(str).str.lower()
    X_cat = pd.get_dummies(d[CAT_COLS], prefix=CAT_COLS, drop_first=False)
    X_num = d[BASE_NUM_COLS + STIFF_NUM_COLS].copy()

    # Interactions: logE_eff × each categorical dummy
    logE = d['logE_eff'].to_numpy().reshape(-1,1)
    inter_list, inter_cols = [], []
    for col in X_cat.columns:
        v = X_cat[[col]].to_numpy()
        inter_list.append((logE * v).ravel())
        inter_cols.append(f"logE_eff__x__{col}")
    X_inter = (pd.DataFrame(np.column_stack(inter_list), columns=inter_cols, index=d.index)
               if inter_list else pd.DataFrame(index=d.index))

    X = pd.concat([X_num, X_cat, X_inter], axis=1)
    return X, X.columns.tolist()

def lopo_train_predict(table: pd.DataFrame):
    X, feat_names = _build_design_matrix(table)
    y = table['eta_mean'].to_numpy(float)
    groups = table[GROUP_COL].astype(str).to_numpy()

    w = None
    if USE_SAMPLE_WEIGHTS:
        w = table['n_bins'].to_numpy(float)
        w = w / np.mean(w)

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, rows = [], [], []

    for fold, (tr, te) in enumerate(logo.split(X, y, groups=groups), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        props_te  = table.iloc[te]['filename'].to_numpy()

        n_train_props = len(np.unique(groups[tr]))
        cv_k = min(5, max(3, n_train_props))
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge',  RidgeCV(alphas=np.logspace(-6, 3, 20), cv=cv_k, scoring='neg_mean_absolute_error'))
        ])
        if w is not None:
            model.fit(Xtr, ytr, ridge__sample_weight=w[tr])
        else:
            model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        for i in range(len(yte)):
            r = table.iloc[te[i]]
            rows.append({
                'fold': fold,
                'held_out_prop': props_te[i],
                'base_prop_id':  r['base_prop_id'],
                'eta_mean_actual': float(yte[i]),
                'eta_mean_pred':   float(yhat[i]),
                'abs_error':       float(abs(yhat[i] - yte[i])),
                'n_bins':          int(r['n_bins']),
                'rpm_min':         int(r['rpm_min']),
                'rpm_max':         int(r['rpm_max']),
                'rpm_span':        int(r['rpm_span']),
                'eta_std_bins':    float(0.0 if np.isnan(r['eta_std_bins']) else r['eta_std_bins']),
            })
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(yhat.tolist())

    pred_df = pd.DataFrame(rows).sort_values(['base_prop_id','held_out_prop']).reset_index(drop=True)
    return np.array(y_true_all), np.array(y_pred_all), pred_df, feat_names

def summarize_and_plot(y_true_all, y_pred_all, pred_df):
    r2  = r2_score(y_true_all, y_pred_all) if len(y_true_all) >= 2 else np.nan
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rho, _ = spearmanr(y_true_all, y_pred_all) if len(np.unique(y_true_all)) > 1 else (np.nan, None)

    print("\nProp-level AVG-η LOPO:")
    print(f"  Global R²: {np.nan_to_num(r2):.4f} | Avg MAE: {mae:.4f} | Spearman ρ: {np.nan_to_num(rho):.3f}")
    print(f"  Held-out props: {len(y_true_all)}")

    pd.DataFrame([{
        'global_R2': r2, 'avg_MAE': mae, 'spearman_rho': rho, 'n_props': len(y_true_all),
    }]).to_csv(METRICS_PATH, index=False)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote metrics to {METRICS_PATH}")
    print(f"Wrote per-prop predictions to {PREDICTIONS_PATH}")

    plt.figure(figsize=(8,7))
    plt.scatter(y_true_all, y_pred_all, s=60)
    lo = min(y_true_all.min(), y_pred_all.min()) - 0.02
    hi = max(y_true_all.max(), y_pred_all.max()) + 0.02
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual average efficiency (η̄)", fontsize=12)
    plt.ylabel("Predicted average efficiency (η̄)", fontsize=12)
    plt.title("Prop-level average efficiency: LOPO predictions", fontsize=14)
    for i, row in pred_df.iterrows():
        txt = row['held_out_prop'].replace('.csv','')
        plt.annotate(txt, (y_true_all[i], y_pred_all[i]), fontsize=8, xytext=(3,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to '{PLOT_PATH}'")

# ---------------- diagnostics ----------------
def _infer_group(feat: str) -> str:
    if feat in BASE_NUM_COLS:
        return 'geometry'
    if feat in STIFF_NUM_COLS:
        return 'stiffness'
    if feat.startswith('process_'):
        return 'process'
    if feat.startswith('material_'):
        return 'material'
    if feat.startswith('orientation_'):
        return 'orientation'
    if feat.startswith('logE_eff__x__process_'):
        return 'interaction:logE×process'
    if feat.startswith('logE_eff__x__material_'):
        return 'interaction:logE×material'
    if feat.startswith('logE_eff__x__orientation_'):
        return 'interaction:logE×orientation'
    return 'other'

def fit_full_and_importance(table: pd.DataFrame, feat_names: list):
    """Fit one model on all data, export standardized coefficients + permutation importances."""
    X, feats = _build_design_matrix(table)
    y = table['eta_mean'].to_numpy(float)

    # Fit one pipeline on all data (CV picks alpha internally)
    cv_k = min(5, max(3, len(table)//3 if len(table) >= 6 else 3))
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  RidgeCV(alphas=np.logspace(-6, 3, 40), cv=cv_k, scoring='neg_mean_absolute_error'))
    ])

    model.fit(X, y)
    coef = model.named_steps['ridge'].coef_
    coef_df = pd.DataFrame({
        'feature': feats,
        'coef_std': coef,
        'abs_coef_std': np.abs(coef),
        'group': [ _infer_group(f) for f in feats ],
    }).sort_values('abs_coef_std', ascending=False).reset_index(drop=True)
    coef_df.to_csv(FI_COEFFS_PATH, index=False)

    # Grouped L2 norm importance (scale-invariant in standardized space)
    grp = coef_df.groupby('group')['coef_std'].apply(lambda v: float(np.sqrt(np.sum(v.values**2))))
    grp_df = grp.sort_values(ascending=False).reset_index().rename(columns={'coef_std':'group_L2_importance'})
    grp_df.to_csv(FI_GROUPS_PATH, index=False)

    # Permutation importance on full data (ΔR²)
    perm = permutation_importance(model, X, y, n_repeats=200, random_state=42, scoring='r2')
    perm_df = pd.DataFrame({
        'feature': feats,
        'perm_importance_mean': perm.importances_mean,
        'perm_importance_std': perm.importances_std,
        'group': [ _infer_group(f) for f in feats ],
    }).sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)
    perm_df.to_csv(FI_PERM_PATH, index=False)

    # Console summary (top 10)
    print("\n[Diagnostics] Top features by standardized |coef|:")
    print(coef_df.head(10).to_string(index=False))
    print("\n[Diagnostics] Grouped L2 importance:")
    print(grp_df.to_string(index=False))
    print("\n[Diagnostics] Top features by permutation ΔR²:")
    print(perm_df.head(10).to_string(index=False))

def main():
    print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
    df = pd.read_parquet(MASTER_DATASET_PATH)
    table = build_prop_avg_table(df)
    y_true_all, y_pred_all, pred_df, feat_names = lopo_train_predict(table)
    summarize_and_plot(y_true_all, y_pred_all, pred_df)

    if RUN_IMPORTANCE_DIAGNOSTICS:
        fit_full_and_importance(table, feat_names)

    print("\nFeatures used:")
    for f in feat_names:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
