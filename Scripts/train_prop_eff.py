# train_gpr_prop_eff.py
# Prop-level model: predict AVERAGE prop efficiency (eta_mean) from geometry only.
# LOPO over props; we report GLOBAL LOPO R^2 (well-defined), MAE, and Spearman ρ.
# Also writes a tidy per-prop predictions CSV and a scatter plot.
#
# Run from VS Code "Run Python File" (paths are relative to this file's location).

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------- paths (relative to this file) ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')
PLOTS_DIR  = os.path.join(PROJ_ROOT, 'Plots')

MASTER_DATASET_PATH = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
METRICS_PATH        = os.path.join(TOOLS_DIR, 'prop_avg_metrics.csv')
PREDICTIONS_PATH    = os.path.join(TOOLS_DIR, 'prop_avg_predictions.csv')
PLOT_PATH           = os.path.join(PLOTS_DIR, 'prop_avg_model_vs_actual.pdf')

os.makedirs(TOOLS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- config ----------------
# Optional constraints; set to None to disable.
RPM_WINDOW   = None            # e.g., (1200, 2600) or None
THRUST_REQ_N = None            # e.g., 15.0 (require at least this thrust in bins used for averaging)

# Geometry-only (design-time safe) features
GEOM_COLS = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)','flexMod (GPA)']

# Column names in master_dataset.parquet
TARGET_BIN_COL = 'prop_efficiency_mean'   # per (prop, rpm_bin) efficiency from process_data.py
GROUP_COL      = 'filename'
RPM_BIN_COL    = 'rpm_bin'
THRUST_COL     = 'Thrust (N)'

def build_prop_avg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per prop with:
      - eta_mean: average of TARGET_BIN_COL over that prop's bins (after optional filters)
      - n_bins, rpm_min/max/span
      - per-prop std of η across bins (diag)
      - geometry columns
    """
    d = df.copy()

    # Optional RPM window
    if RPM_WINDOW is not None:
        lo, hi = RPM_WINDOW
        d = d[(d[RPM_BIN_COL] >= lo) & (d[RPM_BIN_COL] <= hi)].copy()

    # Optional thrust requirement
    if THRUST_REQ_N is not None:
        d = d[d[THRUST_COL] >= THRUST_REQ_N].copy()

    # Drop rows without the target
    d = d.dropna(subset=[TARGET_BIN_COL])

    if d.empty:
        raise SystemExit("No bins remain after applying RPM_WINDOW/THRUST_REQ_N; relax constraints.")

    # Aggregate per prop
    g = d.groupby(GROUP_COL)
    prop_stats = g.agg(
        eta_mean       = (TARGET_BIN_COL, 'mean'),
        eta_std_bins   = (TARGET_BIN_COL, 'std'),
        n_bins         = (TARGET_BIN_COL, 'size'),
        rpm_min        = (RPM_BIN_COL, 'min'),
        rpm_max        = (RPM_BIN_COL, 'max'),
    ).reset_index(drop=False)
    prop_stats['rpm_span'] = prop_stats['rpm_max'] - prop_stats['rpm_min']

    # Bring geometry (one row per prop)
    geom = d[[GROUP_COL] + GEOM_COLS].drop_duplicates(subset=[GROUP_COL])
    table = prop_stats.merge(geom, on=GROUP_COL, how='left')

    # Clamp η to [0,1] defensively
    table['eta_mean'] = table['eta_mean'].clip(0.0, 1.0)

    # Drop any with missing geometry or missing averages
    table = table.dropna(subset=['eta_mean'] + GEOM_COLS).reset_index(drop=True)
    return table

def lopo_train_predict(table: pd.DataFrame):
    """
    LOPO with RidgeCV (geometry-only). Returns:
      - y_true_all, y_pred_all
      - per-prop dataframe with predictions and diagnostics
    """
    X_full = table[GEOM_COLS].to_numpy(dtype=float)
    y_full = table['eta_mean'].to_numpy(dtype=float)
    props  = table[GROUP_COL].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []
    rows = []

    for fold, (tr, te) in enumerate(logo.split(X_full, y_full, groups=props), start=1):
        Xtr, Xte = X_full[tr], X_full[te]
        ytr, yte = y_full[tr], y_full[te]
        props_te = props[te]

        # Ridge with internal CV on training props (use MAE to be robust on small N)
        n_train_props = len(np.unique(props[tr]))
        cv_k = min(5, max(3, n_train_props))  # keep 3..5 folds for stability
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge',  RidgeCV(alphas=np.logspace(-6, 3, 20), cv=cv_k, scoring='neg_mean_absolute_error'))
        ])
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        # Store results (one row per held-out prop)
        for i in range(len(yte)):
            rows.append({
                'fold': fold,
                'held_out_prop': props_te[i],
                'eta_mean_actual': float(yte[i]),
                'eta_mean_pred':   float(yhat[i]),
                'abs_error':       float(abs(yhat[i] - yte[i])),
                # carry some diagnostics for that prop
                'n_bins':          int(table.loc[table[GROUP_COL]==props_te[i], 'n_bins'].iloc[0]),
                'rpm_min':         int(table.loc[table[GROUP_COL]==props_te[i], 'rpm_min'].iloc[0]),
                'rpm_max':         int(table.loc[table[GROUP_COL]==props_te[i], 'rpm_max'].iloc[0]),
                'rpm_span':        int(table.loc[table[GROUP_COL]==props_te[i], 'rpm_span'].iloc[0]),
                'eta_std_bins':    float(table.loc[table[GROUP_COL]==props_te[i], 'eta_std_bins'].iloc[0] if not np.isnan(table.loc[table[GROUP_COL]==props_te[i], 'eta_std_bins'].iloc[0]) else 0.0),
            })

        y_true_all.extend(yte.tolist())
        y_pred_all.extend(yhat.tolist())

    pred_df = pd.DataFrame(rows).sort_values('held_out_prop').reset_index(drop=True)
    return np.array(y_true_all), np.array(y_pred_all), pred_df

def summarize_and_plot(y_true_all, y_pred_all, pred_df):
    # Global LOPO metrics (well-defined)
    r2 = r2_score(y_true_all, y_pred_all) if len(y_true_all) >= 2 else np.nan
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rho, _ = spearmanr(y_true_all, y_pred_all) if len(np.unique(y_true_all)) > 1 else (np.nan, None)

    print("\nProp-level AVG-η LOPO:")
    print(f"  Global R²: {np.nan_to_num(r2):.4f} | Avg MAE: {mae:.4f} | Spearman ρ: {np.nan_to_num(rho):.3f}")
    print(f"  Props: {len(y_true_all)}, Train per fold (median): {int(pred_df['fold'].max()-1)}")

    # Save metrics & predictions
    metrics_df = pd.DataFrame([{
        'global_R2': r2,
        'avg_MAE': mae,
        'spearman_rho': rho,
        'n_props': len(y_true_all),
    }])
    metrics_df.to_csv(METRICS_PATH, index=False)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote metrics to {METRICS_PATH}")
    print(f"Wrote per-prop predictions to {PREDICTIONS_PATH}")

    # Plot
    plt.figure(figsize=(8,7))
    plt.scatter(y_true_all, y_pred_all, s=60)
    # 45° line
    lo = min(y_true_all.min(), y_pred_all.min()) - 0.02
    hi = max(y_true_all.max(), y_pred_all.max()) + 0.02
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual average efficiency (η̄)", fontsize=12)
    plt.ylabel("Predicted average efficiency (η̄)", fontsize=12)
    plt.title("Prop-level average efficiency: LOPO predictions", fontsize=14)
    # small labels near points
    for i, row in pred_df.iterrows():
        txt = row['held_out_prop'].replace('.csv','')
        plt.annotate(txt, (y_true_all[i], y_pred_all[i]), fontsize=8, xytext=(3,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to '{PLOT_PATH}'")

def main():
    print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
    df = pd.read_parquet(MASTER_DATASET_PATH)

    # Build prop-level table using average η across bins
    table = build_prop_avg_table(df)

    # Train/eval
    y_true_all, y_pred_all, pred_df = lopo_train_predict(table)

    # Summarize + plot
    summarize_and_plot(y_true_all, y_pred_all, pred_df)

if __name__ == "__main__":
    main()
