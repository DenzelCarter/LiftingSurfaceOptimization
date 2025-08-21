# make_summary_plots.py
# Generate summary plots for the prop optimization study.
# Outputs go to Experiment/Plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------- Paths ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')
PLOTS_DIR  = os.path.join(PROJ_ROOT, 'Plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MASTER_PARQUET   = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
PREDS_CSV        = os.path.join(TOOLS_DIR, 'prop_avg_predictions.csv')
METRICS_CSV      = os.path.join(TOOLS_DIR, 'prop_avg_metrics.csv')
FI_CSV           = os.path.join(TOOLS_DIR, 'feature_importance_coeffs.csv')
BEMT_CSV         = os.path.join(TOOLS_DIR, 'bemt_predictions_final.csv')      # optional
CFD_SINGLE_CSV   = os.path.join(TOOLS_DIR, 'cfd_single_prior.csv')            # optional
RECS_CSV         = os.path.join(TOOLS_DIR, 'next_props_recommendations.csv')  # optional

GEOM_COLS  = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']

# ---------------- Helpers ----------------
def read_or_none(path, reader):
    try:
        if os.path.exists(path):
            return reader(path)
    except Exception as e:
        print(f"Warn: failed to read {path}: {e}")
    return None

def build_prop_avg_table(m):
    g = (m.dropna(subset=['prop_efficiency_mean'])
           .groupby('filename', as_index=False)['prop_efficiency_mean'].mean()
           .rename(columns={'prop_efficiency_mean':'eta_mean'}))
    g['eta_mean'] = g['eta_mean'].clip(0,1)
    meta_cols = ['filename'] + [c for c in GEOM_COLS if c in m.columns] + [c for c in ['logE_eff','process','material','orientation'] if c in m.columns]
    meta = m[meta_cols].drop_duplicates('filename')
    t = g.merge(meta, on='filename', how='left')
    for c in GEOM_COLS:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors='coerce').round(4)
    return t

def fit_line(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2: return (np.nan, np.nan, np.nan)
    a, b = np.polyfit(x, y, 1)
    # R^2
    yhat = a*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, r2

def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close()

# ---------------- Load core tables ----------------
m = read_or_none(MASTER_PARQUET, pd.read_parquet)
if m is None:
    raise SystemExit(f"Missing master dataset at {MASTER_PARQUET}")

table = build_prop_avg_table(m)

preds = read_or_none(PREDS_CSV, pd.read_csv)
metrics = read_or_none(METRICS_CSV, pd.read_csv)
fi = read_or_none(FI_CSV, pd.read_csv)
bemt = read_or_none(BEMT_CSV, pd.read_csv)
cfd  = read_or_none(CFD_SINGLE_CSV, pd.read_csv)
recs = read_or_none(RECS_CSV, pd.read_csv)

# =========================
# 1) LOPO Parity Plot
# =========================
if preds is not None:
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    x = preds['eta_true'].to_numpy(float)
    y = preds['eta_pred'].to_numpy(float)
    ax.scatter(x, y, s=36, alpha=0.9, edgecolor='none')
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Measured η̄'); ax.set_ylabel('Predicted η̄')
    # Annotate metrics
    r2 = metrics['global_r2'].iloc[0] if metrics is not None and 'global_r2' in metrics.columns else np.nan
    mae = metrics['avg_mae'].iloc[0] if metrics is not None and 'avg_mae' in metrics.columns else np.nan
    rho = metrics['spearman_rho'].iloc[0] if metrics is not None and 'spearman_rho' in metrics.columns else np.nan
    txt = f"$R^2$={r2:0.3f}\nMAE={mae:0.3f}\n$\\rho_s$={rho:0.3f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
    savefig(os.path.join(PLOTS_DIR, 'parity_lopo.pdf'))

# =========================
# 2) Feature Importances
# =========================
if fi is not None and {'feature','abs_coef_std','group'}.issubset(fi.columns):
    topN = 20
    f2 = fi.sort_values('abs_coef_std', ascending=True).tail(topN)
    # Colors by group
    groups = f2['group'].fillna('other').tolist()
    uniq = {g:i for i,g in enumerate(sorted(set(groups)))}
    cmap = plt.get_cmap('tab20')
    colors = [cmap(uniq[g] % 20) for g in groups]

    fig, ax = plt.subplots(figsize=(7.0, max(4.5, 0.30*topN)))
    ax.barh(f2['feature'], f2['abs_coef_std'], color=colors)
    ax.set_xlabel('Absolute standardized coefficient')
    ax.set_ylabel('Feature')
    # legend
    handles = []
    for g in sorted(uniq.keys(), key=lambda k: uniq[k]):
        handles.append(plt.Line2D([0],[0], color=cmap(uniq[g] % 20), lw=8, label=g))
    ax.legend(handles=handles, title='Group', loc='lower right', frameon=True)
    savefig(os.path.join(PLOTS_DIR, 'feature_importance.pdf'))

# =========================
# 3) Prior calibration: BEMT (optional)
# =========================
if bemt is not None and {'filename','prop_eff_bemt'}.issubset(bemt.columns):
    bemt_avg = (bemt.groupby('filename')['prop_eff_bemt']
                    .mean().clip(0,1).reset_index()
                    .rename(columns={'prop_eff_bemt':'eta_bemt_mean'}))
    merged = preds[['filename','eta_true']].merge(bemt_avg, on='filename', how='inner') if preds is not None else \
             table[['filename','eta_mean']].rename(columns={'eta_mean':'eta_true'}).merge(bemt_avg, on='filename', how='inner')
    if not merged.empty:
        x = merged['eta_bemt_mean'].to_numpy(float); y = merged['eta_true'].to_numpy(float)
        a, b, r2 = fit_line(x, y)
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        ax.scatter(x, y, s=36, alpha=0.9, edgecolor='none')
        xs = np.linspace(0,1,100); ax.plot(xs, xs, 'k--', lw=1, label='y=x')
        if np.all(np.isfinite([a,b])):
            ax.plot(xs, a*xs + b, '-', lw=1.5, label=f'calib: y={a:0.2f}x+{b:0.2f}')
        ax.set_xlabel('BEMT (avg across rpm)'); ax.set_ylabel('Measured η̄')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(frameon=True)
        ax.text(0.04, 0.96, f"$R^2$={r2:0.3f}", transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
        savefig(os.path.join(PLOTS_DIR, 'prior_calibration_bemt.pdf'))

# =========================
# 4) Prior calibration: CFD surrogate (optional)
# =========================
if cfd is not None and {'eta_cfd_single'}.issubset(cfd.columns):
    # Fit simple surrogate CFD(geom) on CFD table
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        c = cfd.copy()
        for col in GEOM_COLS:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors='coerce').round(4)
        c = c.dropna(subset=GEOM_COLS + ['eta_cfd_single']).copy()
        if len(c) >= 3 and set(GEOM_COLS).issubset(table.columns):
            Xc = c[GEOM_COLS].to_numpy(float); yc = c['eta_cfd_single'].to_numpy(float)
            model = Pipeline([
                ('imp', SimpleImputer(strategy='mean')),
                ('scl', StandardScaler()),
                ('rid', RidgeCV(alphas=np.logspace(-6,3,40), cv=min(5, max(3, len(yc))), scoring='neg_mean_absolute_error'))
            ])
            model.fit(Xc, yc)
            # Predict on tested geometries
            tested = table[GEOM_COLS + ['filename','eta_mean']].dropna()
            xpred = model.predict(tested[GEOM_COLS].to_numpy(float))
            a,b,r2 = fit_line(xpred, tested['eta_mean'].to_numpy(float))
            fig, ax = plt.subplots(figsize=(5.0, 4.5))
            ax.scatter(xpred, tested['eta_mean'], s=36, alpha=0.9, edgecolor='none')
            xs = np.linspace(0,1,100); ax.plot(xs, xs, 'k--', lw=1, label='y=x')
            if np.all(np.isfinite([a,b])):
                ax.plot(xs, a*xs + b, '-', lw=1.5, label=f'calib: y={a:0.2f}x+{b:0.2f}')
            ax.set_xlabel('CFD surrogate η (raw)'); ax.set_ylabel('Measured η̄')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.legend(frameon=True)
            ax.text(0.04, 0.96, f"$R^2$={r2:0.3f}", transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
            savefig(os.path.join(PLOTS_DIR, 'prior_calibration_cfd_surrogate.pdf'))
    except Exception as e:
        print(f"Warn: CFD surrogate calibration plot skipped: {e}")

# =========================
# 5) Design-space coverage: AR vs lambda colored by η̄
# =========================
if {'AR','lambda','eta_mean'}.issubset(table.columns):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(table['AR'], table['lambda'],
                    c=table['eta_mean'], s=60, cmap='viridis', edgecolor='k', lw=0.3)
    cb = plt.colorbar(sc, ax=ax); cb.set_label('Measured η̄')
    ax.set_xlabel('AR'); ax.set_ylabel('λ')
    ax.grid(True, alpha=0.2)
    savefig(os.path.join(PLOTS_DIR, 'coverage_AR_lambda.pdf'))

# =========================
# 6) η̄ vs logE_eff (material/printing influence)
# =========================
if {'eta_mean','logE_eff'}.issubset(table.columns):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    x = table['logE_eff'].to_numpy(float)
    y = table['eta_mean'].to_numpy(float)
    ax.scatter(x, y, s=60, alpha=0.9, edgecolor='k', lw=0.3)
    # simple linear fit for visual trend
    if np.isfinite(x).sum() >= 2:
        a, b, r2 = fit_line(x, y)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if np.all(np.isfinite([a,b])):
            ax.plot(xs, a*xs + b, '--', lw=1.5, color='k', alpha=0.7, label=f'fit $R^2$={r2:0.2f}')
            ax.legend(frameon=True)
    ax.set_xlabel('log(E_eff) [Pa, natural log]'); ax.set_ylabel('Measured η̄')
    ax.grid(True, alpha=0.2)
    savefig(os.path.join(PLOTS_DIR, 'eta_vs_logE.pdf'))

# =========================
# 7) Recommendations (exploit/explore scores)
# =========================
if recs is not None and {'filename','rank','exploit_score','explore_score','composite_score'}.issubset(recs.columns):
    r = recs.sort_values('rank').copy()
    labels = r['filename'].astype(str).tolist()
    xloc = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6.0, 0.8*len(labels)), 4.2))
    ax.bar(xloc - width/2, r['exploit_score'], width, label='Exploit')
    ax.bar(xloc + width/2, r['explore_score'], width, label='Explore')
    for i, comp in enumerate(r['composite_score'].to_numpy(float)):
        ax.text(i, max(r['exploit_score'].iloc[i], r['explore_score'].iloc[i]) + 0.03,
                f"comp={comp:0.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xloc); ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylabel('Score (0–1)'); ax.set_title('Next-prop selection scores')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis='y', alpha=0.2)
    savefig(os.path.join(PLOTS_DIR, 'recommendations_scores.pdf'))
