# 09_make_plots.py
# Parity plots for hover (η) and cruise (L/D), with raw and inverse-calibrated views.
# Saves:
#   Experiment/outputs/plots/parity_hover_raw.pdf
#   Experiment/outputs/plots/parity_hover_cal.pdf
#   Experiment/outputs/plots/parity_cruise_raw.pdf
#   Experiment/outputs/plots/parity_cruise_cal.pdf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

# -------------------- utilities --------------------
def _per_prop_mean(m, col):
    g = m.dropna(subset=["filename", col]).groupby("filename")[col].mean().reset_index()
    return g.rename(columns={col: f"{col}_mean"})

def _diag(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, alpha=0.7, color="grey", label="y = x")

def _fit_y_on_x(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    X = np.column_stack([x[m], np.ones(m.sum())])
    a, b = np.linalg.lstsq(X, y[m], rcond=None)[0]
    yhat = a * x[m] + b
    ss_res = np.sum((y[m] - yhat) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y[m] - yhat)))
    rho = float(spearmanr(x[m], y[m]).statistic)
    return float(a), float(b), float(r2), float(mae), float(rho), m

def _metrics_vs_identity(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan
    yhat = x[m]  # identity line
    r2 = 1.0 - np.sum((y[m] - yhat) ** 2) / (np.sum((y[m] - np.mean(y[m])) ** 2) + 1e-12)
    mae = float(np.mean(np.abs(y[m] - yhat)))
    rho = float(spearmanr(x[m], y[m]).statistic)
    return r2, mae, rho

def _cfd_surrogate_on_doe(C, mode):
    """Train geometry-only HGBR on CFD master and predict onto DOE filenames."""
    gcols = C["geometry_cols"]
    doe = pd.read_csv(C["paths"]["doe_csv"])[["filename"] + gcols].drop_duplicates("filename").dropna(subset=gcols)
    for c in gcols: doe[c] = pd.to_numeric(doe[c], errors="coerce")

    cfd_path = os.path.join(C["paths"]["outputs_tables_dir"], "cfd_master.csv")
    if not os.path.exists(cfd_path) or doe.empty:
        return pd.DataFrame(columns=["filename", "pred"])

    cfd = pd.read_csv(cfd_path)
    tgt = "eta_cfd" if mode == "vtol" else "LD_cfd"
    need = set(gcols + [tgt, "mode"])
    if not need.issubset(cfd.columns):
        return pd.DataFrame(columns=["filename", "pred"])

    tab = cfd[cfd["mode"] == mode].dropna(subset=gcols + [tgt]).copy()
    for c in gcols: tab[c] = pd.to_numeric(tab[c], errors="coerce")
    tab = tab.dropna(subset=gcols + [tgt])
    if len(tab) < 10:
        return pd.DataFrame(columns=["filename", "pred"])

    X = tab[gcols].to_numpy(float)
    y = tab[tgt].to_numpy(float)
    mdl = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42).fit(X, y)
    pred = mdl.predict(doe[gcols].to_numpy(float))
    out = doe[["filename"]].copy()
    out["pred"] = pred
    return out

def _plot_raw(ax, x, series, style, title, xlab, ylab, clip=None):
    lo = float(np.nanmin([np.nanmin(x)] + [np.nanmin(v) for v in series.values() if np.isfinite(v).any()]))
    hi = float(np.nanmax([np.nanmax(x)] + [np.nanmax(v) for v in series.values() if np.isfinite(v).any()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    _diag(ax, lo, hi)

    for name, y in series.items():
        if clip is not None:
            y = np.clip(y, clip[0], clip[1])
        a = _fit_y_on_x(x, y)
        col, marker = style[name]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=28, alpha=0.85, color=col, marker=marker, label=name)
        if a is not None:
            m, b, r2, mae, rho, mask = a
            xs = np.linspace(lo, hi, 200)
            ys = m * xs + b
            ax.plot(xs, ys, lw=1.6, alpha=0.9, color=col,
                    label=f"{name} fit: y={m:0.3f}x+{b:0.3f} | R²={r2:0.3f}, MAE={mae:0.3f}, ρ={rho:0.3f}")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, alpha=0.25); ax.legend(loc="best", fontsize=8)

def _plot_calibrated(ax, x, series, style, title, xlab, ylab, clip=None):
    # Compute inverse-calibrated predictions y_adj = (y - b)/a for each series
    lo = float(np.nanmin([np.nanmin(x)] + [np.nanmin(v) for v in series.values() if np.isfinite(v).any()]))
    hi = float(np.nanmax([np.nanmax(x)] + [np.nanmax(v) for v in series.values() if np.isfinite(v).any()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    _diag(ax, lo, hi)

    for name, y in series.items():
        col, marker = style[name]
        a = _fit_y_on_x(x, y)
        if a is None:
            continue
        m, b, _, _, _, mask = a
        # Avoid division by zero; if slope ~ 0, skip calibration
        if abs(m) < 1e-8:
            y_adj = y.copy()
            ok = np.isfinite(x) & np.isfinite(y_adj)
        else:
            y_adj = (y - b) / m
            ok = mask  # same mask used for the fit
        if clip is not None:
            y_adj = np.clip(y_adj, clip[0], clip[1])
        r2, mae, rho = _metrics_vs_identity(x, y_adj)
        ax.scatter(x[ok], y_adj[ok], s=28, alpha=0.85, color=col, marker=marker,
                   label=f"{name} (cal): R²={r2:0.3f}, MAE={mae:0.3f}, ρ={rho:0.3f}")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(True, alpha=0.25); ax.legend(loc="best", fontsize=8)

# -------------------- HOVER --------------------
def hover_tables(C):
    tables_dir = C["paths"]["outputs_tables_dir"]
    mpath = C["paths"]["master_parquet"]
    m = pd.read_parquet(mpath) if os.path.exists(mpath) else None
    base = pd.DataFrame({"filename": m["filename"].unique()}) if m is not None and "filename" in m.columns else None
    if base is None:
        return None

    exp = _per_prop_mean(m, "prop_efficiency_mean") if "prop_efficiency_mean" in m.columns else None
    mlp = os.path.join(tables_dir, "hover_predictions.csv")
    ml = pd.read_csv(mlp)[["filename", "y_pred"]].rename(columns={"y_pred": "eta_ml"}) if os.path.exists(mlp) else None
    bemt_p = os.path.join(tables_dir, "bemt_avg_prior.csv")
    bemt = pd.read_csv(bemt_p)[["filename", "eta_bemt_mean"]] if os.path.exists(bemt_p) else None
    cfd = _cfd_surrogate_on_doe(C, mode="vtol")
    cfd = cfd.rename(columns={"pred": "eta_cfd_vtol"}) if not cfd.empty else None

    tab = base.copy()
    if exp is not None:  tab = tab.merge(exp.rename(columns={"prop_efficiency_mean_mean": "eta_exp"}), on="filename", how="left")
    if ml is not None:   tab = tab.merge(ml, on="filename", how="left")
    if bemt is not None: tab = tab.merge(bemt, on="filename", how="left")
    if cfd is not None:  tab = tab.merge(cfd, on="filename", how="left")

    # Reference: experimental if available else ML
    tab["ref"] = tab["eta_exp"].where(tab["eta_exp"].notna(), tab["eta_ml"])
    return tab

def hover_plots(C):
    plots_dir = C["paths"]["outputs_plots_dir"]; os.makedirs(plots_dir, exist_ok=True)
    tab = hover_tables(C)
    if tab is None or tab["ref"].notna().sum() < 2:
        print("Hover parity: not enough data.")
        return

    x = tab["ref"].to_numpy(float)
    series = {
        "ML": tab.get("eta_ml", pd.Series(np.nan, index=tab.index)).to_numpy(float),
        "CFD (surrogate)": tab.get("eta_cfd_vtol", pd.Series(np.nan, index=tab.index)).to_numpy(float),
        "BEMT (avg)": tab.get("eta_bemt_mean", pd.Series(np.nan, index=tab.index)).to_numpy(float),
    }
    style = {"ML": ("C0", "o"), "CFD (surrogate)": ("C2", "^"), "BEMT (avg)": ("C1", "s")}

    # RAW
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), constrained_layout=True)
    _plot_raw(ax, x, series, style,
              "Hover parity (η) — raw",
              "Reference (experimental η̄ if available, else ML)", "Prediction (η̄)",
              clip=(0.0, 1.0))
    fig.savefig(os.path.join(plots_dir, "parity_hover_raw.pdf"), dpi=200)
    plt.close(fig)

    # CALIBRATED (inverse to y=x)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), constrained_layout=True)
    _plot_calibrated(ax, x, series, style,
                     "Hover parity (η) — inverse-calibrated to y = x",
                     "Reference (experimental η̄ if available, else ML)", "Prediction (calibrated η̄)",
                     clip=(0.0, 1.0))
    fig.savefig(os.path.join(plots_dir, "parity_hover_cal.pdf"), dpi=200)
    plt.close(fig)
    print("Wrote hover parity plots.")

# -------------------- CRUISE --------------------
def cruise_tables(C):
    tables_dir = C["paths"]["outputs_tables_dir"]
    mpath = C["paths"]["master_parquet"]
    m = pd.read_parquet(mpath) if os.path.exists(mpath) else None
    base = pd.DataFrame({"filename": m["filename"].unique()}) if m is not None and "filename" in m.columns else None
    if base is None:
        return None

    exp = _per_prop_mean(m, "ld_cruise") if "ld_cruise" in m.columns else None
    mlp = os.path.join(tables_dir, "cruise_predictions.csv")
    ml = pd.read_csv(mlp)[["filename", "y_pred"]].rename(columns={"y_pred": "ld_ml"}) if os.path.exists(mlp) else None
    llp = os.path.join(tables_dir, "ll_cruise_prior.csv")
    ll = pd.read_csv(llp)[["filename", "LD_ll"]] if os.path.exists(llp) else None
    cfd = _cfd_surrogate_on_doe(C, mode="cruise")
    cfd = cfd.rename(columns={"pred": "LD_cfd"}) if not cfd.empty else None

    tab = base.copy()
    if exp is not None: tab = tab.merge(exp.rename(columns={"ld_cruise_mean": "ld_exp"}), on="filename", how="left")
    if ml is not None:  tab = tab.merge(ml, on="filename", how="left")
    if ll is not None:  tab = tab.merge(ll, on="filename", how="left")
    if cfd is not None: tab = tab.merge(cfd, on="filename", how="left")

    tab["ref"] = tab["ld_exp"].where(tab["ld_exp"].notna(), tab["ld_ml"])
    return tab

def cruise_plots(C):
    plots_dir = C["paths"]["outputs_plots_dir"]; os.makedirs(plots_dir, exist_ok=True)
    tab = cruise_tables(C)
    if tab is None or tab["ref"].notna().sum() < 2:
        print("Cruise parity: not enough data.")
        return

    x = tab["ref"].to_numpy(float)
    series = {
        "ML": tab.get("ld_ml", pd.Series(np.nan, index=tab.index)).to_numpy(float),
        "CFD (surrogate)": tab.get("LD_cfd", pd.Series(np.nan, index=tab.index)).to_numpy(float),
        "LL prior": tab.get("LD_ll", pd.Series(np.nan, index=tab.index)).to_numpy(float),
    }
    style = {"ML": ("C0", "o"), "CFD (surrogate)": ("C2", "^"), "LL prior": ("C1", "s")}

    # RAW
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), constrained_layout=True)
    _plot_raw(ax, x, series, style,
              "Cruise parity (L/D) — raw",
              "Reference (experimental L/D if available, else ML)", "Prediction (L/D)")
    fig.savefig(os.path.join(plots_dir, "parity_cruise_raw.pdf"), dpi=200)
    plt.close(fig)

    # CALIBRATED
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5), constrained_layout=True)
    _plot_calibrated(ax, x, series, style,
                     "Cruise parity (L/D) — inverse-calibrated to y = x",
                     "Reference (experimental L/D if available, else ML)", "Prediction (calibrated L/D)")
    fig.savefig(os.path.join(plots_dir, "parity_cruise_cal.pdf"), dpi=200)
    plt.close(fig)
    print("Wrote cruise parity plots.")

# -------------------- main --------------------
def main():
    C = load_cfg()
    hover_plots(C)
    cruise_plots(C)

if __name__ == "__main__":
    main()
