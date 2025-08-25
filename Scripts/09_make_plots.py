# 09_make_plots.py
# Consolidated parity plots (hover & cruise) with calibration lines.
# Writes:
#   Experiment/outputs/plots/parity_hover_vs_ref.pdf
#   Experiment/outputs/plots/parity_cruise_vs_ref.pdf
#
# Notes:
# - X-axis is experimental if available; otherwise ML becomes the reference.
# - Each panel shows a best-fit line y = a x + b with (R², MAE) for that fit.
# - CFD is mapped to DOE filenames via a quick geometry-only surrogate.

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

# ---------- small helpers ----------
def _per_prop_mean(m, col):
    g = m.dropna(subset=["filename", col]).groupby("filename")[col].mean().reset_index()
    return g.rename(columns={col: f"{col}_mean"})

def _diag(ax, lo, hi): ax.plot([lo,hi],[lo,hi], ls="--", lw=1.1, alpha=0.6, label="y = x")

def _fit_line(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    X = np.column_stack([x[m], np.ones(m.sum())])
    a, b = np.linalg.lstsq(X, y[m], rcond=None)[0]
    yhat = a * x[m] + b
    # R² and MAE with respect to the fitted line
    ss_res = np.sum((y[m] - yhat)**2)
    ss_tot = np.sum((y[m] - np.mean(y[m]))**2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y[m] - yhat)))
    return float(a), float(b), float(r2), float(mae), x[m], y[m]

def _panel(ax, x, y, title, xlabel, ylabel):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        ax.set_title(f"{title} (no data)"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); return
    xx, yy = x[ok], y[ok]
    lo = float(np.nanmin([xx.min(), yy.min()])); hi = float(np.nanmax([xx.max(), yy.max()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    pad = 0.05*(hi-lo)
    lo, hi = lo - pad, hi + pad

    ax.scatter(xx, yy, s=28, alpha=0.8, label="data")
    _diag(ax, lo, hi)

    fit = _fit_line(xx, yy)
    if fit is not None:
        a, b, r2, mae, xf, yf = fit
        xs = np.linspace(lo, hi, 100)
        ys = a*xs + b
        ax.plot(xs, ys, lw=1.6, alpha=0.9, label=f"fit: y = {a:0.3f}x + {b:0.3f}\nR²={r2:0.3f}, MAE={mae:0.3f}")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

def _cfd_surrogate_on_doe(C, mode):
    """Train geometry-only HGBR on CFD master and predict for DOE filenames."""
    gcols = C["geometry_cols"]
    doe = pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename").dropna(subset=gcols)
    for c in gcols: doe[c] = pd.to_numeric(doe[c], errors="coerce")

    cfd_path = os.path.join(C["paths"]["outputs_tables_dir"], "cfd_master.csv")
    if not os.path.exists(cfd_path) or doe.empty:
        return pd.DataFrame(columns=["filename", "pred"])

    cfd = pd.read_csv(cfd_path)
    need = gcols + (["eta_cfd"] if mode=="vtol" else ["LD_cfd"])
    if "mode" not in cfd.columns or not set(need).issubset(cfd.columns):
        return pd.DataFrame(columns=["filename", "pred"])

    tgt = "eta_cfd" if mode=="vtol" else "LD_cfd"
    tab = cfd[cfd["mode"]==mode].dropna(subset=gcols+[tgt]).copy()
    for c in gcols: tab[c] = pd.to_numeric(tab[c], errors="coerce")
    tab = tab.dropna(subset=gcols+[tgt])
    if len(tab) < 10:
        return pd.DataFrame(columns=["filename", "pred"])

    X = tab[gcols].to_numpy(float); y = tab[tgt].to_numpy(float)
    mdl = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42).fit(X, y)

    pred = mdl.predict(doe[gcols].to_numpy(float))
    out = doe[["filename"]].copy()
    out["pred"] = pred
    return out

# ---------- HOVER ----------
def hover_parity(C, plots_dir, tables_dir):
    mpath = C["paths"]["master_parquet"]
    m = pd.read_parquet(mpath) if os.path.exists(mpath) else None
    base = pd.DataFrame({"filename": m["filename"].unique()}) if m is not None and "filename" in m.columns else None
    if base is None:
        print("No master dataset; skip hover parity."); return

    # Experimental (x reference if available)
    exp = _per_prop_mean(m, "prop_efficiency_mean") if "prop_efficiency_mean" in m.columns else None

    # ML predictions (per-prop LOPO)
    mlp = os.path.join(tables_dir, "hover_predictions.csv")
    ml = pd.read_csv(mlp)[["filename","y_pred"]].rename(columns={"y_pred":"eta_ml"}) if os.path.exists(mlp) else None

    # BEMT avg prior
    bemt_p = os.path.join(tables_dir, "bemt_avg_prior.csv")
    bemt = pd.read_csv(bemt_p)[["filename","eta_bemt_mean"]] if os.path.exists(bemt_p) else None

    # CFD surrogate mapped to DOE filenames
    cfd_on_doe = _cfd_surrogate_on_doe(C, mode="vtol")
    cfd = cfd_on_doe.rename(columns={"pred":"eta_cfd_vtol"}) if not cfd_on_doe.empty else None

    tab = base.copy()
    if exp is not None:  tab = tab.merge(exp.rename(columns={"prop_efficiency_mean_mean":"eta_exp"}), on="filename", how="left")
    if ml  is not None:  tab = tab.merge(ml, on="filename", how="left")
    if bemt is not None: tab = tab.merge(bemt, on="filename", how="left")
    if cfd is not None:  tab = tab.merge(cfd, on="filename", how="left")

    # reference x
    xref = tab["eta_exp"] if ("eta_exp" in tab.columns and tab["eta_exp"].notna().any()) else tab.get("eta_ml", pd.Series(np.nan, index=tab.index))
    y_ml   = tab.get("eta_ml", pd.Series(np.nan, index=tab.index)).to_numpy(float)
    y_bemt = tab.get("eta_bemt_mean", pd.Series(np.nan, index=tab.index)).to_numpy(float)
    y_cfd  = tab.get("eta_cfd_vtol",  pd.Series(np.nan, index=tab.index)).to_numpy(float)
    xref   = xref.to_numpy(float)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    _panel(axs[0], xref, y_ml,   "ML vs Reference (hover)",      "Reference η̄", "Predicted η̄")
    _panel(axs[1], xref, y_bemt, "BEMT avg vs Reference (hover)","Reference η̄", "BEMT η̄")
    _panel(axs[2], xref, y_cfd,  "CFD surrogate vs Reference",   "Reference η̄", "CFD η̄")
    fout = os.path.join(plots_dir, "parity_hover_vs_ref.pdf")
    fig.savefig(fout, dpi=200)
    print("Wrote", fout)

# ---------- CRUISE ----------
def cruise_parity(C, plots_dir, tables_dir):
    mpath = C["paths"]["master_parquet"]
    m = pd.read_parquet(mpath) if os.path.exists(mpath) else None
    base = pd.DataFrame({"filename": m["filename"].unique()}) if m is not None and "filename" in m.columns else None
    if base is None:
        print("No master dataset; skip cruise parity."); return

    # Experimental (x reference if available)
    exp = _per_prop_mean(m, "ld_cruise") if "ld_cruise" in m.columns else None

    # ML predictions (per-prop LOPO)
    mlp = os.path.join(tables_dir, "cruise_predictions.csv")
    ml = pd.read_csv(mlp)[["filename","y_pred"]].rename(columns={"y_pred":"ld_ml"}) if os.path.exists(mlp) else None

    # Lifting-line prior
    llp = os.path.join(tables_dir, "ll_cruise_prior.csv")
    ll = pd.read_csv(llp)[["filename","LD_ll"]] if os.path.exists(llp) else None

    # CFD surrogate mapped to DOE filenames
    cfd_on_doe = _cfd_surrogate_on_doe(C, mode="cruise")
    cfd = cfd_on_doe.rename(columns={"pred":"LD_cfd"}) if not cfd_on_doe.empty else None

    tab = base.copy()
    if exp is not None: tab = tab.merge(exp.rename(columns={"ld_cruise_mean":"ld_exp"}), on="filename", how="left")
    if ml  is not None: tab = tab.merge(ml, on="filename", how="left")
    if ll  is not None: tab = tab.merge(ll, on="filename", how="left")
    if cfd is not None: tab = tab.merge(cfd, on="filename", how="left")

    # reference x
    xref = tab["ld_exp"] if ("ld_exp" in tab.columns and tab["ld_exp"].notna().any()) else tab.get("ld_ml", pd.Series(np.nan, index=tab.index))
    y_ml = tab.get("ld_ml", pd.Series(np.nan, index=tab.index)).to_numpy(float)
    y_ll = tab.get("LD_ll", pd.Series(np.nan, index=tab.index)).to_numpy(float)
    y_cf = tab.get("LD_cfd", pd.Series(np.nan, index=tab.index)).to_numpy(float)
    xref = xref.to_numpy(float)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    _panel(axs[0], xref, y_ml, "ML vs Reference (cruise)",  "Reference L/D", "Predicted L/D")
    _panel(axs[1], xref, y_ll, "LL prior vs Reference",     "Reference L/D", "LL L/D")
    _panel(axs[2], xref, y_cf, "CFD surrogate vs Reference","Reference L/D", "CFD L/D")
    fout = os.path.join(plots_dir, "parity_cruise_vs_ref.pdf")
    fig.savefig(fout, dpi=200)
    print("Wrote", fout)

def main():
    C = load_cfg()
    plots_dir  = C["paths"]["outputs_plots_dir"];  os.makedirs(plots_dir, exist_ok=True)
    tables_dir = C["paths"]["outputs_tables_dir"]; os.makedirs(tables_dir, exist_ok=True)
    hover_parity(C, plots_dir, tables_dir)
    cruise_parity(C, plots_dir, tables_dir)

if __name__=="__main__":
    main()
