# 99_build_figs.py
# Figures for diagnostics, LOGOCV, case studies, Pareto/UCB, and Active-Learning update.
#
# Reads (from config.yaml paths):
#   - master_parquet
#   - outputs_tables/01_hover_step_diagnostics.csv
#   - outputs_tables/03_loocv_results_hover.csv
#   - outputs_tables/03_loocv_results_cruise.csv
#   - outputs_tables/03_interval_calibration.csv
#   - outputs_tables/03_hover_case_curve.csv
#   - outputs_tables/03_hover_case_measured.csv
#   - outputs_tables/03_cruise_case_curve.csv
#   - outputs_tables/03_cruise_case_measured.csv
#   - outputs_tables/04_optimization_results.csv
#   - outputs_tables/05_baseline_reoperate.csv
#   - outputs_tables/05_baseline_ucb_sweep.csv
#   - outputs_tables/06_active_learning_update.csv
#   - outputs_tables/06_hover_std_field.csv
#
# Writes (to outputs_plots):
#   - Fig_DesignSpace_2D.pdf
#   - Fig_DesignSpace_2D_Unit.pdf  (unit-scaled, optional coverage view)
#   - Fig_Noise_SkewKurt.pdf
#   - Fig_EstimatorMix.pdf
#   - Fig_VarEta_vs_v.pdf
#   - Fig_NegTfrac_vs_v.pdf
#   - Fig_Vibration_vs_VarEta.pdf
#   - Fig_LOOCV_Hover_StdResidHist.pdf
#   - Fig_LOOCV_Hover_AbsRes_vs_Std.pdf
#   - Fig_LOOCV_Cruise_Parity.pdf
#   - Fig_LOOCV_CoverageBars.pdf
#   - Fig_Case_Hover.pdf
#   - Fig_Case_Cruise.pdf
#   - Fig_Pareto_UCB.pdf
#   - Fig_Pareto_Contribs.pdf
#   - Fig_Pareto_Geometry_vs_w.pdf
#   - Fig_AL_U_Global.pdf
#   - Fig_AL_Std_BeforeAfter.pdf
#   - Fig_AL_Std_ReductionHist.pdf
#   - Fig_AL_Std_Reduction_vs_v_alpha.pdf

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import path_utils

# ---------------------------
# Matplotlib (IEEE-ish, grayscale)
# ---------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------
# IO helpers
# ---------------------------
def _try_read(path):
    try:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        if str(p).lower().endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return None

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _save(fig, out_dir, name):
    _ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / name, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Core utilities
# ---------------------------
def _feature_labels_plain():
    # Plain-text labels (omit symbols; include units where helpful)
    return {
        "AR": "Aspect ratio",
        "lambda": "Taper ratio",
        "twist": "Twist (deg)",
        "alpha": "Root angle of attack (deg)",
        "v": "Tip speed (m/s)",
        "performance": "Performance",
        "performance_variance": "Efficiency variance",
    }

def _joined_hover_diag(master, tables_dir):
    diag = _try_read(Path(tables_dir) / "01_hover_step_diagnostics.csv")
    if diag is None or master is None:
        return None
    keys = [k for k in ["k_id","esc_signal"] if k in diag.columns and k in master.columns]
    if not keys:
        return None
    use_master = master[keys + ["flight_mode","performance_variance"]].copy()
    d = diag.merge(use_master, on=keys, how="inner")
    return d[d["flight_mode"]=="hover"].copy()

def _read_loocv(tables_dir):
    h = _try_read(Path(tables_dir) / "03_loocv_results_hover.csv")
    c = _try_read(Path(tables_dir) / "03_loocv_results_cruise.csv")
    cal = _try_read(Path(tables_dir) / "03_interval_calibration.csv")
    return h, c, cal

def _read_case_csvs(tables_dir):
    hc = _try_read(Path(tables_dir) / "03_hover_case_curve.csv")
    hm = _try_read(Path(tables_dir) / "03_hover_case_measured.csv")
    cc = _try_read(Path(tables_dir) / "03_cruise_case_curve.csv")
    cm = _try_read(Path(tables_dir) / "03_cruise_case_measured.csv")
    return hc, hm, cc, cm

def _read_pareto_and_baselines(tables_dir):
    opt = _try_read(Path(tables_dir) / "04_optimization_results.csv")
    base_reop = _try_read(Path(tables_dir) / "05_baseline_reoperate.csv")
    base_sweep = _try_read(Path(tables_dir) / "05_baseline_ucb_sweep.csv")
    return opt, base_reop, base_sweep

def _read_active_learning(tables_dir):
    upd = _try_read(Path(tables_dir) / "06_active_learning_update.csv")
    field = _try_read(Path(tables_dir) / "06_hover_std_field.csv")
    return upd, field

def _lambda95(cal_df, mode):
    if cal_df is None or cal_df.empty or "lambda_95" not in cal_df.columns:
        return 1.0
    r = cal_df[cal_df["mode"].astype(str).str.lower() == mode]
    if r.empty:
        return 1.0
    try:
        val = float(r["lambda_95"].iloc[0])
    except Exception:
        return 1.0
    return val if np.isfinite(val) else 1.0

# ---------------------------
# Figures: Design space & diagnostics
# ---------------------------
def fig_design_space_2d(df_master, out_dir):
    """Pairwise grid for (AR, lambda, twist) with correct axis labeling and lambda ticks."""
    if df_master is None or df_master.empty:
        return
    cols = ["AR","lambda","twist"]
    if any(c not in df_master.columns for c in cols):
        return

    labels = {
        "AR": "Aspect ratio [18.75,31.25]",
        "lambda": "Taper ratio [0.50,1.00]",
        "twist": "Twist (deg) [-15,-3]"
    }

    d = df_master.drop_duplicates(subset=["flight_mode","AR","lambda","twist"]).copy()
    d_h = d[d["flight_mode"]=="hover"].copy()
    d_c = d[d["flight_mode"]=="cruise"].copy()

    # Small jitter to reduce overplotting (esp. lambda)
    rng = np.random.default_rng(0)
    for df_ in (d_h, d_c):
        if not df_.empty:
            df_.loc[:, "lambda"] = df_["lambda"].astype(float) + rng.normal(0.0, 0.002, size=len(df_))

    # Fixed axis ranges
    mins = {c: float(d[c].min()) for c in cols}
    maxs = {c: float(d[c].max()) for c in cols}
    # Clamp lambda visually to design bounds to improve readability
    mins["lambda"], maxs["lambda"] = 0.50, 1.00

    fig, axes = plt.subplots(3, 3, figsize=(6.2, 6.0))
    # Important: columns correspond to X (horizontal), rows to Y (vertical)
    for i, yi in enumerate(cols):          # y variable per row
        for j, xj in enumerate(cols):      # x variable per column
            ax = axes[i, j]
            if i == j:
                # diagonal: show marginals (hover outline; cruise filled)
                ax.hist(d_h[xj].dropna(), bins=25, histtype="step", color="black")
                ax.hist(d_c[xj].dropna(), bins=25, histtype="stepfilled", alpha=0.25, color="0.6")
            else:
                ax.scatter(d_h[xj], d_h[yi], s=10, marker="o", facecolors="none", edgecolors="black", alpha=0.8, label="Hover")
                ax.scatter(d_c[xj], d_c[yi], s=12, marker="x", c="0.35", alpha=0.8, label="Cruise")
                if (i, j) == (0, 1):
                    ax.legend(frameon=False, loc="best")

            ax.set_xlim(mins[xj], maxs[xj])
            ax.set_ylim(mins[yi], maxs[yi])

            # Lambda tick formatting (both x and y where applicable)
            if xj == "lambda":
                ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(np.linspace(0.50, 1.00, 6)))
                ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
            if yi == "lambda":
                ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(np.linspace(0.50, 1.00, 6)))
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

            # Label only outer axes
            if i == len(cols)-1:
                ax.set_xlabel(labels[xj])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[yi])
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    _save(fig, out_dir, "Fig_DesignSpace_2D.pdf")

    # Optional: unit-scaled coverage view
    z = d.copy()
    for c in cols:
        lo, hi = mins[c], maxs[c]
        z[c] = (z[c].astype(float) - lo) / max(hi - lo, 1e-12)
    zh = z[z["flight_mode"]=="hover"]; zc = z[z["flight_mode"]=="cruise"]

    fig2, axes2 = plt.subplots(3, 3, figsize=(6.2, 6.0))
    for i, yi in enumerate(cols):
        for j, xj in enumerate(cols):
            ax = axes2[i, j]
            if i == j:
                ax.hist(zh[xj].dropna(), bins=25, histtype="step", color="black")
                ax.hist(zc[xj].dropna(), bins=25, histtype="stepfilled", alpha=0.25, color="0.6")
            else:
                ax.scatter(zh[xj], zh[yi], s=10, marker="o", facecolors="none", edgecolors="black", alpha=0.8)
                ax.scatter(zc[xj], zc[yi], s=12, marker="x", c="0.35", alpha=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == len(cols)-1: ax.set_xlabel(f"{xj} (scaled)")
            if j == 0: ax.set_ylabel(f"{yi} (scaled)")
    plt.tight_layout()
    _save(fig2, out_dir, "Fig_DesignSpace_2D_Unit.pdf")

def fig_noise_skew_kurt(df_diag, out_dir):
    if df_diag is None or df_diag.empty:
        return
    channels = []
    for base in ["T","Q","RPM","VIB"]:
        if f"{base}_skew" in df_diag.columns and f"{base}_kurt" in df_diag.columns:
            channels.append(base)
    if not channels:
        return

    fig, axes = plt.subplots(2, len(channels), figsize=(3.0*len(channels), 3.8))
    if len(channels) == 1:
        axes = np.array([[axes[0]], [axes[1]]]) if isinstance(axes, np.ndarray) else np.array([[axes],[axes]])

    for j, base in enumerate(channels):
        sk = df_diag[f"{base}_skew"].dropna().to_numpy()
        ku = df_diag[f"{base}_kurt"].dropna().to_numpy()
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        ax_top.hist(sk, bins=30, color="0.35")
        ax_bot.hist(ku, bins=30, color="0.6")
        if j == 0:
            ax_top.set_ylabel("Skewness")
            ax_bot.set_ylabel("Excess kurtosis")
        ax_bot.set_xlabel({"T":"Thrust","Q":"Torque","RPM":"Angular Speed","VIB":"Vibration"}[base])

    plt.tight_layout()
    _save(fig, out_dir, "Fig_Noise_SkewKurt.pdf")

def fig_estimator_mix(df_diag, out_dir):
    if df_diag is None or df_diag.empty:
        return
    methods = ["mean","trim10","trim20","median"]
    channels = [c for c in ["T_method","Q_method","RPM_method","VIB_method"] if c in df_diag.columns]
    if not channels:
        return

    counts = []
    for ch in channels:
        vc = df_diag[ch].value_counts()
        counts.append([int(vc.get(m,0)) for m in methods])
    counts = np.array(counts)

    x = np.arange(len(channels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for i, m in enumerate(methods):
        ax.bar(x + (i-(len(methods)-1)/2)*w, counts[:, i], width=w, label=m, color=str(0.25+0.15*i))
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_method","") for c in channels])
    ax.set_ylabel("Count")
    ax.legend(frameon=False, ncols=len(methods), loc="best")
    _save(fig, out_dir, "Fig_EstimatorMix.pdf")

def fig_var_eta_vs_v(df_master, out_dir):
    if df_master is None or df_master.empty:
        return
    d = df_master[df_master["flight_mode"]=="hover"].copy()
    for col in ["v","performance_variance"]:
        if col not in d.columns: return
    d = d.dropna(subset=["v","performance_variance"])
    if d.empty: return

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(d["v"], d["performance_variance"], s=12, c="0.2", alpha=0.7, marker="o")
    ax.set_xlabel("Tip speed (m/s)")
    ax.set_ylabel("Hover performance variance")
    _save(fig, out_dir, "Fig_VarEta_vs_v.pdf")

def fig_negTfrac_vs_v(df_diag, out_dir):
    if df_diag is None or df_diag.empty:
        return
    for col in ["v","neg_T_frac"]:
        if col not in df_diag.columns: return
    d = df_diag.dropna(subset=["v","neg_T_frac"])
    if d.empty: return
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(d["v"], d["neg_T_frac"], s=12, c="0.4", alpha=0.7, marker="o")
    ax.set_xlabel("Tip speed (m/s)")
    ax.set_ylabel("Fraction with negative thrust")
    _save(fig, out_dir, "Fig_NegTfrac_vs_v.pdf")

def fig_vibration_vs_eta_var(df_diag_or_joined, df_master, out_dir):
    if df_diag_or_joined is None or df_master is None:
        return
    d = df_diag_or_joined.copy()
    if d.empty:
        return
    if "performance_variance" not in d.columns:
        keys = [k for k in ["k_id","esc_signal"] if k in d.columns and k in df_master.columns]
        if not keys:
            return
        use_master = df_master[keys + ["flight_mode","performance_variance"]].copy()
        d = d.merge(use_master, on=keys, how="inner")
    d = d[d["flight_mode"]=="hover"].copy()
    if d.empty or "performance_variance" not in d.columns:
        return

    if "VIB_loc" in d.columns and np.isfinite(d["VIB_loc"]).any():
        x = d["VIB_loc"].to_numpy()
        xlabel = "Vibration"
    else:
        if not {"RPM_loc","RPM_std_raw"}.issubset(d.columns):
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            jitter = np.where(d["RPM_loc"]>0, 100.0*d["RPM_std_raw"]/d["RPM_loc"], np.nan)
        x = jitter
        xlabel = "RPM jitter (%)"
    y = d["performance_variance"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return
    if m.sum() >= 3:
        X1 = np.vstack([x[m], np.ones(m.sum())]).T
        beta, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
        xx = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 100)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.scatter(x[m], y[m], s=12, c="0.2", alpha=0.7, marker="o")
    if m.sum() >= 3:
        ax.plot(xx, beta[0]*xx + beta[1], color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hover performance variance")
    _save(fig, out_dir, "Fig_Vibration_vs_VarEta.pdf")

# ---------------------------
# LOGOCV & case studies
# ---------------------------
def _std_residuals(y, mu, s, meas_var=None):
    s = np.asarray(s, float)
    s = np.maximum(s, 1e-12)
    if meas_var is not None:
        s = np.sqrt(s**2 + np.maximum(np.asarray(meas_var, float), 0.0))
    r = (np.asarray(y, float) - np.asarray(mu, float)) / s
    return r

def fig_loocv_hover_hist(df_h, out_dir):
    """Hover LOGOCV standardized residual histogram (separate figure)."""
    if df_h is None or df_h.empty:
        return
    need = {"ground_truth","prediction","model_uncertainty_std","measurement_variance"}
    if not need.issubset(df_h.columns):
        return
    r = _std_residuals(df_h["ground_truth"], df_h["prediction"],
                       df_h["model_uncertainty_std"], df_h["measurement_variance"])
    r = r[np.isfinite(r)]
    if r.size == 0:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.hist(r, bins=30, color="0.35")
    ax.axvline(0.0, color="0.1", linewidth=1.0)
    ax.axvline(1.0, color="0.6", linestyle="--", linewidth=1.0)
    ax.axvline(-1.0, color="0.6", linestyle="--", linewidth=1.0)
    ax.axvline(1.96, color="0.75", linestyle=":", linewidth=1.0)
    ax.axvline(-1.96, color="0.75", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Count")
    _save(fig, out_dir, "Fig_LOOCV_Hover_StdResidHist.pdf")

def fig_loocv_hover_absres_vs_std(df_h, out_dir):
    """Hover LOGOCV absolute residual vs predicted std (separate figure)."""
    if df_h is None or df_h.empty:
        return
    need = {"ground_truth","prediction","model_uncertainty_std"}
    if not need.issubset(df_h.columns):
        return
    abs_res = np.abs(df_h["ground_truth"] - df_h["prediction"]).to_numpy()
    std = df_h["model_uncertainty_std"].to_numpy()
    m = np.isfinite(abs_res) & np.isfinite(std)
    if m.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.scatter(std[m], abs_res[m], s=10, c="0.2", alpha=0.6, marker="o")
    ax.set_xlabel("Predicted standard deviation")
    ax.set_ylabel("Absolute residual")
    _save(fig, out_dir, "Fig_LOOCV_Hover_AbsRes_vs_Std.pdf")

def fig_loocv_cruise_parity(df_c, out_dir):
    if df_c is None or df_c.empty:
        return
    need = {"ground_truth","prediction","model_uncertainty_std"}
    if not need.issubset(df_c.columns):
        return
    y = df_c["ground_truth"].to_numpy()
    mu = df_c["prediction"].to_numpy()
    s = df_c["model_uncertainty_std"].to_numpy()
    m = np.isfinite(y) & np.isfinite(mu) & np.isfinite(s)
    if m.sum() == 0: return

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    lo = np.nanmin([y[m].min(), mu[m].min()])
    hi = np.nanmax([y[m].max(), mu[m].max()])
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1.0)
    ax.errorbar(mu[m], y[m], xerr=s[m], fmt="o", ms=3, mfc="none", mec="0.2",
                ecolor="0.5", elinewidth=0.8, capsize=2, alpha=0.8)
    ax.set_xlabel("Predicted lift/drag")
    ax.set_ylabel("Measured lift/drag")
    _save(fig, out_dir, "Fig_LOOCV_Cruise_Parity.pdf")

def fig_loocv_coverage_bars(df_h, df_c, out_dir):
    vals = []
    if df_h is not None and not df_h.empty and {"ground_truth","prediction","model_uncertainty_std","measurement_variance"}.issubset(df_h.columns):
        r_h = _std_residuals(df_h["ground_truth"], df_h["prediction"], df_h["model_uncertainty_std"], df_h["measurement_variance"])
        r_h = r_h[np.isfinite(r_h)]
        if r_h.size:
            vals.append(("Hover 68%", float(np.mean(np.abs(r_h) <= 1.0))))
            vals.append(("Hover 95%", float(np.mean(np.abs(r_h) <= 1.96))))
    if df_c is not None and not df_c.empty and {"ground_truth","prediction","model_uncertainty_std"}.issubset(df_c.columns):
        r_c = _std_residuals(df_c["ground_truth"], df_c["prediction"], df_c["model_uncertainty_std"], None)
        r_c = r_c[np.isfinite(r_c)]
        if r_c.size:
            vals.append(("Cruise 68%", float(np.mean(np.abs(r_c) <= 1.0))))
            vals.append(("Cruise 95%", float(np.mean(np.abs(r_c) <= 1.96))))
    if not vals:
        return

    labels, data = zip(*vals)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, data, width=0.6, color="0.6")
    ax.axhline(0.68, color="0.3", linestyle="--", linewidth=1.0)
    ax.axhline(0.95, color="0.7", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed coverage")
    _save(fig, out_dir, "Fig_LOOCV_CoverageBars.pdf")

# ---------------------------
# Case studies
# ---------------------------
def fig_case_hover(h_curve, h_meas, cal_df, out_dir):
    if h_curve is None or h_curve.empty:
        return
    lam95 = _lambda95(cal_df, "hover")
    if "95_conf_lower" not in h_curve.columns or "95_conf_upper" not in h_curve.columns:
        mu = h_curve["predicted_performance"].to_numpy()
        s  = h_curve["model_uncertainty_std"].to_numpy()
        h_curve["95_conf_lower"] = mu - 1.96 * lam95 * s
        h_curve["95_conf_upper"] = mu + 1.96 * lam95 * s

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = h_curve["v"].to_numpy() if "v" in h_curve.columns else np.arange(len(h_curve))
    mu = h_curve["predicted_performance"].to_numpy()
    lo = h_curve["95_conf_lower"].to_numpy()
    hi = h_curve["95_conf_upper"].to_numpy()

    ax.plot(x, mu, color="0.1", linewidth=1.2)
    ax.fill_between(x, lo, hi, color="0.8", alpha=0.5, linewidth=0)

    if h_meas is not None and not h_meas.empty:
        if "v" in h_meas.columns and "measured_performance" in h_meas.columns:
            if "measurement_variance" in h_meas.columns:
                se = np.sqrt(np.maximum(h_meas["measurement_variance"].to_numpy(), 0.0))
                ax.errorbar(h_meas["v"], h_meas["measured_performance"], yerr=se, fmt="o",
                            ms=3, mfc="none", mec="0.2", ecolor="0.5", elinewidth=0.8, capsize=2, alpha=0.9)
            else:
                ax.scatter(h_meas["v"], h_meas["measured_performance"], s=12, marker="o",
                           facecolors="none", edgecolors="0.2", alpha=0.9)

    ax.set_xlabel("Tip speed (m/s)")
    ax.set_ylabel("Performance")
    _save(fig, out_dir, "Fig_Case_Hover.pdf")

def fig_case_cruise(c_curve, c_meas, cal_df, out_dir):
    if c_curve is None or c_curve.empty:
        return
    lam95 = _lambda95(cal_df, "cruise")
    if "95_conf_lower" not in c_curve.columns or "95_conf_upper" not in c_curve.columns:
        mu = c_curve["predicted_performance"].to_numpy()
        s  = c_curve["model_uncertainty_std"].to_numpy()
        c_curve["95_conf_lower"] = mu - 1.96 * lam95 * s
        c_curve["95_conf_upper"] = mu + 1.96 * lam95 * s

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = c_curve["alpha"].to_numpy() if "alpha" in c_curve.columns else np.arange(len(c_curve))
    mu = c_curve["predicted_performance"].to_numpy()
    lo = c_curve["95_conf_lower"].to_numpy()
    hi = c_curve["95_conf_upper"].to_numpy()

    ax.plot(x, mu, color="0.1", linewidth=1.2)
    ax.fill_between(x, lo, hi, color="0.8", alpha=0.5, linewidth=0)

    if c_meas is not None and not c_meas.empty:
        if "alpha" in c_meas.columns and "measured_performance" in c_meas.columns:
            ax.scatter(c_meas["alpha"], c_meas["measured_performance"], s=12, marker="o",
                       facecolors="none", edgecolors="0.2", alpha=0.9)

    ax.set_xlabel("Root angle of attack (deg)")
    ax.set_ylabel("Performance")
    _save(fig, out_dir, "Fig_Case_Cruise.pdf")

# ---------------------------
# Pareto / UCB
# ---------------------------
def fig_pareto_ucb(df_opt, df_base_sweep, out_dir):
    if df_opt is None or df_opt.empty:
        return
    if "w_hover" not in df_opt.columns or "ucb_total" not in df_opt.columns:
        return
    do = df_opt.sort_values("w_hover").copy()

    have_base = df_base_sweep is not None and not df_base_sweep.empty and {"w_hover","ucb_total"}.issubset(df_base_sweep.columns)
    if have_base:
        db = df_base_sweep.copy()
        grp = db.groupby("w_hover")
        w_vals = sorted(grp.groups.keys())
        base_max = np.array([grp.get_group(w)["ucb_total"].max() for w in w_vals], float)
        base_med = np.array([grp.get_group(w)["ucb_total"].median() for w in w_vals], float)
        base_lo  = np.array([np.quantile(grp.get_group(w)["ucb_total"], 0.10) for w in w_vals], float)
        base_hi  = np.array([np.quantile(grp.get_group(w)["ucb_total"], 0.90) for w in w_vals], float)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    if have_base:
        ax.fill_between(w_vals, base_lo, base_hi, color="0.85", alpha=0.6, linewidth=0)
        ax.plot(w_vals, base_med, color="0.55", linestyle="--", linewidth=1.0, label="Baseline median")
        ax.plot(w_vals, base_max, color="0.35", linestyle="-.", linewidth=1.0, label="Baseline best")
    ax.plot(do["w_hover"], do["ucb_total"], color="0.1", linewidth=1.2, label="Optimized")
    ax.set_xlabel("Hover weight")
    ax.set_ylabel("Calibrated UCB (normalized)")
    ax.legend(frameon=False, loc="best")
    _save(fig, out_dir, "Fig_Pareto_UCB.pdf")

def fig_pareto_contribs(df_opt, out_dir):
    if df_opt is None or df_opt.empty:
        return
    need = {"w_hover","exploit_hover","explore_hover","exploit_cruise","explore_cruise"}
    if not need.issubset(df_opt.columns):
        return
    d = df_opt.sort_values("w_hover")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(d["w_hover"], d["exploit_hover"], color="0.1", linestyle="-",  label="Hover exploit")
    ax.plot(d["w_hover"], d["explore_hover"], color="0.5", linestyle="--", label="Hover explore")
    ax.plot(d["w_hover"], d["exploit_cruise"], color="0.3", linestyle="-",  label="Cruise exploit")
    ax.plot(d["w_hover"], d["explore_cruise"], color="0.7", linestyle="--", label="Cruise explore")
    ax.set_xlabel("Hover weight")
    ax.set_ylabel("Normalized contribution")
    ax.legend(frameon=False, ncols=2, loc="best")
    _save(fig, out_dir, "Fig_Pareto_Contribs.pdf")

def fig_pareto_geometry_vs_w(df_opt, out_dir):
    if df_opt is None or df_opt.empty:
        return
    need = {"w_hover","AR","lambda","twist"}
    if not need.issubset(df_opt.columns):
        return
    d = df_opt.sort_values("w_hover")
    labels = _feature_labels_plain()
    fig, axes = plt.subplots(3, 1, figsize=(3.2, 5.1), sharex=True)
    axes[0].plot(d["w_hover"], d["AR"], color="0.1")
    axes[0].set_ylabel(labels["AR"])
    axes[1].plot(d["w_hover"], d["lambda"], color="0.35")
    axes[1].set_ylabel(labels["lambda"])
    axes[2].plot(d["w_hover"], d["twist"], color="0.6")
    axes[2].set_ylabel(labels["twist"])
    axes[2].set_xlabel("Hover weight")
    axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    axes[1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    _save(fig, out_dir, "Fig_Pareto_Geometry_vs_w.pdf")

def fig_pareto_all_stacked(df_opt, df_base_sweep, out_dir, include_operations=True):
    """
    One vertical, shared-x figure stacking:
      [1] UCB vs w (optimized; optional baseline envelope)
      [2] Contributions (hover/cruise exploit/explore)
      [3] Geometry vs w: AR
      [4] Geometry vs w: lambda
      [5] Geometry vs w: twist
      [6-9] (optional) Operations vs w: alpha_hover, v_hover, alpha_cruise, v_cruise
    Saves: Fig_Pareto_Stacked.pdf
    """
    # Basic checks
    if df_opt is None or df_opt.empty or "w_hover" not in df_opt.columns:
        return

    need_base = {"ucb_total"}
    have_base = (df_base_sweep is not None and
                 not df_base_sweep.empty and
                 {"w_hover","ucb_total"}.issubset(df_base_sweep.columns))

    need_opt = {
        "ucb_total",
        "exploit_hover","explore_hover","exploit_cruise","explore_cruise",
        "AR","lambda","twist",
        "alpha_hover","v_hover","alpha_cruise","v_cruise"
    }
    # We'll plot only what exists (geom & ops lines require their columns)
    do = df_opt.sort_values("w_hover").copy()
    w = do["w_hover"].to_numpy()

    # Row planning
    rows = 5  # UCB + contribs + 3 geometry rows
    if include_operations:
        rows += 4  # alpha_h, v_h, alpha_c, v_c

    # Make the figure (IEEE column width ~3.3–3.6in; set per-row height small)
    height_per_row = 1.0  # inches; bump slightly if labels are crowded
    fig, axes = plt.subplots(rows, 1, figsize=(3.4, height_per_row*rows), sharex=True)
    if rows == 1:
        axes = [axes]
    axes = np.asarray(axes)

    idx = 0
    # -------- [1] UCB vs w (with baseline) --------
    ax = axes[idx]; idx += 1
    if have_base:
        db = df_base_sweep.copy()
        grp = db.groupby("w_hover")
        w_vals = sorted(grp.groups.keys())
        base_med = np.array([grp.get_group(w_)["ucb_total"].median() for w_ in w_vals], float)
        base_lo  = np.array([np.quantile(grp.get_group(w_)["ucb_total"], 0.10) for w_ in w_vals], float)
        base_hi  = np.array([np.quantile(grp.get_group(w_)["ucb_total"], 0.90) for w_ in w_vals], float)
        ax.fill_between(w_vals, base_lo, base_hi, color="0.85", alpha=0.6, linewidth=0)
        ax.plot(w_vals, base_med, color="0.55", linestyle="--", linewidth=1.0, label="Baseline median")
    ax.plot(w, do["ucb_total"], color="0.1", linewidth=1.2, label="Optimized")
    ax.set_ylabel("UCB (norm.)")
    if have_base:
        ax.legend(frameon=False, loc="best")

    # -------- [2] Contributions --------
    ax = axes[idx]; idx += 1
    ok = {"exploit_hover","explore_hover","exploit_cruise","explore_cruise"}.issubset(do.columns)
    if ok:
        ax.plot(w, do["exploit_hover"], color="0.1", linestyle="-",  label="Hover exploit")
        ax.plot(w, do["explore_hover"], color="0.5", linestyle="--", label="Hover explore")
        ax.plot(w, do["exploit_cruise"], color="0.3", linestyle="-",  label="Cruise exploit")
        ax.plot(w, do["explore_cruise"], color="0.7", linestyle="--", label="Cruise explore")
        ax.set_ylabel("Contrib. (norm.)")
        ax.legend(frameon=False, ncols=2, loc="best")

    # -------- [3-5] Geometry: AR, lambda, twist --------
    for col, color, ylab in [
        ("AR", "0.1", "Aspect ratio"),
        ("lambda", "0.35", "Taper ratio"),
        ("twist", "0.6", "Twist (deg)"),
    ]:
        ax = axes[idx]; idx += 1
        if col in do.columns:
            ax.plot(w, do[col], color=color)
            ax.set_ylabel(ylab)
            if col == "lambda":
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    # -------- [6-9] Operations: alpha_h, v_h, alpha_c, v_c --------
    if include_operations:
        for col, color, ylab in [
            ("alpha_hover",  "0.15", "Hover angle of attack (deg)"),
            ("v_hover",      "0.35", "Hover tip speed (m/s)"),
            ("alpha_cruise", "0.5",  "Cruise angle of attack (deg)"),
            ("v_cruise",     "0.7",  "Cruise speed (m/s)"),
        ]:
            ax = axes[idx]; idx += 1
            if col in do.columns:
                ax.plot(w, do[col], color=color)
                ax.set_ylabel(ylab)

    # Shared x
    axes[-1].set_xlabel("Hover weight")
    axes[-1].set_xlim(0.0, 1.0)

    plt.tight_layout(h_pad=0.2)
    _save(fig, out_dir, "Fig_Pareto_Stacked.pdf")

# ---------------------------
# Active Learning (06)
# ---------------------------
def fig_al_u_global(df_update, out_dir):
    """Bar plot of global mean-normalized uncertainty before vs after; text shows percent reduction."""
    if df_update is None or df_update.empty:
        return
    r = df_update.iloc[0]
    ub = float(r.get("U_before_percent", np.nan))
    ua = float(r.get("U_after_percent", np.nan))
    red = float(r.get("reduction_percent", np.nan))

    vals = [ub, ua]
    labels = ["Before", "After"]
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    ax.bar(np.arange(2), vals, width=0.6, color=["0.6","0.3"])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Global mean-normalized std (%)")
    if np.isfinite(red):
        ax.text(0.5, max(vals)*1.02, f"Reduction: {red:.1f}%", ha="center", va="bottom")
    _save(fig, out_dir, "Fig_AL_U_Global.pdf")

def fig_al_std_before_after(df_field, out_dir):
    """Scatter of std_before vs std_after with y=x reference."""
    if df_field is None or df_field.empty:
        return
    if not {"std_before","std_after"}.issubset(df_field.columns):
        return
    s0 = df_field["std_before"].to_numpy(float)
    s1 = df_field["std_after"].to_numpy(float)
    m = np.isfinite(s0) & np.isfinite(s1)
    if m.sum() == 0:
        return
    lo = float(np.nanmin([s0[m].min(), s1[m].min()]))
    hi = float(np.nanmax([s0[m].max(), s1[m].max()]))

    fig, ax = plt.subplots(figsize=(3.1, 2.8))
    ax.plot([lo, hi], [lo, hi], color="0.7", linestyle="--", linewidth=1.0)
    ax.scatter(s0[m], s1[m], s=8, c="0.2", alpha=0.6, marker="o")
    ax.set_xlabel("Predicted std (before)")
    ax.set_ylabel("Predicted std (after)")
    _save(fig, out_dir, "Fig_AL_Std_BeforeAfter.pdf")

def fig_al_std_reduction_hist(df_field, out_dir):
    """Histogram of relative std reduction percent across sampled points."""
    if df_field is None or df_field.empty:
        return
    col = "std_rel_reduction_percent"
    if col not in df_field.columns:
        return
    x = df_field[col].to_numpy(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    ax.hist(x, bins=40, color="0.5")
    ax.set_xlabel("Std relative reduction (%)")
    ax.set_ylabel("Count")
    _save(fig, out_dir, "Fig_AL_Std_ReductionHist.pdf")

def fig_al_std_reduction_vs_v_alpha(df_field, out_dir):
    """Relative std reduction vs v and vs alpha (two stacked panels)."""
    if df_field is None or df_field.empty:
        return
    need = {"v","alpha","std_rel_reduction_percent"}
    if not need.issubset(df_field.columns):
        return
    m = np.isfinite(df_field["v"]) & np.isfinite(df_field["alpha"]) & np.isfinite(df_field["std_rel_reduction_percent"])
    if m.sum() == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 5.0), sharex=False)
    # vs v
    axes[0].scatter(df_field.loc[m,"v"], df_field.loc[m,"std_rel_reduction_percent"], s=8, c="0.2", alpha=0.6)
    axes[0].set_xlabel("Tip speed (m/s)")
    axes[0].set_ylabel("Std reduction (%)")
    # vs alpha
    axes[1].scatter(df_field.loc[m,"alpha"], df_field.loc[m,"std_rel_reduction_percent"], s=8, c="0.35", alpha=0.6)
    axes[1].set_xlabel("Root angle of attack (deg)")
    axes[1].set_ylabel("Std reduction (%)")
    plt.tight_layout()
    _save(fig, out_dir, "Fig_AL_Std_Reduction_vs_v_alpha.pdf")

def fig_pareto_design_ops_vs_w(df_opt, out_dir, fname="Fig_Pareto_DesignOps_vs_w.pdf",
                               cols_geom=("AR","lambda","twist"),
                               cols_ops=("alpha_hover","v_hover","alpha_cruise","v_cruise")):
    """
    Plot only geometry and operating parameters vs w_hover as a vertical stack.
    - df_opt: DataFrame from 04 optimization with at least 'w_hover' and some of cols_geom/cols_ops.
    - out_dir: directory to save the PDF.
    - fname: output filename.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if df_opt is None or len(df_opt) == 0 or "w_hover" not in df_opt.columns:
        return

    # Collapse to one row per w_hover if multiple (e.g., multiple seeds); use median for robustness
    num_cols = [c for c in df_opt.columns if pd.api.types.is_numeric_dtype(df_opt[c])]
    g = (df_opt[num_cols + ["w_hover"]]
         .groupby("w_hover", as_index=False, sort=True)
         .median(numeric_only=True))
    w = g["w_hover"].to_numpy()

    # Determine which columns are available
    geom_avail = [c for c in cols_geom if c in g.columns]
    ops_avail  = [c for c in cols_ops  if c in g.columns]
    n_rows = len(geom_avail) + len(ops_avail)
    if n_rows == 0:
        return

    # Figure size: IEEE column width ~3.4 in; ~0.9 in per row keeps it readable
    height_per_row = 0.9
    fig, axes = plt.subplots(n_rows, 1, figsize=(3.4, height_per_row * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    axes = np.asarray(axes)

    # Grayscale palette for visual separation
    greys = ["0.1", "0.35", "0.6", "0.75", "0.5", "0.3", "0.7", "0.2"]

    # Helper to format lambda nicely
    def _maybe_format_axis(ax, col):
        if col == "lambda":
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        elif col in ("alpha_hover","alpha_cruise","twist"):
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    # Labels (plain text, no symbols-only)
    ylabels = {
        "AR": "Aspect ratio",
        "lambda": "Taper ratio",
        "twist": "Twist (deg)",
        "alpha_hover": "Hover angle of attack (deg)",
        "v_hover": "Hover tip speed (m/s)",
        "alpha_cruise": "Cruise angle of attack (deg)",
        "v_cruise": "Cruise speed (m/s)",
    }

    idx = 0
    # Geometry rows
    for j, col in enumerate(geom_avail):
        ax = axes[idx]; idx += 1
        ax.plot(w, g[col].to_numpy(), color=greys[j % len(greys)], linewidth=1.2)
        ax.set_ylabel(ylabels.get(col, col))
        _maybe_format_axis(ax, col)
        ax.grid(True, color="0.9", linewidth=0.6, alpha=0.8)

    # Operations rows
    for j, col in enumerate(ops_avail):
        ax = axes[idx]; idx += 1
        ax.plot(w, g[col].to_numpy(), color=greys[(j+3) % len(greys)], linewidth=1.2)
        ax.set_ylabel(ylabels.get(col, col))
        _maybe_format_axis(ax, col)
        ax.grid(True, color="0.9", linewidth=0.6, alpha=0.8)

    # Shared x-axis
    axes[-1].set_xlabel("Hover weight")
    axes[-1].set_xlim(0.0, 1.0)
    for ax in axes:
        ax.tick_params(axis="both", which="both", length=3)

    plt.tight_layout(h_pad=0.15)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------
def main():
    cfg = path_utils.load_cfg()
    P = cfg.get("paths", {})
    master_pq = P.get("master_parquet")
    tables_dir = P.get("outputs_tables") or P.get("dir_processed")
    plots_dir  = P.get("outputs_plots")  or P.get("dir_processed")

    df_master = _try_read(master_pq)
    df_diag   = _try_read(Path(tables_dir) / "01_hover_step_diagnostics.csv")
    df_h, df_c, cal = _read_loocv(tables_dir)
    h_curve, h_meas, c_curve, c_meas = _read_case_csvs(tables_dir)
    df_opt, df_base_reop, df_base_sweep = _read_pareto_and_baselines(tables_dir)
    df_al_update, df_al_field = _read_active_learning(tables_dir)

    # Joined diagnostics for vibration vs variance
    df_diag_joined = _joined_hover_diag(df_master, tables_dir)

    # Design-space & diagnostics
    fig_design_space_2d(df_master, plots_dir)
    fig_noise_skew_kurt(df_diag, plots_dir)
    fig_estimator_mix(df_diag, plots_dir)
    fig_var_eta_vs_v(df_master, plots_dir)
    fig_negTfrac_vs_v(df_diag, plots_dir)
    fig_vibration_vs_eta_var(df_diag_joined if df_diag_joined is not None else df_diag, df_master, plots_dir)

    # LOGOCV (hover split into two figures)
    fig_loocv_hover_hist(df_h, plots_dir)
    fig_loocv_hover_absres_vs_std(df_h, plots_dir)
    fig_loocv_cruise_parity(df_c, plots_dir)
    fig_loocv_coverage_bars(df_h, df_c, plots_dir)

    # Case studies
    fig_case_hover(h_curve, h_meas, cal, plots_dir)
    fig_case_cruise(c_curve, c_meas, cal, plots_dir)

    # Pareto / UCB
    fig_pareto_ucb(df_opt, df_base_sweep, plots_dir)
    fig_pareto_contribs(df_opt, plots_dir)
    fig_pareto_geometry_vs_w(df_opt, plots_dir)
    fig_pareto_all_stacked(df_opt, df_base_sweep, plots_dir)

    # Active Learning update
    fig_al_u_global(df_al_update, plots_dir)
    fig_al_std_before_after(df_al_field, plots_dir)
    fig_al_std_reduction_hist(df_al_field, plots_dir)
    fig_al_std_reduction_vs_v_alpha(df_al_field, plots_dir)

    print(f"Figures written to: {plots_dir}")

if __name__ == "__main__":
    main()
