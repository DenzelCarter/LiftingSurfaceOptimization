# Scripts/05_reschedule_baselines.py
# Goal: Provide an apples-to-apples baseline for comparison with the Pareto-optimized designs.
#
# What this script does (two complementary tracks):
#   1) "reoperated"  — For every unique geometry (AR, lambda, twist), we use the trained
#      hover/cruise surrogates to *re-operate* that geometry: maximize the predicted mean
#      over each mode’s operating variables within bounded ranges. This gives each baseline
#      geometry its best achievable performance under the *same* search rules as optimization.
#
#   2) "as_tested"   — For geometries that were physically tested in a mode, we record the
#      *best measured* step (max performance) and its operating point. For the *other* mode
#      (if not tested), we fill in using the surrogate at that geometry, maximizing over
#      the same operating bounds as in (1). This preserves each design’s original scheduling
#      where it exists, and uses the surrogate only where data are absent.
#
# Outputs (to paths.outputs_tables):
#   - 05_baseline_reoperate.csv
#       Per-geometry table with:
#         * measured maxima (if present) per mode
#         * reoperated (predicted) optima per mode + predicted std + 95% CI (λ95-calibrated)
#         * normalized scores (÷ per-mode median), geometry ids, counts, etc.
#
#   - 05_baseline_ucb_sweep.csv
#       For each geometry and each w_hover in the config grid, the UCB decomposition using
#       the reoperated (predicted) optima per mode, with κ-schedule and λ95 calibration
#       consistent with 04_ucb_pareto_search.py.
#
# Notes:
#   * Feature order assumed in ALL models: ["AR","lambda","twist","alpha","v"] (from config.input_cols)
#   * Operating bounds come from config.optimization.bounds if provided, otherwise empirical min/max.
#   * λ95 multipliers are loaded from 03_interval_calibration.csv if available (fallback = 1.0).
#   * All predictions are grayscale-agnostic; plotting handled elsewhere.
#
# ------------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats.qmc import LatinHypercube
import warnings
import path_utils

# ----------------------------- Utilities -----------------------------

def _round_geom(df: pd.DataFrame, cols=("AR","lambda","twist"), nd=6):
    """Round geometry columns to stabilize equality joins."""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].astype(float).round(nd)
    return d

def _ensure_features(cfg):
    feats = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])
    if feats != ["AR","lambda","twist","alpha","v"]:
        raise ValueError(f"Expected input_cols=['AR','lambda','twist','alpha','v'], got {feats}")
    return feats

def _read_calibration(tables_dir: Path):
    lam68_h = lam95_h = lam68_c = lam95_c = 1.0
    p = tables_dir / "03_interval_calibration.csv"
    if not p.exists():
        return lam68_h, lam95_h, lam68_c, lam95_c
    try:
        df = pd.read_csv(p)
        df["mode"] = df["mode"].astype(str).str.lower()
        def get(mode, col, dflt):
            try:
                return float(df.loc[df["mode"]==mode, col].iloc[0])
            except Exception:
                return dflt
        return get("hover","lambda_68",1.0), get("hover","lambda_95",1.0), \
               get("cruise","lambda_68",1.0), get("cruise","lambda_95",1.0)
    except Exception:
        return lam68_h, lam95_h, lam68_c, lam95_c

def _bounds_from_config_or_data(cfg, df_hover: pd.DataFrame, df_cruise: pd.DataFrame):
    """
    Return dict:
      bounds = {
        "hover":  {"alpha": (lo,hi), "v": (lo,hi)},
        "cruise": {"alpha": (lo,hi), "v": (lo,hi)}
      }
    Config takes precedence; otherwise use empirical min/max from each mode slice.
    """
    def emp(df, col):
        lo = float(df[col].min()); hi = float(df[col].max())
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            # pad a touch if degenerate
            pad = 1e-6
            lo, hi = lo - pad, lo + pad
        return (lo, hi)

    # Start with empirical
    hb = {"alpha": emp(df_hover, "alpha"), "v": emp(df_hover, "v")} if not df_hover.empty else None
    cb = {"alpha": emp(df_cruise,"alpha"), "v": emp(df_cruise,"v")} if not df_cruise.empty else None

    # Config overrides
    opt = cfg.get("optimization", {})
    cfgb = opt.get("bounds", None)
    if isinstance(cfgb, dict):
        hov = cfgb.get("hover", {})
        cru = cfgb.get("cruise", {})
        def pick(d, k, default):
            if isinstance(d, dict) and k in d and isinstance(d[k], (list, tuple)) and len(d[k])==2:
                return (float(d[k][0]), float(d[k][1]))
            return default
        if hb is not None:
            hb = {"alpha": pick(hov,"alpha", hb["alpha"]),
                  "v":     pick(hov,"v",     hb["v"])}
        if cb is not None:
            cb = {"alpha": pick(cru,"alpha", cb["alpha"]),
                  "v":     pick(cru,"v",     cb["v"])}

    return {"hover": hb, "cruise": cb}

def _grid_argmax(model, features, geom, bounds, n=(64,64)):
    """
    Vectorized 2D grid search over (alpha, v) for a fixed geometry.
    Returns: best (alpha, v), mu, std
    """
    (a0,b0) = bounds["alpha"]; (a1,b1) = bounds["v"]
    g0 = np.linspace(a0, b0, n[0]); g1 = np.linspace(a1, b1, n[1])
    A, V = np.meshgrid(g0, g1, indexing="xy")
    npts = A.size
    df = pd.DataFrame({
        "AR":     np.full(npts, geom["AR"]),
        "lambda": np.full(npts, geom["lambda"]),
        "twist":  np.full(npts, geom["twist"]),
        "alpha":  A.ravel(),
        "v":      V.ravel(),
    })[features]
    mu, s = model.predict(df, return_std=True)
    idx = int(np.argmax(mu))
    best = {"alpha": float(df.iloc[idx]["alpha"]), "v": float(df.iloc[idx]["v"])}
    return best, float(mu[idx]), float(s[idx])

def _calc_global_uncertainty(model, df_mode: pd.DataFrame, median_perf: float, features, n_samples=10000):
    """Raw mean-normalized uncertainty (%) by sampling within empirical min/max box."""
    if df_mode.empty or not np.isfinite(median_perf) or median_perf == 0:
        return np.nan
    lo = df_mode[features].min(numeric_only=True).astype(float).to_numpy()
    hi = df_mode[features].max(numeric_only=True).astype(float).to_numpy()
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
        return np.nan
    sampler = LatinHypercube(d=len(features), seed=17)
    U = sampler.random(n=n_samples)
    X = lo + U*(hi-lo)
    dfX = pd.DataFrame(X, columns=features)
    _, s = model.predict(dfX, return_std=True)
    return 100.0*float(np.mean(s))/float(median_perf)

def _schedule_kappa(U_raw_percent: float, sched: dict) -> float:
    kmax = float(sched.get("kappa_max", 2.5))
    kmin = float(sched.get("kappa_min", 0.1))
    u0   = float(sched.get("uncertainty_initial", 16.0))
    ut   = float(sched.get("uncertainty_target",   5.0))
    if not np.isfinite(U_raw_percent) or u0 <= ut:
        return kmin
    z = (U_raw_percent - ut)/(u0 - ut)
    z = float(np.clip(z, 0.0, 1.0))
    return kmin + (kmax - kmin)*(z**2)

def _attach_ci(mu: float, s: float, lam95: float) -> tuple[float,float]:
    lo = mu - 1.96*lam95*s
    hi = mu + 1.96*lam95*s
    return float(lo), float(hi)

# ----------------------------- Main -----------------------------

def main():
    print("--- Baseline (re)operation for fair comparison ---")
    cfg = path_utils.load_cfg()
    P   = cfg["paths"]
    features = _ensure_features(cfg)

    tables_dir = Path(P.get("outputs_tables") or P.get("dir_processed")); tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(P["outputs_models"])
    master_pq  = Path(P["master_parquet"])

    if not master_pq.exists():
        raise FileNotFoundError(f"Master parquet not found: {master_pq}")

    # Load data & models
    df = pd.read_parquet(master_pq)
    df = df[[c for c in df.columns if c in (features + ["flight_mode","performance","performance_variance"])]]  # keep needed
    df = _round_geom(df, cols=["AR","lambda","twist"], nd=6)

    gpr_h = load(models_dir / "gpr_hover_model.joblib")
    gpr_c = load(models_dir / "gpr_cruise_model.joblib")

    # Slices and medians
    dH = df[df["flight_mode"]=="hover"].dropna(subset=["performance"] + features).copy()
    dC = df[df["flight_mode"]=="cruise"].dropna(subset=["performance"] + features).copy()

    if dH.empty and dC.empty:
        raise ValueError("No clean data in master parquet; run 01→03 first.")

    medH = float(dH["performance"].median()) if not dH.empty else np.nan
    medC = float(dC["performance"].median()) if not dC.empty else np.nan

    # λ95 calibration
    lam68_h, lam95_h, lam68_c, lam95_c = _read_calibration(tables_dir)

    # κ schedule
    KAP = cfg.get("optimization", {}).get("kappa_schedule", {})
    Uh_raw = _calc_global_uncertainty(gpr_h, dH, medH, features) if not dH.empty else np.nan
    Uc_raw = _calc_global_uncertainty(gpr_c, dC, medC, features) if not dC.empty else np.nan
    alpha_neut = float(KAP.get("lambda95_alpha", 0.5))
    kappa_h_raw = _schedule_kappa(Uh_raw, KAP)
    kappa_c_raw = _schedule_kappa(Uc_raw, KAP)
    kappa_h = kappa_h_raw / (lam95_h**alpha_neut if lam95_h>0 else 1.0)
    kappa_c = kappa_c_raw / (lam95_c**alpha_neut if lam95_c>0 else 1.0)

    # Operating bounds
    B = _bounds_from_config_or_data(cfg, dH, dC)  # dict per mode

    # Unique geometries (union over both modes)
    geos = pd.concat([dH, dC], axis=0, ignore_index=True)[["AR","lambda","twist"]].drop_duplicates().reset_index(drop=True)
    geos["geom_id"] = [f"G{str(i+1).zfill(3)}" for i in range(len(geos))]

    # Convenience lookup for measured per-geometry maxima
    def best_measured(df_mode, geom):
        if df_mode.empty: return None
        m = (df_mode["AR"]==geom["AR"]) & (df_mode["lambda"]==geom["lambda"]) & (df_mode["twist"]==geom["twist"])
        sub = df_mode.loc[m]
        if sub.empty: return None
        row = sub.loc[sub["performance"].idxmax()]
        out = {
            "alpha": float(row["alpha"]),
            "v":     float(row["v"]),
            "mu":    float(row["performance"]),
            "var_meas": float(row["performance_variance"]) if "performance_variance" in row and pd.notna(row["performance_variance"]) else np.nan
        }
        return out

    # Compute per-geometry baseline table
    rows = []
    for _, g in geos.iterrows():
        geom = {"AR": float(g["AR"]), "lambda": float(g["lambda"]), "twist": float(g["twist"])}
        row = {
            "geom_id": g["geom_id"], "AR": geom["AR"], "lambda": geom["lambda"], "twist": geom["twist"]
        }

        # Counts
        nH = int(((dH["AR"]==geom["AR"]) & (dH["lambda"]==geom["lambda"]) & (dH["twist"]==geom["twist"])).sum()) if not dH.empty else 0
        nC = int(((dC["AR"]==geom["AR"]) & (dC["lambda"]==geom["lambda"]) & (dC["twist"]==geom["twist"])).sum()) if not dC.empty else 0
        row["n_hover_steps"]  = nH
        row["n_cruise_steps"] = nC

        # As-tested (measured) maxima if present
        mH = best_measured(dH, geom)
        mC = best_measured(dC, geom)
        if mH:
            row.update({
                "meas_hover_alpha": mH["alpha"], "meas_hover_v": mH["v"],
                "meas_hover_mu": mH["mu"], "meas_hover_var": mH["var_meas"]
            })
        else:
            row.update({"meas_hover_alpha": np.nan, "meas_hover_v": np.nan, "meas_hover_mu": np.nan, "meas_hover_var": np.nan})
        if mC:
            row.update({
                "meas_cruise_alpha": mC["alpha"], "meas_cruise_v": mC["v"],
                "meas_cruise_mu": mC["mu"]
            })
        else:
            row.update({"meas_cruise_alpha": np.nan, "meas_cruise_v": np.nan, "meas_cruise_mu": np.nan})

        # Reoperated (predicted) optima per mode
        if B["hover"] is not None:
            bestH, muH, sH = _grid_argmax(gpr_h, features, geom, B["hover"], n=(64,64))
            loH, hiH = _attach_ci(muH, sH, lam95_h)
            row.update({
                "pred_hover_alpha": bestH["alpha"], "pred_hover_v": bestH["v"],
                "pred_hover_mu": muH, "pred_hover_std": sH,
                "pred_hover_lower95": loH, "pred_hover_upper95": hiH
            })
        else:
            row.update(dict.fromkeys(["pred_hover_alpha","pred_hover_v","pred_hover_mu","pred_hover_std","pred_hover_lower95","pred_hover_upper95"], np.nan))

        if B["cruise"] is not None:
            bestC, muC, sC = _grid_argmax(gpr_c, features, geom, B["cruise"], n=(64,64))
            loC, hiC = _attach_ci(muC, sC, lam95_c)
            row.update({
                "pred_cruise_alpha": bestC["alpha"], "pred_cruise_v": bestC["v"],
                "pred_cruise_mu": muC, "pred_cruise_std": sC,
                "pred_cruise_lower95": loC, "pred_cruise_upper95": hiC
            })
        else:
            row.update(dict.fromkeys(["pred_cruise_alpha","pred_cruise_v","pred_cruise_mu","pred_cruise_std","pred_cruise_lower95","pred_cruise_upper95"], np.nan))

        # Normalized scores (÷ per-mode medians)
        row["pred_hover_mu_norm"]  = (row["pred_hover_mu"]/medH) if np.isfinite(medH) and np.isfinite(row.get("pred_hover_mu", np.nan)) else np.nan
        row["pred_cruise_mu_norm"] = (row["pred_cruise_mu"]/medC) if np.isfinite(medC) and np.isfinite(row.get("pred_cruise_mu", np.nan)) else np.nan
        row["meas_hover_mu_norm"]  = (row["meas_hover_mu"]/medH) if np.isfinite(medH) and np.isfinite(row.get("meas_hover_mu", np.nan)) else np.nan
        row["meas_cruise_mu_norm"] = (row["meas_cruise_mu"]/medC) if np.isfinite(medC) and np.isfinite(row.get("meas_cruise_mu", np.nan)) else np.nan

        rows.append(row)

    df_out = pd.DataFrame(rows)
    out1 = tables_dir / "05_baseline_reoperate.csv"
    df_out.to_csv(out1, index=False, float_format="%.6g")
    print(f"Wrote: {out1}  (rows={len(df_out)})")

    # --------------------- UCB sweep for baselines (reoperated) ---------------------
    OPT = cfg.get("optimization", {})
    nW = int(OPT.get("num_weights", 21))
    w_grid = np.linspace(0.0, 1.0, nW)

    sweep_rows = []
    for _, r in df_out.iterrows():
        mu_h = float(r.get("pred_hover_mu", np.nan));   s_h = float(r.get("pred_hover_std", np.nan))
        mu_c = float(r.get("pred_cruise_mu", np.nan));  s_c = float(r.get("pred_cruise_std", np.nan))
        if not (np.isfinite(mu_h) and np.isfinite(s_h) and np.isfinite(mu_c) and np.isfinite(s_c)):
            continue
        # normalized exploit/explore terms
        if not (np.isfinite(medH) and medH != 0.0 and np.isfinite(medC) and medC != 0.0):
            continue
        exploit_h = mu_h/medH
        exploit_c = mu_c/medC
        explore_h = (kappa_h * lam95_h * s_h)/medH
        explore_c = (kappa_c * lam95_c * s_c)/medC
        for w in w_grid:
            ucb_h = exploit_h + explore_h
            ucb_c = exploit_c + explore_c
            ucb_total = float(w*ucb_h + (1.0-w)*ucb_c)
            sweep_rows.append({
                "geom_id": r["geom_id"], "AR": r["AR"], "lambda": r["lambda"], "twist": r["twist"],
                "w_hover": float(w),
                "ucb_total": ucb_total,
                "ucb_hover": float(ucb_h), "ucb_cruise": float(ucb_c),
                "exploit_hover": float(exploit_h), "explore_hover": float(explore_h),
                "exploit_cruise": float(exploit_c), "explore_cruise": float(explore_c),
                "hover_mu": mu_h, "hover_std_raw": s_h, "hover_lower95": r["pred_hover_lower95"], "hover_upper95": r["pred_hover_upper95"],
                "cruise_mu": mu_c, "cruise_std_raw": s_c, "cruise_lower95": r["pred_cruise_lower95"], "cruise_upper95": r["pred_cruise_upper95"],
                "hover_alpha": r["pred_hover_alpha"], "hover_v": r["pred_hover_v"],
                "cruise_alpha": r["pred_cruise_alpha"], "cruise_v": r["pred_cruise_v"],
                "median_hover": medH, "median_cruise": medC,
                "kappa_hover_eff": kappa_h, "kappa_cruise_eff": kappa_c,
                "lambda95_hover": lam95_h, "lambda95_cruise": lam95_c,
            })

    df_sweep = pd.DataFrame(sweep_rows)
    out2 = tables_dir / "05_baseline_ucb_sweep.csv"
    df_sweep.to_csv(out2, index=False, float_format="%.6g")
    print(f"Wrote: {out2}  (rows={len(df_sweep)})")

    print("\nNotes:")
    print(" - '05_baseline_reoperate.csv' gives both measured maxima (if any) and surrogate-reoperated optima per geometry.")
    print(" - '05_baseline_ucb_sweep.csv' provides UCB decompositions per w_hover using the reoperated optima,")
    print("    directly comparable to 04_optimization_results.csv.")

if __name__ == "__main__":
    main()
