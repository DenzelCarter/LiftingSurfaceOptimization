# Scripts/04_ucb_pareto_search.py
# Pareto search over hover/cruise using λ95-calibrated UCB, sweeping w_hover ∈ [0,1].
# - Reads:
#     models:   outputs/models/gpr_hover_model.joblib, gpr_cruise_model.joblib
#     data:     paths.master_parquet
#     calib:    paths.outputs_tables/03_interval_calibration.csv (optional)
# - Config:
#     input_cols = ["AR","lambda","twist","alpha","v"]
#     optimization.de_params (strategy, popsize, etc.)
#     optimization.kappa_schedule (kappa_max/min, uncertainty_initial/target, lambda95_alpha)
#     optimization.bounds (OPTIONAL explicit bounds; see YAML snippet below)
# - Writes:
#     paths.outputs_tables/04_optimization_results.csv (rich columns for plotting)

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube
import warnings
import path_utils

# --------------- helpers ---------------

def schedule_kappa(U_raw_percent: float, sched: dict) -> float:
    """
    Quadratic schedule for κ based on *raw* mean-normalized uncertainty (percent).
    """
    kmax = float(sched.get("kappa_max", 2.5))
    kmin = float(sched.get("kappa_min", 0.1))
    u0   = float(sched.get("uncertainty_initial", 15.0))
    ut   = float(sched.get("uncertainty_target",  5.0))

    if not np.isfinite(U_raw_percent):
        return kmin
    if u0 <= ut:
        return kmin

    z = (U_raw_percent - ut) / (u0 - ut)
    z = float(np.clip(z, 0.0, 1.0))
    return kmin + (kmax - kmin) * (z ** 2)


def _calc_global_uncertainty(model, df_mode: pd.DataFrame, median_perf: float, features: list[str], n_samples=10000) -> float:
    """
    RAW Mean-Normalized Uncertainty (%): 100 * mean(std(x)) / median_perf,
    sampling within the empirical min/max box of df_mode[features].
    """
    if not np.isfinite(median_perf) or median_perf == 0:
        return np.nan
    if df_mode.empty:
        return np.nan

    bounds = []
    for f in features:
        lo = float(df_mode[f].min())
        hi = float(df_mode[f].max())
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return np.nan
        bounds.append((lo, hi))
    bounds = np.array(bounds)

    sampler = LatinHypercube(d=len(features), seed=42)
    U = sampler.random(n=n_samples)
    X = bounds[:, 0] + U * (bounds[:, 1] - bounds[:, 0])
    X = pd.DataFrame(X, columns=features)

    _, s = model.predict(X, return_std=True)
    return 100.0 * float(np.mean(s)) / float(median_perf)


def _load_interval_calibration(tables_dir: Path) -> tuple[float, float, float, float]:
    lam68_h = lam95_h = lam68_c = lam95_c = 1.0
    cal = tables_dir / "03_interval_calibration.csv"
    if not cal.exists():
        return lam68_h, lam95_h, lam68_c, lam95_c
    try:
        df = pd.read_csv(cal)
        if not {"mode","lambda_68","lambda_95"}.issubset(df.columns):
            return lam68_h, lam95_h, lam68_c, lam95_c
        df["mode"] = df["mode"].astype(str).str.strip().str.lower()
        def get(mode, col, dflt):
            try:
                return float(df.loc[df["mode"]==mode, col].iloc[0])
            except Exception:
                return dflt
        return get("hover","lambda_68",1.0), get("hover","lambda_95",1.0), \
               get("cruise","lambda_68",1.0), get("cruise","lambda_95",1.0)
    except Exception:
        return lam68_h, lam95_h, lam68_c, lam95_c


def _bounds_from_config_or_data(cfg: dict, df_hover: pd.DataFrame, df_cruise: pd.DataFrame) -> list[tuple[float,float]]:
    """
    Build 7D bounds in the order:
    [AR, lambda, twist, alpha_h, v_h, alpha_c, v_c]
    Prefer config.optimization.bounds if present; otherwise fallback to empirical min/max.
    """
    # defaults from data (empirical)
    def emp(df, col):
        lo = float(df[col].min()); hi = float(df[col].max())
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            # fallback to both modes union
            lo = float(min(df_hover[col].min(), df_cruise[col].min()))
            hi = float(max(df_hover[col].max(), df_cruise[col].max()))
        return lo, hi

    geo_cols = ["AR","lambda","twist"]
    b_AR, b_la, b_tw = (emp(pd.concat([df_hover, df_cruise]), c) for c in geo_cols)
    b_ah = emp(df_hover,  "alpha")
    b_vh = emp(df_hover,  "v")
    b_ac = emp(df_cruise, "alpha")
    b_vc = emp(df_cruise, "v")

    bounds = {
        "AR": b_AR, "lambda": b_la, "twist": b_tw,
        "hover": {"alpha": b_ah, "v": b_vh},
        "cruise":{"alpha": b_ac, "v": b_vc},
    }

    # override with config if provided
    opt = cfg.get("optimization", {})
    cfgb = opt.get("bounds", None)
    if isinstance(cfgb, dict):
        def pick(d, k, default):
            if k in d and isinstance(d[k], (list, tuple)) and len(d[k])==2:
                return float(d[k][0]), float(d[k][1])
            return default
        b_AR = pick(cfgb, "AR", bounds["AR"])
        b_la = pick(cfgb, "lambda", bounds["lambda"])
        b_tw = pick(cfgb, "twist", bounds["twist"])
        hov  = cfgb.get("hover", {})
        cru  = cfgb.get("cruise", {})
        b_ah = pick(hov, "alpha", bounds["hover"]["alpha"])
        b_vh = pick(hov, "v",     bounds["hover"]["v"])
        b_ac = pick(cru, "alpha", bounds["cruise"]["alpha"])
        b_vc = pick(cru, "v",     bounds["cruise"]["v"])

    return [b_AR, b_la, b_tw, b_ah, b_vh, b_ac, b_vc]


def _ucb_objective_7d(x7,
                      w_hover: float,
                      gpr_hover, gpr_cruise,
                      med_hover: float, med_cruise: float,
                      kappa_h: float, kappa_c: float,
                      lam95_h: float, lam95_c: float) -> float:
    """
    x7 = [AR, lambda, twist, alpha_h, v_h, alpha_c, v_c]
    UCB_total = w*(μ_h/med_h + κ_h*λ95_h*σ_h/med_h) + (1-w)*(μ_c/med_c + κ_c*λ95_c*σ_c/med_c)
    Return NEGATIVE for minimization.
    """
    AR, lam, tw, ah, vh, ac, vc = x7
    # feature order for both models is input_cols: ["AR","lambda","twist","alpha","v"]
    Xh = pd.DataFrame([[AR, lam, tw, ah, vh]], columns=["AR","lambda","twist","alpha","v"])
    Xc = pd.DataFrame([[AR, lam, tw, ac, vc]], columns=["AR","lambda","twist","alpha","v"])

    mu_h, s_h = gpr_hover.predict(Xh, return_std=True)
    mu_c, s_c = gpr_cruise.predict(Xc, return_std=True)

    mu_h = float(mu_h[0]); s_h = float(s_h[0])
    mu_c = float(mu_c[0]); s_c = float(s_c[0])

    if not (np.isfinite(med_hover) and med_hover != 0.0 and np.isfinite(med_cruise) and med_cruise != 0.0):
        return 1e9

    exploit_h = mu_h / med_hover
    exploit_c = mu_c / med_cruise
    explore_h = (kappa_h * lam95_h * s_h) / med_hover
    explore_c = (kappa_c * lam95_c * s_c) / med_cruise

    ucb = w_hover * (exploit_h + explore_h) + (1.0 - w_hover) * (exploit_c + explore_c)

    if not np.isfinite(ucb):
        return 1e9
    return -float(ucb)


# --------------- main ---------------

def main():
    cfg = path_utils.load_cfg()
    P   = cfg["paths"]
    OPT = cfg.get("optimization", {})
    KAP = OPT.get("kappa_schedule", {})

    # Input features (both models trained on the same order)
    features = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])
    assert features == ["AR","lambda","twist","alpha","v"], \
        "This script assumes input_cols order = ['AR','lambda','twist','alpha','v']."

    # Load data & models
    master = Path(P["master_parquet"])
    if not master.exists():
        raise FileNotFoundError(f"Master parquet not found: {master}")
    df = pd.read_parquet(master)

    models_dir = Path(P["outputs_models"])
    gpr_hover  = load(models_dir / "gpr_hover_model.joblib")
    gpr_cruise = load(models_dir / "gpr_cruise_model.joblib")

    tables_dir = Path(P.get("outputs_tables") or P.get("dir_processed"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    lam68_h, lam95_h, lam68_c, lam95_c = _load_interval_calibration(tables_dir)

    # Clean slices
    need_cols = set(features + ["flight_mode","performance"])
    df = df[[c for c in df.columns if c in need_cols]].copy()
    df_hover  = df[df["flight_mode"]=="hover"].dropna(subset=features+["performance"]).copy()
    df_cruise = df[df["flight_mode"]=="cruise"].dropna(subset=features+["performance"]).copy()
    if df_hover.empty or df_cruise.empty:
        raise ValueError("Empty hover or cruise slice after cleaning. Check 01/02/03 outputs.")

    med_hover  = float(df_hover["performance"].median())
    med_cruise = float(df_cruise["performance"].median())

    # Global uncertainties (raw %)
    U_h_raw = _calc_global_uncertainty(gpr_hover,  df_hover,  med_hover,  features)
    U_c_raw = _calc_global_uncertainty(gpr_cruise, df_cruise, med_cruise, features)

    # κ scheduling + λ95 neutralization (tunable α; 0=no neutralization; 1=full)
    alpha_neut = float(KAP.get("lambda95_alpha", 0.5))
    kappa_h_raw = schedule_kappa(U_h_raw, KAP)
    kappa_c_raw = schedule_kappa(U_c_raw, KAP)
    kappa_h = kappa_h_raw / (lam95_h ** alpha_neut if lam95_h > 0 else 1.0)
    kappa_c = kappa_c_raw / (lam95_c ** alpha_neut if lam95_c > 0 else 1.0)

    # Bounds: [AR, lambda, twist, alpha_h, v_h, alpha_c, v_c]
    bounds_7d = _bounds_from_config_or_data(cfg, df_hover, df_cruise)

    # DE params (be conservative with workers on Windows to avoid pickling issues)
    defaults = dict(strategy='best1bin', maxiter=OPT.get("max_iter", 150), popsize=OPT.get("pop_size", 20),
                    mutation=(0.5, 1.0), recombination=0.7, tol=0.01,
                    workers=1, updating='deferred', seed=42)
    de_params = {**defaults, **OPT.get("de_params", {})}

    # Sweep w_hover
    nW = int(OPT.get("num_weights", 21))
    w_grid = np.linspace(0.0, 1.0, nW)

    rows = []
    for w_h in w_grid:
        res = differential_evolution(
            func=_ucb_objective_7d,
            bounds=bounds_7d,
            args=(w_h, gpr_hover, gpr_cruise, med_hover, med_cruise, kappa_h, kappa_c, lam95_h, lam95_c),
            **de_params
        )

        AR, lam, tw, ah, vh, ac, vc = res.x
        Xh = pd.DataFrame([[AR, lam, tw, ah, vh]], columns=features)
        Xc = pd.DataFrame([[AR, lam, tw, ac, vc]], columns=features)

        mu_h, s_h = gpr_hover.predict(Xh, return_std=True)
        mu_c, s_c = gpr_cruise.predict(Xc, return_std=True)
        mu_h = float(mu_h[0]); s_h = float(s_h[0])
        mu_c = float(mu_c[0]); s_c = float(s_c[0])

        # Contributions
        exploit_h = mu_h / med_hover
        exploit_c = mu_c / med_cruise
        explore_h = (kappa_h * lam95_h * s_h) / med_hover
        explore_c = (kappa_c * lam95_c * s_c) / med_cruise

        ucb_h = exploit_h + explore_h
        ucb_c = exploit_c + explore_c
        ucb_total = w_h * ucb_h + (1.0 - w_h) * ucb_c

        row = {
            # weight & final objective
            "w_hover": w_h,
            "ucb_total": ucb_total,
            # per-mode UCB and parts (normalized)
            "ucb_hover": ucb_h, "ucb_cruise": ucb_c,
            "exploit_hover": exploit_h, "explore_hover": explore_h,
            "exploit_cruise": exploit_c, "explore_cruise": explore_c,
            # raw predictions / stds
            "hover_eta": mu_h, "hover_std_raw": s_h, "hover_std_lam95": lam95_h * s_h,
            "cruise_ld": mu_c, "cruise_std_raw": s_c, "cruise_std_lam95": lam95_c * s_c,
            # calibrated 95% bands (useful for plots)
            "hover_lower95":  mu_h - 1.96 * lam95_h * s_h,
            "hover_upper95":  mu_h + 1.96 * lam95_h * s_h,
            "cruise_lower95": mu_c - 1.96 * lam95_c * s_c,
            "cruise_upper95": mu_c + 1.96 * lam95_c * s_c,
            # medians used for normalization (for reproducibility)
            "median_hover": med_hover, "median_cruise": med_cruise,
            # κ and λ actually used
            "kappa_hover_eff": kappa_h, "kappa_cruise_eff": kappa_c,
            "lambda95_hover": lam95_h, "lambda95_cruise": lam95_c,
            # global raw uncertainty (%), handy for reporting
            "U_hover_raw_percent": U_h_raw, "U_cruise_raw_percent": U_c_raw,
            # chosen parameters (geo shared; ops per mode)
            "AR": AR, "lambda": lam, "twist": tw,
            "alpha_hover": ah, "v_hover": vh,
            "alpha_cruise": ac, "v_cruise": vc,
            # DE housekeeping
            "de_fun": float(res.fun), "de_nit": int(getattr(res, "nit", -1)),
            "de_nfev": int(getattr(res, "nfev", -1)), "de_success": bool(res.success),
            "de_message": str(res.message),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = tables_dir / "04_optimization_results.csv"
    out.to_csv(out_path, index=False, float_format="%.6g")
    print(f"Wrote Pareto sweep results → {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()
