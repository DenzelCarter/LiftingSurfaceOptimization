# Scripts/06_active_learning_update.py
# Quantify how much adding a newly tested geometry (from doe_optima.csv) reduces
# global hover model uncertainty. This script:
#   1) Loads the existing master parquet and trained HOVER GPR.
#   2) Computes a "global mean-normalized uncertainty" metric U_before (%).
#   3) Processes NEW hover raw CSVs for k_id present in doe_optima.csv but not in doe_initial.csv,
#      using the same robust per-step estimator + pair bootstrap as 01_process_data.py.
#   4) Retrains the hover GPR on the augmented dataset and recomputes U_after on the SAME bounds.
#   5) Writes a summary CSV and an optional std-field CSV for deeper analysis.
#
# Outputs (to paths.outputs_tables):
#   - 06_active_learning_update.csv
#       {U_before_percent, U_after_percent, reduction_percent, n_new_k, n_new_steps, k_new_list}
#   - 06_hover_std_field.csv
#       Random design points within a fixed bounds box with std_before/std_after and absolute/relative reductions.
#
# Notes:
#   * Uses input_cols=['AR','lambda','twist','alpha','v'].
#   * Hover GPR is heteroscedastic (alpha=performance_variance).
#   * Bounds come from config.optimization.bounds if present; otherwise computed from the ORIGINAL hover set
#     so U_before and U_after are compared on the SAME region (no moving target).
#   * If paths.doe_optima is not in config, we look next to paths.doe_csv for 'doe_optima.csv'.
#   * Tare/no-load correction is applied if present (same approach as 01).
#
# Run after: 01 → 02 → 03 → (collect new LS_*.csv for held-out k) → 06

from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import trim_mean, skew, kurtosis
from scipy.stats.qmc import LatinHypercube

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import path_utils


# ----------------------------- Config & helpers -----------------------------

RAW_NAME_HINTS = {
    "esc": ["ESC signal (µs)", "ESC signal (us)", "esc", "esc_signal", "esc signal"],
    "rpm": ["Motor Electrical Speed (RPM)", "RPM", "rpm"],
    "T":   ["Thrust (N)", "thrust", "T (N)", "T"],
    "Q":   ["Torque (N·m)", "Torque (Nm)", "torque", "Q (N*m)", "Q (N·m)", "Q"],
    "k_id":["k_id", "filename", "file", "design_id"],
    "vib":["Vibration (g)","vibration (g)","vibration","imu (g)","imu_g","accel_g","accelerometer (g)"]
}

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def find_col(df: pd.DataFrame, candidates) -> str | None:
    candidates = _as_list(candidates)
    canon = {re.sub(r'\s+', ' ', c.strip().lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r'\s+', ' ', str(name).strip().lower())
        if key in canon:
            return canon[key]
    for key, orig in canon.items():
        for name in candidates:
            cand = re.sub(r'\s+', ' ', str(name).strip().lower())
            if cand and cand in key:
                return orig
    return None

def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n >= 2: return float(np.std(x, ddof=1))
    if n == 1: return 0.0
    return np.nan

def load_config() -> Dict:
    cfg = path_utils.load_cfg()
    P = cfg.get("paths", {})
    F = cfg.get("fluids", {})
    G = cfg.get("geometry", {})

    paths = {
        "hover_raw":      P.get("dir_hover_raw"),
        "hover_tare":     P.get("dir_hover_tare"),
        "processed_dir":  P.get("dir_processed"),
        "master_parquet": P.get("master_parquet"),
        "doe_csv":        P.get("doe_csv"),
        "doe_optima":     P.get("doe_optima"),   # optional
        "outputs_models": P.get("outputs_models"),
        "outputs_tables": P.get("outputs_tables") or P.get("dir_processed"),
    }

    rho_default = 1.204
    nu_default  = 1.51e-5
    rho = float(F.get("rho", rho_default)) if F is not None else rho_default
    nu  = float(F.get("nu", nu_default)) if F is not None else nu_default

    r_hub   = float(G.get("r_hub_m", 0.046))
    l_blade = float(G.get("l_blade_m", 0.184))
    r_tip   = float(G.get("r_tip_m", r_hub + l_blade))
    disk_A  = float(np.pi * r_tip**2)

    features = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])

    return {
        "cfg": cfg, "paths": paths,
        "rho": rho, "nu": nu, "r_tip": r_tip, "disk_A": disk_A,
        "features": features
    }

# ----------------------------- Tare, estimators, eta -----------------------------

def load_tare_interps(dir_hover_tare: str):
    if not dir_hover_tare:
        return None
    candidates = ["Tare_01.csv", "No_Load_Rotate_01.csv", "tare.csv"]
    tare_file = None
    for name in candidates:
        p = Path(dir_hover_tare) / name
        if p.exists(): tare_file = p; break
    if tare_file is None:
        warnings.warn("No tare file found; proceeding without tare correction.")
        return None

    df = pd.read_csv(tare_file, low_memory=False)
    rpm_col = find_col(df, RAW_NAME_HINTS["rpm"])
    T_col   = find_col(df, RAW_NAME_HINTS["T"])
    Q_col   = find_col(df, RAW_NAME_HINTS["Q"])
    if not all([rpm_col, T_col, Q_col]):
        warnings.warn(f"{tare_file.name} missing required columns; skipping tare correction.")
        return None

    grp = df.groupby(df[rpm_col].round(0)).agg({
        rpm_col:"mean", T_col:["mean","std"], Q_col:["mean","std"]
    })
    grp.columns = ["rpm_mean","T_mean","T_std","Q_mean","Q_std"]
    grp = grp.sort_values("rpm_mean").reset_index(drop=True)
    x = grp["rpm_mean"].to_numpy()
    def interp(col):
        yv = grp[col].to_numpy()
        return lambda r: np.interp(r, x, yv, left=yv[0], right=yv[-1])
    return {
        "T_mean": interp("T_mean"),
        "Q_mean": interp("Q_mean"),
    }

def estimator_select(x: np.ndarray):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "mean", np.nan, np.nan
    g1 = skew(x, bias=False)
    g2 = kurtosis(x, fisher=True, bias=False)
    if not np.isfinite(g1) or not np.isfinite(g2):
        return "trim10", float(g1), float(g2)
    if (abs(g1) <= 0.5) and (-0.5 <= g2 <= 1.0):
        m = "mean"
    elif (abs(g1) <= 1.0) and (-1.0 <= g2 <= 2.0):
        m = "trim10"
    else:
        m = "trim20"
    return m, float(g1), float(g2)

def estimator_apply(x: np.ndarray, method: str) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    if method == "mean":   return float(np.mean(x))
    if method == "trim10": return float(trim_mean(x, 0.10))
    if method == "trim20": return float(trim_mean(x, 0.20))
    if method == "median": return float(np.median(x))
    raise ValueError(f"unknown method {method}")

def bootstrap_loc_var(x: np.ndarray, method: str, B: int = 500, rng=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3: return np.nan, np.nan
    rng = rng or np.random.default_rng()
    est = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        est[b] = estimator_apply(x[idx], method)
    var = float(np.var(est, ddof=1))
    se  = float(np.sqrt(var))
    return var, se

def eta_ratio_of_means(T: np.ndarray, Q: np.ndarray, rpm: np.ndarray,
                       rho: float, disk_A: float,
                       mN: str, mD: str) -> float:
    w = rpm * (2.0*np.pi/60.0)  # rad/s
    N = np.sqrt(np.clip(T, 0.0, None)**3 / (2.0 * rho * disk_A))
    D = Q * w
    Nbar = estimator_apply(N, mN)
    Dbar = estimator_apply(D, mD)
    if not np.isfinite(Nbar) or not np.isfinite(Dbar) or Dbar <= 0:
        return np.nan
    return float(Nbar / Dbar)

def bootstrap_eta_pairs(T: np.ndarray, Q: np.ndarray, rpm: np.ndarray,
                        rho: float, disk_A: float,
                        mN: str, mD: str,
                        B: int = 500, rng=None):
    T = np.asarray(T, float); Q = np.asarray(Q, float); rpm = np.asarray(rpm, float)
    mask = np.isfinite(T) & np.isfinite(Q) & np.isfinite(rpm)
    T, Q, rpm = T[mask], Q[mask], rpm[mask]
    n = T.size
    if n < 5: return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng()
    point = eta_ratio_of_means(T, Q, rpm, rho, disk_A, mN, mD)
    if not np.isfinite(point): return np.nan, np.nan, np.nan
    boot = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = eta_ratio_of_means(T[idx], Q[idx], rpm[idx], rho, disk_A, mN, mD)
    var = float(np.var(boot, ddof=1))
    se  = float(np.sqrt(var))
    return point, var, se


# ----------------------------- IO: DOE, raw hover -----------------------------

def read_doe(path_csv: str) -> pd.DataFrame:
    if not path_csv or not os.path.exists(path_csv):
        return pd.DataFrame()
    df = pd.read_csv(path_csv)
    if "k_id" in df.columns:
        df["k_id"] = df["k_id"].astype(str)
    return df

def resolve_doe_optima_path(paths: Dict) -> str | None:
    if paths.get("doe_optima"):
        return paths["doe_optima"]
    base = paths.get("doe_csv")
    if not base: return None
    p = Path(base).parent / "doe_optima.csv"
    return str(p) if p.exists() else None

def read_hover_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    esc_col = find_col(df, RAW_NAME_HINTS["esc"])
    rpm_col = find_col(df, RAW_NAME_HINTS["rpm"])
    T_col   = find_col(df, RAW_NAME_HINTS["T"])
    Q_col   = find_col(df, RAW_NAME_HINTS["Q"])
    if not all([esc_col, rpm_col, T_col, Q_col]):
        raise ValueError(f"{path.name}: missing required columns (esc/rpm/T/Q)")

    out = pd.DataFrame({
        "esc_signal": df[esc_col].astype(float),
        "rpm": df[rpm_col].astype(float),
        "T": df[T_col].astype(float),
        "Q": df[Q_col].astype(float),
    })
    vib_col = find_col(df, RAW_NAME_HINTS["vib"])
    if vib_col:
        out["vibration_g"] = df[vib_col].astype(float)

    kid_col = find_col(df, RAW_NAME_HINTS["k_id"])
    if kid_col:
        out["k_id"] = df[kid_col].astype(str)
    else:
        m = re.search(r"(LS_\d{3}_\d{2})", path.stem)
        out["k_id"] = m.group(1) if m else path.stem

    out["source_file"] = path.name
    return out

def load_hover_raw_for_k(dir_hover_raw: str, wanted_k: set[str]) -> pd.DataFrame:
    if not dir_hover_raw or not os.path.isdir(dir_hover_raw) or not wanted_k:
        return pd.DataFrame()
    files = sorted([p for p in Path(dir_hover_raw).glob("*.csv") if not p.name.lower().startswith("tare")])
    dfs = []
    for p in files:
        try:
            dfp = read_hover_csv(p)
            if "k_id" not in dfp.columns: continue
            sub = dfp[dfp["k_id"].isin(wanted_k)]
            if not sub.empty:
                dfs.append(sub)
        except Exception as e:
            warnings.warn(f"Skipping {p.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ----------------------------- Process hover steps (new only) -----------------------------

def process_hover_steps(df_hover: pd.DataFrame, cfgd: Dict) -> pd.DataFrame:
    """Return a master-like dataframe for HOVER ONLY with new k_id rows."""
    if df_hover.empty:
        return pd.DataFrame()
    rho = cfgd["rho"]; disk_A = cfgd["disk_A"]; r_tip = cfgd["r_tip"]
    tare = load_tare_interps(cfgd["paths"]["hover_tare"])

    rows = []
    # Drop the entire first step per k where esc_signal==1000
    # We'll do that by grouping by k and filtering steps.
    for k_id, gk in df_hover.groupby("k_id", sort=False):
        # find steps (esc_signal unique), drop the one == 1000 if present
        steps = sorted(gk["esc_signal"].unique())
        for esc in steps:
            try:
                if float(esc) == 1000.0:
                    continue
            except Exception:
                pass
            g = gk[gk["esc_signal"]==esc]

            T   = g["T"].to_numpy(float)
            Q   = g["Q"].to_numpy(float)
            rpm = g["rpm"].to_numpy(float)

            if tare:
                T = T - tare["T_mean"](rpm)
                Q = Q - tare["Q_mean"](rpm)

            mT, _, _ = estimator_select(T)
            mQ, _, _ = estimator_select(Q)
            mR, _, _ = estimator_select(rpm)

            T_loc = estimator_apply(T, mT)
            Q_loc = estimator_apply(Q, mQ)
            R_loc = estimator_apply(rpm, mR)

            mN = "trim10"; mD = "trim10"  # safe defaults (already robust above)
            eta_point, eta_var, eta_se = bootstrap_eta_pairs(T, Q, rpm, rho, disk_A, mN, mD, B=500)

            omega_loc = R_loc * (2.0*np.pi/60.0)
            v_tip = omega_loc * r_tip

            rows.append({
                "flight_mode":"hover",
                "k_id": k_id,
                "esc_signal": esc,
                "rpm": R_loc,
                "v": v_tip,
                "performance": eta_point,
                "performance_variance": eta_var
            })

    return pd.DataFrame(rows)


# ----------------------------- GPR & uncertainty metric -----------------------------

def build_hover_pipe(D):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-9,          # replaced per-fit with heteroscedastic vector
            normalize_y=False,
            n_restarts_optimizer=8,
            random_state=0,
        ))
    ])

def bounds_from_config_or_fixed(cfg: dict,
                                df_hover_orig: pd.DataFrame,
                                features: List[str]) -> np.ndarray:
    """
    Return a fixed bounds box in the order of `features`.
    Priority:
      - config.optimization.bounds (AR, lambda, twist, hover.alpha, hover.v)
      - else empirical min/max from ORIGINAL HOVER SET (df_hover_orig), to keep the region fixed
    """
    opt = cfg.get("optimization", {})
    cfgb = opt.get("bounds", None)

    # Defaults from original hover set
    def emp(col):
        lo = float(df_hover_orig[col].min()); hi = float(df_hover_orig[col].max())
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            # pad slightly if degenerate
            eps = 1e-6; return (lo-eps, lo+eps)
        return (lo, hi)

    if isinstance(cfgb, dict):
        def pick(d, k, fallback):
            if isinstance(d, (list, tuple)) and len(d)==2:
                return (float(d[0]), float(d[1]))
            return fallback
        # geometry
        b_AR = pick(cfgb.get("AR"), None, None) if isinstance(cfgb.get("AR"), (list,tuple)) else None
        b_la = pick(cfgb.get("lambda"), None, None) if isinstance(cfgb.get("lambda"), (list,tuple)) else None
        b_tw = pick(cfgb.get("twist"), None, None) if isinstance(cfgb.get("twist"), (list,tuple)) else None
        hov  = cfgb.get("hover", {})
        b_ah = pick(hov.get("alpha"), None, None) if isinstance(hov, dict) else None
        b_vh = pick(hov.get("v"), None, None) if isinstance(hov, dict) else None
    else:
        b_AR = b_la = b_tw = b_ah = b_vh = None

    bounds_map = {
        "AR":      b_AR or emp("AR"),
        "lambda":  b_la or emp("lambda"),
        "twist":   b_tw or emp("twist"),
        "alpha":   b_ah or emp("alpha"),
        "v":       b_vh or emp("v"),
    }
    return np.array([bounds_map[f] for f in features], float)

def global_uncertainty_percent(model, features: List[str], bounds: np.ndarray, n_samples=10000) -> float:
    """
    RAW Mean-Normalized Uncertainty (%):
      100 * mean(std(x)) / median(mu(x)), sampling X ~ LHS(bounds)
    """
    sampler = LatinHypercube(d=len(features), seed=123)
    U = sampler.random(n=n_samples)
    X = bounds[:,0] + U*(bounds[:,1]-bounds[:,0])
    X = pd.DataFrame(X, columns=features)
    mu, s = model.predict(X, return_std=True)
    med = np.median(mu)
    if not np.isfinite(med) or med == 0:
        return np.nan
    return 100.0 * float(np.mean(s)) / float(med)


# ----------------------------- Main -----------------------------

def main():
    cfgd = load_config()
    cfg  = cfgd["cfg"]
    P    = cfgd["paths"]
    features = cfgd["features"]

    if features != ["AR","lambda","twist","alpha","v"]:
        raise ValueError("This script assumes input_cols=['AR','lambda','twist','alpha','v'].")

    Path(P["outputs_tables"]).mkdir(parents=True, exist_ok=True)

    # Load master + original hover slice
    master_pq = Path(P["master_parquet"])
    if not master_pq.exists():
        raise FileNotFoundError(f"Master parquet not found: {master_pq}")
    df_master = pd.read_parquet(master_pq)

    dH_orig = df_master[df_master["flight_mode"]=="hover"].dropna(subset=features+["performance"]).copy()
    if dH_orig.empty:
        raise ValueError("No hover data in master parquet; run 01 first.")

    # Construct fixed bounds for evaluation (config bounds preferred; else ORIGINAL hover empirical)
    bounds_fixed = bounds_from_config_or_fixed(cfg, dH_orig, features)

    # --- Train baseline hover model on ORIGINAL
    X0 = dH_orig[features].copy()
    y0 = dH_orig["performance"].astype(float).copy()
    if "performance_variance" not in dH_orig.columns:
        raise ValueError("Hover rows lack 'performance_variance' in master parquet.")
    a0 = dH_orig["performance_variance"].astype(float).to_numpy()

    pipe0 = build_hover_pipe(D=len(features))
    pipe0.named_steps["gpr"].alpha = np.maximum(a0, 1e-12)
    pipe0.fit(X0, y0)

    U_before = global_uncertainty_percent(pipe0, features, bounds_fixed, n_samples=10000)

    # --- Identify NEW k from doe_optima.csv (not present in doe_initial.csv)
    doe_init = read_doe(P["doe_csv"])
    doe_opt_path = resolve_doe_optima_path(P)
    doe_opt = read_doe(doe_opt_path) if doe_opt_path else pd.DataFrame()

    if doe_opt.empty or doe_init.empty or "k_id" not in doe_opt.columns or "k_id" not in doe_init.columns:
        warnings.warn("DOE optima or initial missing/invalid; no new k identified. Exiting after baseline summary.")
        out = pd.DataFrame([{
            "U_before_percent": U_before, "U_after_percent": np.nan,
            "reduction_percent": np.nan, "n_new_k": 0, "n_new_steps": 0, "k_new_list": ""
        }])
        out.to_csv(Path(P["outputs_tables"]) / "06_active_learning_update.csv", index=False)
        print("Wrote baseline-only summary (no new k found).")
        return

    k_old = set(doe_init["k_id"].astype(str).unique())
    k_new = sorted(list(set(doe_opt["k_id"].astype(str).unique()) - k_old))
    if not k_new:
        warnings.warn("No new k_id found in doe_optima vs doe_initial; nothing to add.")
        out = pd.DataFrame([{
            "U_before_percent": U_before, "U_after_percent": np.nan,
            "reduction_percent": np.nan, "n_new_k": 0, "n_new_steps": 0, "k_new_list": ""
        }])
        out.to_csv(Path(P["outputs_tables"]) / "06_active_learning_update.csv", index=False)
        print("Wrote baseline-only summary (no new k found).")
        return

    # --- Load raw hover for NEW k and process per-step robust stats
    df_hover_new_raw = load_hover_raw_for_k(P["hover_raw"], set(k_new))
    if df_hover_new_raw.empty:
        warnings.warn("No hover raw CSVs found for new k_id; cannot augment.")
        out = pd.DataFrame([{
            "U_before_percent": U_before, "U_after_percent": np.nan,
            "reduction_percent": np.nan, "n_new_k": len(k_new), "n_new_steps": 0, "k_new_list": ";".join(k_new)
        }])
        out.to_csv(Path(P["outputs_tables"]) / "06_active_learning_update.csv", index=False)
        print("Wrote baseline-only summary (no raw found).")
        return

    df_hover_new_master = process_hover_steps(df_hover_new_raw, cfgd)

    # Join geometry/alpha from DOE optima (k_id, AR, lambda, twist, alpha)
    use_cols = ["k_id","AR","lambda","twist","alpha"]
    have_cols = [c for c in use_cols if c in doe_opt.columns]
    if have_cols:
        df_hover_new_master = df_hover_new_master.merge(
            doe_opt[have_cols].drop_duplicates("k_id"), on="k_id", how="left"
        )
    else:
        warnings.warn("DOE optima missing geometry columns; new rows will lack AR/lambda/twist/alpha if not in raw.")

    # Ensure required feature columns exist
    for c in features:
        if c not in df_hover_new_master.columns:
            df_hover_new_master[c] = np.nan

    # Drop rows without performance OR missing any feature
    df_hover_new_master = df_hover_new_master.dropna(subset=["performance"] + features)
    n_new_steps = int(len(df_hover_new_master))

    if n_new_steps == 0:
        warnings.warn("Processed new hover data has no valid steps after cleaning.")
        out = pd.DataFrame([{
            "U_before_percent": U_before, "U_after_percent": np.nan,
            "reduction_percent": np.nan, "n_new_k": len(k_new), "n_new_steps": 0, "k_new_list": ";".join(k_new)
        }])
        out.to_csv(Path(P["outputs_tables"]) / "06_active_learning_update.csv", index=False)
        print("Wrote baseline-only summary (new steps invalid).")
        return

    # --- Augment ORIGINAL hover set with NEW steps and retrain
    dH_aug = pd.concat([dH_orig[features + ["performance","performance_variance"]],
                        df_hover_new_master[features + ["performance","performance_variance"]]],
                       ignore_index=True)

    X1 = dH_aug[features].copy()
    y1 = dH_aug["performance"].astype(float).copy()
    a1 = dH_aug["performance_variance"].astype(float).to_numpy()

    pipe1 = build_hover_pipe(D=len(features))
    pipe1.named_steps["gpr"].alpha = np.maximum(a1, 1e-12)
    pipe1.fit(X1, y1)

    U_after = global_uncertainty_percent(pipe1, features, bounds_fixed, n_samples=10000)

    # --- Write summary
    red = (U_before - U_after)/U_before*100.0 if (np.isfinite(U_before) and U_before>0 and np.isfinite(U_after)) else np.nan
    out = pd.DataFrame([{
        "U_before_percent": U_before,
        "U_after_percent":  U_after,
        "reduction_percent": red,
        "n_new_k": len(k_new),
        "n_new_steps": n_new_steps,
        "k_new_list": ";".join(k_new)
    }])
    out_path = Path(P["outputs_tables"]) / "06_active_learning_update.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    # --- Optional: std field sample before/after (for plots/inspection)
    sampler = LatinHypercube(d=len(features), seed=321)
    n_field = 4000
    U = sampler.random(n=n_field)
    Xs = bounds_fixed[:,0] + U*(bounds_fixed[:,1]-bounds_fixed[:,0])
    dfX = pd.DataFrame(Xs, columns=features)
    _, s0 = pipe0.predict(dfX, return_std=True)
    _, s1 = pipe1.predict(dfX, return_std=True)

    std_field = dfX.copy()
    std_field["std_before"] = s0
    std_field["std_after"]  = s1
    std_field["std_abs_reduction"] = s0 - s1
    with np.errstate(divide="ignore", invalid="ignore"):
        std_field["std_rel_reduction_percent"] = np.where(s0>0, (s0 - s1)/s0*100.0, np.nan)

    field_path = Path(P["outputs_tables"]) / "06_hover_std_field.csv"
    std_field.to_csv(field_path, index=False)
    print(f"Wrote: {field_path}")

    print("Active-learning uncertainty update complete.")

if __name__ == "__main__":
    main()
