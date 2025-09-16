# 01_process_data.py
# Ingest hover/cruise data, compute robust per-step hover stats, and build master parquet.
# Hover:
#   - Per (k_id, esc_signal) step, dynamic estimators for T/Q/RPM/IMU; pairs bootstrap for eta variance.
#   - Drop entire first step per k where esc_signal==1000 (artifact).
#   - Keep ONLY hover k_id present in DOE (golden sample excluded).
# Cruise:
#   - Read comsol_cruise_01.txt; performance = comp1.L / comp1.D.
#   - Assign cruise k_id by grouping unique geometry (AR, lambda, twist) → same k across alpha sweep.
# Outputs:
#   - paths.master_parquet
#   - paths.outputs_tables/01_hover_step_diagnostics.csv

import os
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import trim_mean, skew, kurtosis

import path_utils

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
        "cruise_comsol":  P.get("dir_cruise_comsol"),
        "processed_dir":  P.get("dir_processed"),
        "master_parquet": P.get("master_parquet"),
        "doe_csv":        P.get("doe_csv"),
        "outputs_tables": P.get("outputs_tables") or P.get("dir_processed"),
    }

    rho_default = 1.204
    nu_default  = 1.51e-5
    rho = float(F.get("rho", rho_default)) if F is not None else rho_default
    nu  = float(F.get("nu", nu_default)) if F is not None else nu_default
    mu  = float(F.get("mu", rho*nu)) if F is not None else rho*nu

    r_hub   = float(G.get("r_hub_m", 0.046))
    l_blade = float(G.get("l_blade_m", 0.184))
    r_tip   = float(G.get("r_tip_m", r_hub + l_blade))

    disk_A = float(np.pi * r_tip**2)  # full disk

    input_cols = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])

    return {
        "cfg": cfg,
        "paths": paths,
        "rho": rho, "mu": mu, "nu": nu,
        "r_hub": r_hub, "l_blade": l_blade, "r_tip": r_tip,
        "disk_A": disk_A,
        "input_cols": input_cols
    }

RAW_NAME_HINTS = {
    "esc": ["ESC signal (µs)", "ESC signal (us)", "esc", "esc_signal", "esc signal"],
    "rpm": ["Motor Electrical Speed (RPM)", "RPM", "rpm"],
    "T":   ["Thrust (N)", "thrust", "T (N)", "T"],
    "Q":   ["Torque (N·m)", "Torque (Nm)", "torque", "Q (N*m)", "Q (N·m)", "Q"],
    "k_id":["k_id", "filename", "file", "design_id"],
    "vib":["Vibration (g)","vibration (g)","vibration","imu (g)","imu_g","accel_g","accelerometer (g)"]
}

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

def load_hover_raw(dir_hover_raw: str) -> pd.DataFrame:
    if not dir_hover_raw or not os.path.isdir(dir_hover_raw):
        warnings.warn(f"Hover raw directory not found: {dir_hover_raw}")
        return pd.DataFrame()
    files = sorted([p for p in Path(dir_hover_raw).glob("*.csv") if not p.name.lower().startswith("tare")])
    if not files:
        warnings.warn(f"No hover CSVs found in {dir_hover_raw}")
        return pd.DataFrame()
    dfs = []
    for p in files:
        try:
            dfs.append(read_hover_csv(p))
        except Exception as e:
            warnings.warn(f"Skipping {p.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_tare_interps(dir_hover_tare: str):
    if not dir_hover_tare:
        return None
    candidates = ["Tare_01.csv", "No_Load_Rotate_01.csv", "tare.csv"]
    tare_file = None
    for name in candidates:
        p = Path(dir_hover_tare) / name
        if p.exists():
            tare_file = p; break
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
        "T_std":  interp("T_std"),
        "Q_mean": interp("Q_mean"),
        "Q_std":  interp("Q_std"),
    }

def estimator_select(x: np.ndarray) -> Tuple[str, float, float]:
    """
    Choose a robust location estimator for 1D samples x.
    Returns: (method, skewness g1, excess kurtosis g2)
      method ∈ {"mean","trim10","trim20","median"}

    Rules:
      - Prefer mean only when skew is tiny and tails are near-normal or flat (g2 ≤ ~0.5).
      - Trim earlier for leptokurtic (g2>0) than platykurtic (g2<0).
      - Escalate based on robust outlier fraction p_out from a 3*MAD threshold.
      - Small-n guard biases toward light/medium trimming unless outliers are heavy.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return "trim10", np.nan, np.nan  # neutral default on empty

    # Sample skew/kurtosis (excess kurtosis: 0 for normal)
    g1 = skew(x, bias=False) if n >= 3 else 0.0
    g2 = kurtosis(x, fisher=True, bias=False) if n >= 4 else 0.0

    # Robust outlier rate via MAD (fallback to IQR if MAD=0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    madn = 1.4826 * mad
    if madn > 0:
        p_out = float(np.mean(np.abs(x - med) > 3.0 * madn))
    else:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            p_out = float(np.mean((x < q1 - 3.0*iqr) | (x > q3 + 3.0*iqr)))
        else:
            p_out = 0.0

    # Initial choice from skew/kurtosis (asymmetric: heavier penalty for g2>0)
    if not np.isfinite(g1) or not np.isfinite(g2):
        m = "trim10"
    else:
        if (abs(g1) <= 0.3) and (-0.8 <= g2 <= 0.5):
            m = "mean"          # near-normal or flat tails, tiny skew
        elif (abs(g1) <= 0.9) and (g2 <= 1.5):
            m = "trim10"        # mild skew OR mild leptokurtic
        elif (abs(g1) <= 1.5) and (g2 <= 3.5):
            m = "trim20"        # moderate skew/tails
        else:
            m = "median"        # very heavy tails or strong asymmetry

        # Be forgiving for clearly platykurtic with tiny skew and few outliers
        if (g2 < -0.8) and (abs(g1) <= 0.5) and (p_out < 0.03):
            m = "mean"

    # Small-sample guard (avoid overreacting to noisy g1/g2)
    if n < 6:
        if p_out >= 0.15:
            m = "median"
        elif p_out >= 0.05:
            m = "trim20"
        else:
            m = "trim10"
        return m, float(g1), float(g2)

    # Escalate robustness based on outlier fraction
    if p_out >= 0.20:
        m = "median"
    elif p_out >= 0.08:
        if m == "mean":
            m = "trim10"
        elif m == "trim10":
            m = "trim20"
    elif p_out >= 0.03:
        if m == "mean":
            m = "trim10"

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

def bootstrap_loc_var(x: np.ndarray, method: str, B: int = 1000, rng=None):
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
                        B: int = 1000, rng=None):
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

def process_hover(df_hover: pd.DataFrame, cfgd: Dict):
    if df_hover.empty:
        return pd.DataFrame(), pd.DataFrame()
    rho = cfgd["rho"]; disk_A = cfgd["disk_A"]; r_tip = cfgd["r_tip"]

    tare = load_tare_interps(cfgd["paths"]["hover_tare"])

    groups = df_hover.groupby(["k_id","esc_signal"], sort=False)
    diag_rows = []
    master_rows = []

    for (k_id, esc), g in groups:
        # Skip the entire first step per k (artifact: esc_signal == 1000)
        try:
            if float(esc) == 1000.0:
                continue
        except Exception:
            pass

        T   = g["T"].to_numpy(float)
        Q   = g["Q"].to_numpy(float)
        rpm = g["rpm"].to_numpy(float)
        vib = g["vibration_g"].to_numpy(float) if "vibration_g" in g.columns else np.full(len(g), np.nan)

        if tare:
            T = T - tare["T_mean"](rpm)
            Q = Q - tare["Q_mean"](rpm)

        mT, skT, kuT = estimator_select(T)
        mQ, skQ, kuQ = estimator_select(Q)
        mR, skR, kuR = estimator_select(rpm)
        mV, skV, kuV = estimator_select(vib)

        T_loc = estimator_apply(T, mT); T_std_raw = safe_std(T)
        Q_loc = estimator_apply(Q, mQ); Q_std_raw = safe_std(Q)
        R_loc = estimator_apply(rpm, mR); R_std_raw = safe_std(rpm)
        V_loc = estimator_apply(vib, mV); V_std_raw = safe_std(vib)

        T_var_b, T_se_b = bootstrap_loc_var(T, mT, B=1000)
        Q_var_b, Q_se_b = bootstrap_loc_var(Q, mQ, B=1000)
        R_var_b, R_se_b = bootstrap_loc_var(rpm, mR, B=1000)
        V_var_b, V_se_b = bootstrap_loc_var(vib, mV, B=1000) if np.isfinite(vib).any() else (np.nan, np.nan)

        w = rpm * (2.0*np.pi/60.0)
        N = np.sqrt(np.clip(T,0.0,None)**3 / (2.0 * rho * disk_A))
        D = Q * w
        mN, skN, kuN = estimator_select(N)
        mD, skD, kuD = estimator_select(D)

        eta_point, eta_var, eta_se = bootstrap_eta_pairs(T, Q, rpm, rho, disk_A, mN, mD, B=1000)

        omega_loc = R_loc * (2.0*np.pi/60.0)
        v_tip = omega_loc * r_tip

        neg_T_frac = float(np.mean(T < 0)) if len(T) else np.nan

        master_rows.append({
            "flight_mode":"hover",
            "k_id": k_id,
            "esc_signal": esc,
            "v": v_tip,
            "rpm": R_loc,
            "performance": eta_point,
            "performance_variance": eta_var
        })

        diag_rows.append({
            "k_id": k_id, "esc_signal": esc, "n_reps": int(len(g)),
            "T_method": mT, "T_loc": T_loc, "T_std_raw": T_std_raw, "T_skew": skT, "T_kurt": kuT, "T_var_boot": T_var_b, "T_se_boot": T_se_b,
            "Q_method": mQ, "Q_loc": Q_loc, "Q_std_raw": Q_std_raw, "Q_skew": skQ, "Q_kurt": kuQ, "Q_var_boot": Q_var_b, "Q_se_boot": Q_se_b,
            "RPM_method": mR, "RPM_loc": R_loc, "RPM_std_raw": R_std_raw, "RPM_skew": skR, "RPM_kurt": kuR, "RPM_var_boot": R_var_b, "RPM_se_boot": R_se_b,
            "VIB_method": mV, "VIB_loc": V_loc, "VIB_std_raw": V_std_raw, "VIB_skew": skV, "VIB_kurt": kuV, "VIB_var_boot": V_var_b, "VIB_se_boot": V_se_b,
            "N_method": mN, "N_skew": skN, "N_kurt": kuN, "D_method": mD, "D_skew": skD, "D_kurt": kuD,
            "eta_point": eta_point, "eta_var_boot": eta_var, "eta_se_boot": eta_se,
            "v": v_tip, "neg_T_frac": neg_T_frac
        })

    return pd.DataFrame(master_rows), pd.DataFrame(diag_rows)

def load_doe(doe_csv: str) -> pd.DataFrame:
    if not doe_csv or not os.path.exists(doe_csv):
        warnings.warn(f"DOE CSV not found: {doe_csv}")
        return pd.DataFrame()
    df = pd.read_csv(doe_csv)
    if "material" in df.columns:
        df = df.drop(columns=["material"])
    return df

def join_geometry(df_master: pd.DataFrame, df_doe: pd.DataFrame) -> pd.DataFrame:
    if df_master.empty or df_doe.empty:
        return df_master
    use_cols = ["k_id","AR","lambda","twist","alpha"]
    missing = [c for c in use_cols if c not in df_doe.columns]
    if missing:
        warnings.warn(f"DOE missing columns: {missing}.")
        use_cols = [c for c in use_cols if c in df_doe.columns]
    merged = df_master.merge(df_doe[use_cols].drop_duplicates("k_id"), on="k_id", how="left")
    return merged

def load_cruise_comsol(dir_cruise_comsol: str) -> pd.DataFrame:
    if not dir_cruise_comsol or not os.path.isdir(dir_cruise_comsol):
        warnings.warn(f"Cruise COMSOL dir not found: {dir_cruise_comsol}")
        return pd.DataFrame()
    candidate = Path(dir_cruise_comsol) / "comsol_cruise_01.txt"
    if not candidate.exists():
        warnings.warn(f"{candidate.name} not found; skipping cruise.")
        return pd.DataFrame()

    names = ["AR","lambda","twist (deg)","aoa_root (deg)","U_cruise (m/s)","comp1.L","comp1.D"]
    df = pd.read_csv(candidate, sep=r"\s+", engine="python", comment="%", header=None, names=names)

    df = df.rename(columns={
        "twist (deg)":"twist",
        "aoa_root (deg)":"alpha",
        "U_cruise (m/s)":"v"
    })

    if "comp1.L" in df.columns and "comp1.D" in df.columns:
        df["performance"] = (df["comp1.L"].astype(float) / df["comp1.D"].astype(float)).replace([np.inf,-np.inf], np.nan)
    else:
        df["performance"] = np.nan

    df["performance_variance"] = 1e-6 * float(np.nanmax([1.0, df["performance"].abs().max(skipna=True)]))
    df["flight_mode"] = "cruise"
    df["esc_signal"] = np.nan
    df["rpm"] = np.nan

    keep = ["flight_mode","AR","lambda","twist","alpha","v","performance","performance_variance","esc_signal","rpm"]
    for c in keep:
        if c not in df.columns: df[c] = np.nan
    df = df[keep]

    # Assign cruise k_id by grouping identical geometry (AR, lambda, twist)
    # Round to mitigate floating noise before grouping
    gkeys = (df["AR"].round(6), df["lambda"].round(6), df["twist"].round(6))
    geom_code, uniques = pd.factorize(pd.Series(list(zip(*gkeys))))
    df["k_id"] = [f"CR_{i:03d}" for i in geom_code]

    return df[["flight_mode","k_id","esc_signal","rpm","AR","lambda","twist","alpha","v","performance","performance_variance"]]

def main():
    cfgd = load_config()
    P = cfgd["paths"]

    Path(P["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(P["outputs_tables"]).mkdir(parents=True, exist_ok=True)

    df_doe = load_doe(P["doe_csv"])
    df_hover_raw = load_hover_raw(P["hover_raw"])
    if not df_doe.empty and not df_hover_raw.empty and "k_id" in df_hover_raw.columns and "k_id" in df_doe.columns:
        allowed = set(df_doe["k_id"].astype(str).unique())
        df_hover_raw = df_hover_raw[df_hover_raw["k_id"].astype(str).isin(allowed)]

    df_hover_master, df_hover_diag = process_hover(df_hover_raw, cfgd)
    df_hover_master = join_geometry(df_hover_master, df_doe)

    frames = []
    if not df_hover_master.empty:
        needed = ["flight_mode","k_id","esc_signal","rpm","AR","lambda","twist","alpha","v","performance","performance_variance"]
        for c in needed:
            if c not in df_hover_master.columns:
                df_hover_master[c] = np.nan
        frames.append(df_hover_master[needed])

    df_cruise = load_cruise_comsol(P["cruise_comsol"])
    if not df_cruise.empty:
        frames.append(df_cruise)

    if frames:
        df_master = pd.concat(frames, ignore_index=True)
    else:
        df_master = pd.DataFrame(columns=["flight_mode","k_id","esc_signal","rpm","AR","lambda","twist","alpha","v","performance","performance_variance"])

    master_path = P["master_parquet"]
    df_master.to_parquet(master_path, index=False)
    print(f"Wrote master dataset: {master_path} ({len(df_master)} rows)")

    if not df_hover_diag.empty:
        diag_path = Path(P["outputs_tables"]) / "01_hover_step_diagnostics.csv"
        df_hover_diag.to_csv(diag_path, index=False)
        print(f"Wrote hover diagnostics: {diag_path} ({len(df_hover_diag)} rows)")

if __name__ == "__main__":
    main()
