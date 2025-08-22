# Lifting-Line cruise prior (NACA0012). Uses airfoil polars if present; otherwise fallback drag model.
import os, glob, numpy as np, pandas as pd
from math import pi

def _load_polars(airfoil_dir: str):
    if not airfoil_dir or not os.path.isdir(airfoil_dir):
        return None
    combined = os.path.join(airfoil_dir, "naca0012_all_re.csv")
    if os.path.exists(combined):
        df = pd.read_csv(combined)
        cols = {c.lower(): c for c in df.columns}
        if {"reynolds","alpha","cl","cd"}.issubset(cols):
            out = pd.DataFrame({
                "alpha_deg": pd.to_numeric(df[cols["alpha"]], errors="coerce"),
                "cl":        pd.to_numeric(df[cols["cl"]], errors="coerce"),
                "cd":        pd.to_numeric(df[cols["cd"]], errors="coerce"),
                "Re":        pd.to_numeric(df[cols["reynolds"]], errors="coerce"),
            }).dropna()
            return out if not out.empty else None
    files = glob.glob(os.path.join(airfoil_dir, "*.csv"))
    rows=[]
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        if "Re" not in base: continue
        try:
            token = base.split("Re",1)[1]
            num = "".join([ch for ch in token if ch.isdigit() or ch in ".eE-+"])
            Re = float(num)
        except Exception:
            continue
        df = pd.read_csv(fp)
        col = {c.lower(): c for c in df.columns}
        if {"alpha_deg","cl","cd"}.issubset(col):
            tmp = pd.DataFrame({
                "alpha_deg": pd.to_numeric(df[col["alpha_deg"]], errors="coerce"),
                "cl":        pd.to_numeric(df[col["cl"]], errors="coerce"),
                "cd":        pd.to_numeric(df[col["cd"]], errors="coerce"),
                "Re":        Re*np.ones(len(df), float),
            }).dropna()
            if not tmp.empty: rows.append(tmp)
    if not rows: return None
    return pd.concat(rows, ignore_index=True)

def _cdp_at_cl(polars, target_cl):
    cds=[]
    for Re, grp in polars.groupby("Re"):
        g = grp.sort_values("cl").drop_duplicates("cl")
        if len(g)<3: continue
        cd = np.interp(target_cl, g["cl"].to_numpy(float), g["cd"].to_numpy(float), left=np.nan, right=np.nan)
        if np.isfinite(cd): cds.append(cd)
    if not cds: return None
    return float(np.nanmean(cds))

def _e_from_taper(AR, lam, e_base, k):
    return max(0.70, min(0.98, e_base - k*(1.0 - lam)**2))

def ll_prior_for_doe(doe_df, target_CL, airfoil_dir, CD0, K2, e_base, e_taper_k):
    polars = _load_polars(airfoil_dir)
    CDp_est = _cdp_at_cl(polars, target_CL) if polars is not None else None
    rows=[]
    for _, r in doe_df.iterrows():
        AR  = float(r["AR"]); lam = float(r["lambda"])
        e   = _e_from_taper(AR, lam, e_base, e_taper_k)
        CDp = CDp_est if CDp_est is not None else (CD0 + K2*(target_CL**2))
        CD  = CDp + target_CL**2/(pi*AR*e)
        LD  = target_CL/CD if CD>1e-12 else np.nan
        rows.append({"filename": r["filename"], "LD_ll": LD, "CL_target": target_CL, "e_oswald": e, "CDp": CDp})
    return pd.DataFrame(rows)
