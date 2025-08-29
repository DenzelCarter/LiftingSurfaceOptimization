# Polar-aware lifting-line prior (cruise) using naca0012_all_re.csv
# Writes: Experiment/outputs/tables/ll_cruise_prior.csv

import os, sys, numpy as np, pandas as pd
from math import pi
from scipy.interpolate import LinearNDInterpolator

# Make ../../ (Scripts/) importable
_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from path_utils import load_cfg

def _build_airfoil_itp(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[["reynolds","alpha"]].to_numpy()
    clv = df["cl"].to_numpy(float)
    cdv = df["cd"].to_numpy(float)
    a_min, a_max = float(df["alpha"].min()), float(df["alpha"].max())
    cl_itp = LinearNDInterpolator(pts, clv, fill_value=np.nan)
    cd_itp = LinearNDInterpolator(pts, cdv, fill_value=np.nan)
    return cl_itp, cd_itp, a_min, a_max

def _planform(AR, lam, span_ref):
    b = span_ref; S = b*b/AR
    c_root = 2.0*S/(b*(1.0+lam))
    MAC = (2.0/3.0)*c_root*(1.0+lam+lam*lam)/(1.0+lam)
    return S, MAC

def _oswald(AR, lam, e_base, k):
    return float(np.clip(e_base - k*(1.0-lam)**2, 0.70, 0.98))

def _cd_polar_at_CL(Re, CL, cl_itp, cd_itp, a_min, a_max):
    alphas = np.linspace(a_min, a_max, 361)
    R = np.column_stack([np.full_like(alphas, Re, float), alphas])
    cl = cl_itp(R); cd = cd_itp(R)
    mask = np.isfinite(cl) & np.isfinite(cd)
    if not np.any(mask): return np.nan
    i = int(np.argmin(np.abs(cl[mask] - CL)))
    return float(cd[mask][i])

def main():
    C = load_cfg()
    # paths
    doe_path   = C["paths"]["doe_csv"]
    airfoil_csv = C["paths"]["airfoil_csv"]
    out_tables = C["paths"]["outputs_tables_dir"]
    os.makedirs(out_tables, exist_ok=True)

    gcols = C["geometry_cols"]
    doe = pd.read_csv(doe_path)[["filename"]+gcols].drop_duplicates("filename")
    for c in gcols: doe[c] = pd.to_numeric(doe[c], errors="coerce")
    doe = doe.dropna(subset=gcols).reset_index(drop=True)

    use_polar = bool(C["cruise_ll"]["use_polar"])
    rho = float(C["fluids"]["rho"]); mu = float(C["fluids"]["mu"])
    cl_itp = cd_itp = a_min = a_max = None
    if use_polar:
        cl_itp, cd_itp, a_min, a_max = _build_airfoil_itp(airfoil_csv)

    V = float(C["cruise_ll_polar"]["V_cruise"])
    span_ref = C["cruise_ll_polar"]["span_ref_m"]
    if span_ref is None:
        span_ref = 2.0 * float(C["geometry"]["span_blade_m"])
    CLt = float(C["cruise_ll"]["target_CL"])
    e_base = float(C["cruise_ll"]["e_base"]); e_k = float(C["cruise_ll"]["e_taper_k"])
    CD0 = float(C["cruise_ll"]["CD0"]); K2 = float(C["cruise_ll"]["K2"])

    LD=[]
    for _, r in doe.iterrows():
        AR = float(r["AR"]); lam = float(r["lambda"])
        e  = _oswald(AR, lam, e_base, e_k)
        if use_polar:
            _, MAC = _planform(AR, lam, span_ref)
            Re = rho*V*MAC/max(mu, 1e-12)
            cd_p = _cd_polar_at_CL(Re, CLt, cl_itp, cd_itp, a_min, a_max)
            if not np.isfinite(cd_p):
                cd_p = CD0 + K2*CLt*CLt
        else:
            cd_p = CD0 + K2*CLt*CLt
        CD = cd_p + CLt*CLt/(pi*AR*e)
        LD.append(float(CLt/CD) if CD>1e-12 else np.nan)

    out = pd.DataFrame({"filename": doe["filename"], "LD_ll": np.asarray(LD, float)})
    out.to_csv(os.path.join(out_tables, "ll_cruise_prior.csv"), index=False)
    print("Saved:", os.path.join(out_tables, "ll_cruise_prior.csv"))

if __name__ == "__main__":
    main()
