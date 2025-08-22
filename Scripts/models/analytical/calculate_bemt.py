# Static BEMT across DOE & RPM grid. Importable API + CLI that writes detailed grid & avg prior.
import os, sys, numpy as np, pandas as pd
from scipy.interpolate import LinearNDInterpolator

# project roots to import Scripts.path_utils when run via CLI
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
SCRIPTS_DIR = os.path.join(PROJ_ROOT, "Scripts")
if SCRIPTS_DIR not in sys.path: sys.path.append(SCRIPTS_DIR)

# constants
RHO = 1.225
MU_AIR = 1.81e-5
R_HUB = 0.046
SPAN_BLADE = 0.184
R_TIP = R_HUB + SPAN_BLADE
D = 2.0 * R_TIP
A = np.pi * (D/2.0)**2

NUM_BLADES = 2
NUM_ELEMENTS = 16
MAX_ITERS = 60
RELAX = 0.5
PHI_FALLBACK_DEG = 2.5

def _bin_nearest(v, w):
    v = np.asarray(v, float)
    return (np.floor((v + 0.5*w)/w)*w).astype(int)

def _load_polars_all(airfoil_dir):
    import glob
    if not airfoil_dir or not os.path.isdir(airfoil_dir): return None
    combined = os.path.join(airfoil_dir, "naca0012_all_re.csv")
    if os.path.exists(combined):
        df = pd.read_csv(combined)
        cols = {c.lower(): c for c in df.columns}
        if {"reynolds","alpha","cl","cd"}.issubset(cols):
            out = pd.DataFrame({
                "Re":        pd.to_numeric(df[cols["reynolds"]], errors="coerce"),
                "alpha_deg": pd.to_numeric(df[cols["alpha"]], errors="coerce"),
                "cl":        pd.to_numeric(df[cols["cl"]], errors="coerce"),
                "cd":        pd.to_numeric(df[cols["cd"]], errors="coerce"),
            }).dropna()
            return out if not out.empty else None
    rows=[]
    for fp in glob.glob(os.path.join(airfoil_dir, "*.csv")):
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
                "Re":        Re*np.ones(len(df), float),
                "alpha_deg": pd.to_numeric(df[col["alpha_deg"]], errors="coerce"),
                "cl":        pd.to_numeric(df[col["cl"]], errors="coerce"),
                "cd":        pd.to_numeric(df[col["cd"]], errors="coerce"),
            }).dropna()
            if not tmp.empty: rows.append(tmp)
    if not rows: return None
    return pd.concat(rows, ignore_index=True)

def _build_interpolators(polars):
    pts = polars[["Re","alpha_deg"]].to_numpy(float)
    clv = polars["cl"].to_numpy(float)
    cdv = polars["cd"].to_numpy(float)
    cl_itp = LinearNDInterpolator(pts, clv, fill_value=np.nan)
    cd_itp = LinearNDInterpolator(pts, cdv, fill_value=np.nan)
    a_min, a_max = polars["alpha_deg"].min(), polars["alpha_deg"].max()
    re_min, re_max = polars["Re"].min(), polars["Re"].max()
    return cl_itp, cd_itp, float(a_min), float(a_max), float(re_min), float(re_max)

def _lookup_cl_cd(cl_itp, cd_itp, Re, alpha_deg, re_min, re_max, a_min, a_max):
    Re = np.clip(Re, re_min, re_max); alpha_deg = np.clip(alpha_deg, a_min, a_max)
    Q = np.column_stack([Re, alpha_deg])
    cl = cl_itp(Q); cd = cd_itp(Q)
    bad = ~np.isfinite(cl)
    if bad.any():
        Qj = Q.copy()
        Qj[bad,0] = 0.999*Qj[bad,0] + 0.001*(0.5*(re_min+re_max))
        Qj[bad,1] = 0.999*Qj[bad,1] + 0.001*(0.5*(a_min+a_max))
        cl[bad] = cl_itp(Qj[bad]); cd[bad] = cd_itp(Qj[bad])
    cl = np.where(np.isfinite(cl), cl, 0.0)
    cd = np.where(np.isfinite(cd), cd, 1e-3)
    cd = np.clip(cd, 1e-4, None)
    return cl, cd

def _eval_section(chord, twist_local, r, Vax, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max):
    phi = np.arctan2(Vax, Vtan)
    alpha = twist_local - phi
    a_deg = np.degrees(alpha)
    Vrel = np.hypot(Vax, Vtan)
    Re = (RHO * Vrel * chord) / MU_AIR
    cl, cd = _lookup_cl_cd(cl_itp, cd_itp, Re, a_deg, re_min, re_max, a_min, a_max)
    q = 0.5 * RHO * Vrel**2
    L  = q * chord * cl
    Df = q * chord * cd
    return L, Df, phi

def _run_bemt_row(row, rpm, cl_itp, cd_itp, a_min, a_max, re_min, re_max):
    r_edges = np.linspace(R_HUB, R_TIP, NUM_ELEMENTS + 1)
    r = 0.5*(r_edges[:-1] + r_edges[1:]); dr = r_edges[1] - r_edges[0]
    AR = float(row['AR']); lam = float(row['lambda'])
    aoa_root = float(row['aoaRoot (deg)']); aoa_tip  = float(row['aoaTip (deg)'])
    root_chord = (2.0 * SPAN_BLADE) / (AR * (1.0 + lam))
    tip_chord  = lam * root_chord
    chord = root_chord + (tip_chord - root_chord) * (r - R_HUB) / SPAN_BLADE
    twist_local = np.deg2rad(aoa_root + (aoa_tip - aoa_root) * (r - R_HUB) / SPAN_BLADE)
    omega = rpm * (2*np.pi/60.0); Vtan = omega * r
    v_i = np.zeros_like(r)
    for _ in range(MAX_ITERS):
        L, Df, phi = _eval_section(chord, twist_local, r, v_i, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
        sin_phi = np.sin(np.clip(phi, 1e-6, None))
        F = (2/np.pi) * np.arccos(np.exp(-(NUM_BLADES/2.0) * (R_TIP - r) / (r * sin_phi)))
        F = np.clip(F, 1e-3, 1.0)
        dT = (L * np.cos(phi) - Df * np.sin(phi)) * NUM_BLADES
        vi_new = dT / (4.0 * np.pi * r * RHO * F + 1e-12)
        v_i = RELAX*v_i + (1-RELAX)*np.clip(vi_new, 0.0, None)
    L, Df, phi = _eval_section(chord, twist_local, r, v_i, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
    sin_phi = np.sin(np.clip(phi, 1e-6, None)); cos_phi = np.cos(phi)
    dT = (L * cos_phi - Df * sin_phi) * NUM_BLADES
    dQ = (L * sin_phi + Df * cos_phi) * r * NUM_BLADES
    T = float(np.sum(dT * dr))
    Q_total = float(np.sum(dQ * dr))
    Q_profile = float(np.sum((Df * cos_phi) * r * NUM_BLADES * dr))
    if T <= 0.0:
        phi_fb = np.deg2rad(PHI_FALLBACK_DEG)
        Vax_fb = np.tan(phi_fb) * Vtan
        Lf, Dff, phif = _eval_section(chord, twist_local, r, Vax_fb, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
        sin_f = np.sin(np.clip(phif, 1e-6, None)); cos_f = np.cos(phif)
        dT_fb = (Lf * cos_f - Dff * sin_f) * NUM_BLADES
        dQ_fb = (Lf * sin_f + Dff * cos_f) * r * NUM_BLADES
        T = float(np.sum(dT_fb * dr)); Q_total = float(np.sum(dQ_fb * dr))
    Q = Q_total if Q_total > 0.0 else max(Q_profile, 0.0)
    T = max(T, 0.0); omega = max(omega, 1e-9)
    P_mech = Q * omega
    P_ideal = np.sqrt(T**3 / (2 * RHO * A)) if T > 0 else 0.0
    eta = float(np.clip(P_ideal / P_mech if P_mech > 1e-12 else 0.0, 0.0, 1.0))
    n = rpm / 60.0
    CT = T / (RHO * (n**2) * (D**4) + 1e-12)
    CP = P_mech / (RHO * (n**3) * (D**5) + 1e-12)
    r75 = R_HUB + 0.75*SPAN_BLADE
    i75 = int(np.argmin(np.abs(r - r75)))
    lam75 = float((v_i[i75] / (omega * r[i75])) if omega*r[i75] > 1e-12 else 0.0)
    phi75 = float(np.degrees(phi[i75]))
    return T, Q, eta, CT, CP, lam75, phi75

def bemt_avg_for_doe(doe_df, airfoil_dir, rpm_min=350, rpm_max=4000, rpm_step=50, bin_rpm=150):
    polars = _load_polars_all(airfoil_dir)
    if polars is None or polars.empty:
        raise RuntimeError(f"No airfoil polars found in {airfoil_dir}")
    cl_itp, cd_itp, a_min, a_max, re_min, re_max = _build_interpolators(polars)
    rpm_list = np.arange(rpm_min, rpm_max+1, rpm_step)
    rows=[]
    for _, row in doe_df.iterrows():
        etas=[]
        for rpm in rpm_list:
            _, _, eta, *_ = _run_bemt_row(row, rpm, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
            etas.append(eta)
        rows.append({"filename": row["filename"], "eta_bemt_mean": float(np.mean(etas))})
    return pd.DataFrame(rows)

def _cli():
    # Resolve paths via Scripts/path_utils
    from path_utils import load_cfg
    C = load_cfg()
    tools = C["paths"]["tools_dir"]; airfoil_dir = C["paths"]["airfoil_dir"]; doe_csv=C["paths"]["doe_csv"]
    os.makedirs(tools, exist_ok=True)
    rpm_min, rpm_max, rpm_step, bin_rpm = 350, 4000, 50, 150
    doe = pd.read_csv(doe_csv)[["filename","AR","lambda","aoaRoot (deg)","aoaTip (deg)"]].drop_duplicates("filename")
    polars = _load_polars_all(airfoil_dir)
    if polars is None: raise SystemExit(f"No airfoil polars found under {airfoil_dir}")
    cl_itp, cd_itp, a_min, a_max, re_min, re_max = _build_interpolators(polars)
    rpm_list = np.arange(rpm_min, rpm_max+1, rpm_step)
    grid=[]
    for idx, r in doe.iterrows():
        fname = r["filename"]
        for rpm in rpm_list:
            T, Q, eta, CT, CP, lam75, phi75 = _run_bemt_row(r, rpm, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
            grid.append({
                "filename": fname,
                "rpm": int(rpm),
                "rpm_bemt": int(_bin_nearest(rpm, bin_rpm)),
                "prop_eff_bemt": float(eta),
                "CT_bemt": float(CT),
                "CP_bemt": float(CP),
                "lambda75_bemt": float(lam75),
                "phi75_bemt": float(phi75),
            })
        print(f"  BEMT: finished {idx+1}/{len(doe)} -> {fname}")
    df_grid = pd.DataFrame(grid)
    df_grid.to_csv(os.path.join(tools, "bemt_predictions_final.csv"), index=False)
    avg = df_grid.groupby("filename", as_index=False)["prop_eff_bemt"].mean().rename(columns={"prop_eff_bemt":"eta_bemt_mean"})
    avg["eta_bemt_mean"] = avg["eta_bemt_mean"].clip(0,1)
    avg.to_csv(os.path.join(tools, "bemt_avg_prior.csv"), index=False)
    print(f"Wrote bemt_predictions_final.csv and bemt_avg_prior.csv to {tools}")

if __name__ == "__main__":
    _cli()
