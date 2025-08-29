# BEMT over DOE × RPM using airfoil polars (Re, alpha).
# Writes:
#   Experiment/outputs/tables/bemt_predictions_final.csv   (per-RPM grid)
#   Experiment/outputs/tables/bemt_avg_prior.csv           (rpm-averaged η per filename)

import os, sys, numpy as np, pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Make ../../ (Scripts/) importable, so we can import path_utils
_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from path_utils import load_cfg

def _bin_nearest(val, w):
    v = np.asarray(val, dtype=float)
    return (np.floor((v + 0.5 * w) / w) * w).astype(int)

def _build_airfoil_itp(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[["reynolds", "alpha"]].to_numpy()
    clv = df["cl"].to_numpy(float)
    cdv = df["cd"].to_numpy(float)
    a_min, a_max = float(df["alpha"].min()), float(df["alpha"].max())
    re_min, re_max = float(df["reynolds"].min()), float(df["reynolds"].max())
    cl_itp = LinearNDInterpolator(pts, clv, fill_value=np.nan)
    cd_itp = LinearNDInterpolator(pts, cdv, fill_value=np.nan)
    return cl_itp, cd_itp, a_min, a_max, re_min, re_max

def _lookup_cl_cd(cl_itp, cd_itp, Re, alpha_deg, re_min, re_max, a_min, a_max):
    Re_clp = np.clip(Re, re_min, re_max)
    a_clp  = np.clip(alpha_deg, a_min, a_max)
    Q = np.column_stack([Re_clp, a_clp])
    cl = cl_itp(Q); cd = cd_itp(Q)
    m = ~np.isfinite(cl)
    if m.any():
        Qj = Q.copy()
        Qj[m, 0] = 0.999*Qj[m, 0] + 0.001*(0.5*(re_min+re_max))
        Qj[m, 1] = 0.999*Qj[m, 1] + 0.001*(0.5*(a_min+a_max))
        cl[m] = cl_itp(Qj[m]); cd[m] = cd_itp(Qj[m])
    cl = np.where(np.isfinite(cl), cl, 0.0)
    cd = np.where(np.isfinite(cd), cd, 1e-3)
    cd = np.clip(cd, 1e-4, None)
    return cl, cd

def _evaluate_section(chord, twist_local, r, Vax, Vtan,
                      cl_itp, cd_itp, a_min, a_max, re_min, re_max, rho, mu):
    phi = np.arctan2(Vax, Vtan)
    alpha = twist_local - phi
    a_deg = np.degrees(alpha)
    Vrel = np.sqrt(Vax**2 + Vtan**2)
    Re = (rho * Vrel * chord) / mu
    cl, cd = _lookup_cl_cd(cl_itp, cd_itp, Re, a_deg, re_min, re_max, a_min, a_max)
    L  = 0.5 * rho * Vrel**2 * chord * cl
    Df = 0.5 * rho * Vrel**2 * chord * cd
    return L, Df, phi

def run_bemt_row(row, rpm, cfg, cl_itp, cd_itp, a_min, a_max, re_min, re_max,
                 r_hub, r_tip, span, n_blades, n_elems, rho, mu):
    r_edges = np.linspace(r_hub, r_tip, n_elems + 1)
    r = 0.5*(r_edges[:-1] + r_edges[1:]); dr = r_edges[1] - r_edges[0]

    AR = float(row["AR"]); lam = float(row["lambda"])
    aoa_root = float(row["aoaRoot (deg)"]); aoa_tip = float(row["aoaTip (deg)"])

    root_chord = (2.0 * span) / (AR * (1.0 + lam))
    tip_chord  = lam * root_chord
    chord = root_chord + (tip_chord - root_chord) * (r - r_hub) / span
    twist_local = np.deg2rad(aoa_root + (aoa_tip - aoa_root) * (r - r_hub) / span)

    omega = rpm * (2*np.pi/60.0)
    Vtan = omega * r

    v_i = np.zeros_like(r)
    relax = float(cfg["bemt"]["relax"])
    for _ in range(int(cfg["bemt"]["max_iters"])):
        L, Df, phi = _evaluate_section(chord, twist_local, r, v_i, Vtan,
                                       cl_itp, cd_itp, a_min, a_max, re_min, re_max, rho, mu)
        sin_phi = np.sin(np.clip(phi, 1e-6, None))
        F = (2/np.pi) * np.arccos(np.exp(-(n_blades/2.0) * (r_tip - r) / (r * sin_phi)))
        F = np.clip(F, 1e-3, 1.0)
        dT = (L * np.cos(phi) - Df * np.sin(phi)) * n_blades
        vi_new = dT / (4.0 * np.pi * r * rho * F + 1e-12)
        v_i = relax * v_i + (1 - relax) * np.clip(vi_new, 0.0, None)

    L, Df, phi = _evaluate_section(chord, twist_local, r, v_i, Vtan,
                                   cl_itp, cd_itp, a_min, a_max, re_min, re_max, rho, mu)
    sin_phi = np.sin(np.clip(phi, 1e-6, None)); cos_phi = np.cos(phi)
    dT = (L * cos_phi - Df * sin_phi) * n_blades
    dQ = (L * sin_phi + Df * cos_phi) * r * n_blades

    T = np.sum(dT * dr); Q_total = np.sum(dQ * dr)
    Q_profile = np.sum((Df * cos_phi) * r * n_blades * dr)

    if T <= 0.0:
        phi_fb = np.deg2rad(cfg["bemt"]["phi_fallback_deg"])
        Vax_fb = np.tan(phi_fb) * Vtan
        Lf, Dff, phif = _evaluate_section(chord, twist_local, r, Vax_fb, Vtan,
                                          cl_itp, cd_itp, a_min, a_max, re_min, re_max, rho, mu)
        sin_phif = np.sin(np.clip(phif, 1e-6, None)); cos_phif = np.cos(phif)
        dT_fb = (Lf * cos_phif - Dff * sin_phif) * n_blades
        dQ_fb = (Lf * sin_phif + Dff * cos_phif) * r * n_blades
        T_fb  = np.sum(dT_fb * dr); Q_fb = np.sum(dQ_fb * dr)
        if T_fb > 0: T, Q_total = T_fb, max(Q_fb, 0.0)

    Q = Q_total if Q_total > 0.0 else max(Q_profile, 0.0)
    T = max(T, 0.0)

    disk_A = np.pi * (r_tip**2)
    n_rev = rpm/60.0
    D = 2.0 * r_tip
    P_mech  = Q * max(omega, 1e-12)
    P_ideal = np.sqrt(T**3 / (2 * rho * disk_A)) if T > 0 else 0.0
    eta = float(np.clip(P_ideal / max(P_mech, 1e-12), 0.0, 1.0))

    CT = T / (rho * (n_rev**2) * (D**4) + 1e-12)
    CP = P_mech / (rho * (n_rev**3) * (D**5) + 1e-12)
    return eta, CT, CP

def main():
    C = load_cfg()

    # paths
    doe_path   = C["paths"]["doe_csv"]
    airfoil_csv = C["paths"]["airfoil_csv"]
    out_tables = C["paths"]["outputs_tables_dir"]
    os.makedirs(out_tables, exist_ok=True)

    # fluids & geometry
    rho = float(C["fluids"]["rho"]); mu = float(C["fluids"]["mu"])
    r_hub = float(C["geometry"]["r_hub_m"])
    span  = float(C["geometry"]["span_blade_m"])
    r_tip = r_hub + span
    n_blades = int(C["geometry"]["n_blades"])
    n_elems  = int(C["geometry"]["n_elems"])

    # rpm grid
    rpm_min = int(C["bemt"]["rpm_min"]); rpm_max = int(C["bemt"]["rpm_max"])
    rpm_step = int(C["bemt"]["rpm_step"]); bin_rpm = int(C["bemt"]["bin_rpm"])

    # DOE
    gcols = C["geometry_cols"]
    if not os.path.exists(doe_path):
        raise SystemExit(f"DOE CSV not found: {doe_path}")
    doe = pd.read_csv(doe_path)[["filename"] + gcols].drop_duplicates("filename")
    for c in gcols: doe[c] = pd.to_numeric(doe[c], errors="coerce")
    doe = doe.dropna(subset=gcols).reset_index(drop=True)

    # airfoil polars
    cl_itp, cd_itp, a_min, a_max, re_min, re_max = _build_airfoil_itp(airfoil_csv)
    rpm_list = np.arange(rpm_min, rpm_max + 1, rpm_step)

    rows=[]
    for idx, row in doe.iterrows():
        fn = row["filename"]
        for rpm in rpm_list:
            eta, CT, CP = run_bemt_row(row, rpm, C, cl_itp, cd_itp, a_min, a_max, re_min, re_max,
                                       r_hub, r_tip, span, n_blades, n_elems, rho, mu)
            rows.append({
                "filename": fn,
                "rpm": int(rpm),
                "rpm_bemt": int(_bin_nearest(rpm, bin_rpm)),
                "prop_eff_bemt": float(eta),
                "CT_bemt": float(CT),
                "CP_bemt": float(CP),
            })
        print(f"  - BEMT: {idx+1}/{len(doe)} {fn}")

    per_rpm = pd.DataFrame(rows)
    per_rpm.to_csv(os.path.join(out_tables, "bemt_predictions_final.csv"), index=False)

    avg = per_rpm.groupby("filename")["prop_eff_bemt"].mean().clip(0,1).reset_index()
    avg = avg.rename(columns={"prop_eff_bemt": "eta_bemt_mean"})
    avg.to_csv(os.path.join(out_tables, "bemt_avg_prior.csv"), index=False)
    print("Saved:\n  -", os.path.join(out_tables, "bemt_predictions_final.csv"),
          "\n  -", os.path.join(out_tables, "bemt_avg_prior.csv"))

if __name__ == "__main__":
    main()
