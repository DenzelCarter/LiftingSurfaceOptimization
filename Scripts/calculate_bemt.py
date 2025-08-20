# calculate_bemt.py
# Static BEMT over DOE with rpm 350..4000 (step 50).
# Fast (reuses LinearNDInterpolator), robust binning, and guards:
# - torque fallback (profile torque) if total torque <= 0
# - thrust fallback (small axial inflow) if T <= 0
# - CLAMP prop_eff_bemt to [0, 1]
# Writes: Experiment/tools/bemt_predictions_final.csv (with rpm_bemt for 150-RPM bins)

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# ---------------- paths (relative to this file) ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

DOE_01 = os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv')
DOE_02 = os.path.join(TOOLS_DIR, 'doe_test_plan_02.csv')
AIRFOIL_DATA_PATH = os.path.join(TOOLS_DIR, 'naca0012_all_re.csv')
OUTPUT_FILE_PATH  = os.path.join(TOOLS_DIR, 'bemt_predictions_final.csv')

# ---------------- constants ----------------
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

RPM_MIN = 350
RPM_MAX = 4000
RPM_STEP = 50

BIN_RPM = 150  # must match process_data.py
PHI_FALLBACK_DEG = 2.5  # small axial inflow if solver yields T <= 0

# ---------------- binning ----------------
def bin_nearest(val, w):
    v = np.asarray(val, dtype=float)
    b = np.floor((v + 0.5 * w) / w) * w
    return b.astype(int)

# ---------------- airfoil interpolation ----------------
def build_airfoil_interpolators(path):
    df = pd.read_csv(path)
    pts = df[['reynolds', 'alpha']].to_numpy()
    clv = df['cl'].to_numpy()
    cdv = df['cd'].to_numpy()
    a_min, a_max = df['alpha'].min(), df['alpha'].max()
    re_min, re_max = df['reynolds'].min(), df['reynolds'].max()
    cl_itp = LinearNDInterpolator(pts, clv, fill_value=np.nan)
    cd_itp = LinearNDInterpolator(pts, cdv, fill_value=np.nan)
    print(f"Airfoil data loaded. Alpha range: [{a_min:.2f}, {a_max:.2f}] deg | Re range: [{re_min:.2e}, {re_max:.2e}]")
    return cl_itp, cd_itp, a_min, a_max, re_min, re_max

def lookup_cl_cd(cl_itp, cd_itp, Re, alpha_deg, re_min, re_max, a_min, a_max):
    Re_clp = np.clip(Re, re_min, re_max)
    a_clp  = np.clip(alpha_deg, a_min, a_max)
    Q = np.column_stack([Re_clp, a_clp])
    cl = cl_itp(Q)
    cd = cd_itp(Q)
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

# ---------------- section forces ----------------
def evaluate_section_forces(chord, twist_local, r, Vax, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max):
    phi = np.arctan2(Vax, Vtan)
    alpha = twist_local - phi
    a_deg = np.degrees(alpha)
    Vrel = np.sqrt(Vax**2 + Vtan**2)
    Re = (RHO * Vrel * chord) / MU_AIR
    cl, cd = lookup_cl_cd(cl_itp, cd_itp, Re, a_deg, re_min, re_max, a_min, a_max)
    L  = 0.5 * RHO * Vrel**2 * chord * cl
    Df = 0.5 * RHO * Vrel**2 * chord * cd
    return L, Df, phi, Vrel

# ---------------- BEMT ----------------
def run_bemt_static(prop_row, rpm, cl_itp, cd_itp, a_min, a_max, re_min, re_max):
    r_edges = np.linspace(R_HUB, R_TIP, NUM_ELEMENTS + 1)
    r = 0.5*(r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    AR = float(prop_row['AR'])
    lam = float(prop_row['lambda'])
    aoa_root = float(prop_row['aoaRoot (deg)'])
    aoa_tip  = float(prop_row['aoaTip (deg)'])

    root_chord = (2.0 * SPAN_BLADE) / (AR * (1.0 + lam))
    tip_chord  = lam * root_chord
    chord = root_chord + (tip_chord - root_chord) * (r - R_HUB) / SPAN_BLADE
    twist_local = np.deg2rad(aoa_root + (aoa_tip - aoa_root) * (r - R_HUB) / SPAN_BLADE)

    omega = rpm * (2*np.pi/60.0)
    Vtan = omega * r

    v_i = np.zeros_like(r)
    for _ in range(MAX_ITERS):
        L, Df, phi, Vrel = evaluate_section_forces(chord, twist_local, r, v_i, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
        sin_phi = np.sin(np.clip(phi, 1e-6, None))
        F = (2/np.pi) * np.arccos(np.exp(-(NUM_BLADES/2.0) * (R_TIP - r) / (r * sin_phi)))
        F = np.clip(F, 1e-3, 1.0)
        dT = (L * np.cos(phi) - Df * np.sin(phi)) * NUM_BLADES
        vi_new = dT / (4.0 * np.pi * r * RHO * F + 1e-12)
        v_i = RELAX*v_i + (1-RELAX)*np.clip(vi_new, 0.0, None)

    # final
    L, Df, phi, Vrel = evaluate_section_forces(chord, twist_local, r, v_i, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
    sin_phi = np.sin(np.clip(phi, 1e-6, None))
    cos_phi = np.cos(phi)

    dT = (L * cos_phi - Df * sin_phi) * NUM_BLADES
    dQ = (L * sin_phi + Df * cos_phi) * r * NUM_BLADES

    T = np.sum(dT * dr)
    Q_total = np.sum(dQ * dr)
    Q_profile = np.sum((Df * cos_phi) * r * NUM_BLADES * dr)

    # thrust fallback if needed
    if T <= 0.0:
        phi_fb = np.deg2rad(PHI_FALLBACK_DEG)
        Vax_fb = np.tan(phi_fb) * Vtan
        Lf, Dff, phif, _ = evaluate_section_forces(chord, twist_local, r, Vax_fb, Vtan, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
        sin_phif = np.sin(np.clip(phif, 1e-6, None))
        cos_phif = np.cos(phif)
        dT_fb = (Lf * cos_phif - Dff * sin_phif) * NUM_BLADES
        dQ_fb = (Lf * sin_phif + Dff * cos_phif) * r * NUM_BLADES
        T_fb  = np.sum(dT_fb * dr)
        Q_fb  = np.sum(dQ_fb * dr)
        if T_fb > 0:
            T = T_fb
            Q_total = max(Q_fb, 0.0)

    Q = Q_total if Q_total > 0.0 else max(Q_profile, 0.0)
    T = max(T, 0.0)

    omega = max(omega, 1e-9)
    P_mech = Q * omega
    P_ideal = np.sqrt(T**3 / (2 * RHO * A)) if T > 0 else 0.0
    eta = (P_ideal / P_mech) if P_mech > 1e-12 else 0.0
    # clamp efficiency to [0, 1]
    eta = float(np.clip(eta, 0.0, 1.0))

    n = rpm / 60.0
    CT = T / (RHO * (n**2) * (D**4) + 1e-12)
    CP = P_mech / (RHO * (n**3) * (D**5) + 1e-12)

    # 75% diagnostics
    r75 = R_HUB + 0.75*SPAN_BLADE
    i75 = np.argmin(np.abs(r - r75))
    lam75 = (v_i[i75] / (omega * r[i75])) if omega*r[i75] > 1e-12 else 0.0
    phi75 = float(np.degrees(phi[i75]))

    return T, Q, eta, CT, CP, lam75, phi75

def main():
    frames = []
    if os.path.exists(DOE_01): frames.append(pd.read_csv(DOE_01))
    if os.path.exists(DOE_02): frames.append(pd.read_csv(DOE_02))
    if not frames:
        raise SystemExit("DOE files not found.")
    doe = pd.concat(frames, ignore_index=True)
    print(f"Loaded DOE rows: {len(doe)}")

    cl_itp, cd_itp, a_min, a_max, re_min, re_max = build_airfoil_interpolators(AIRFOIL_DATA_PATH)
    rpm_list = np.arange(RPM_MIN, RPM_MAX+1, RPM_STEP)

    rows = []
    print("\nRunning BEMT across DOE × RPM grid...")
    for idx, row in doe.iterrows():
        fname = row['filename']
        for rpm in rpm_list:
            T, Q, eta, CT, CP, lam75, phi75 = run_bemt_static(row, rpm, cl_itp, cd_itp, a_min, a_max, re_min, re_max)
            rows.append({
                'filename': fname,
                'rpm': int(rpm),
                'rpm_bemt': int(bin_nearest(rpm, BIN_RPM)),
                'prop_eff_bemt': eta,      # already clamped to [0,1]
                'CT_bemt': float(CT),
                'CP_bemt': float(CP),
                'lambda75_bemt': float(lam75),
                'phi75_bemt': float(phi75),
            })
        print(f"  - Finished {idx+1}/{len(doe)}: {fname}")

    out = pd.DataFrame(rows)
    os.makedirs(TOOLS_DIR, exist_ok=True)
    out.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nBEMT predictions saved to '{OUTPUT_FILE_PATH}'")

if __name__ == "__main__":
    main()
