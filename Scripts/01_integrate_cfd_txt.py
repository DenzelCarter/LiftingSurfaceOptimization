# 01_integrate_cfd_txt.py
# Parse COMSOL .txt tables → geometry-keyed CFD master (no DOE dependency).
# Writes:
#   Experiment/outputs/tables/cfd_master.csv
#   Experiment/outputs/tables/cfd_master_vtol.csv
#   Experiment/outputs/tables/cfd_master_cruise.csv

import os, re, glob, numpy as np, pandas as pd
from path_utils import load_cfg

# ------------------------
# Robust parsing utilities
# ------------------------

_FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def _extract_floats(line: str):
    """Return all numeric tokens found in line as floats."""
    return [float(m.group(0)) for m in _FLOAT.finditer(line)]

def _split_header_by_gaps(line: str):
    """
    Split header into columns using runs of >=2 whitespace as separators.
    (Do NOT split on commas — commas belong to the label text.)
    """
    raw = line.rstrip()
    # Normalize tabs to spaces, then split on runs of >=2 spaces
    raw = raw.replace("\t", "    ")
    cols = re.split(r"\s{2,}", raw)
    # Trim and drop empties
    cols = [c.strip() for c in cols if c.strip()]
    return cols

def _norm(s: str) -> str:
    """Normalize header label: lowercase and strip non-alphanumerics."""
    return re.sub(r'[^a-z0-9]+', '', s.lower()) if s is not None else ''

def _find_col_exact(cols, names):
    """Find column whose normalized header exactly matches any normalized name."""
    if cols is None: return None
    want = {_norm(n) for n in names}
    for i, c in enumerate(cols):
        if _norm(c) in want:
            return i
    return None

def _find_col_tokens(cols, token_groups):
    """
    Find column whose normalized header contains all tokens in any provided group.
    Example: token_groups=[["rot","1s"],["rotation","1s"]] matches 'rot (1/s)'.
    """
    if cols is None: return None
    lc = [_norm(c) for c in cols]
    for i, c in enumerate(lc):
        for group in token_groups:
            if all(_norm(t) in c for t in group):
                return i
    return None

# ------------------------
# Table reader
# ------------------------

def _read_table(fp):
    """
    Read a COMSOL .txt export:
    - Detect header (first non-comment line containing both 'AR' and 'lambda', case-insensitive)
    - Parse numeric rows (lines with >=5 floats)
    Returns (cols:list[str] or None, data: np.ndarray[N, M] of floats).
    """
    cols = None
    rows = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            if cols is None and ("ar" in line.lower() and "lambda" in line.lower()):
                cols = _split_header_by_gaps(line)
                continue
            vals = _extract_floats(line)
            if len(vals) >= 5:
                rows.append(vals)
    data = np.array(rows, float) if rows else np.zeros((0, 0), float)
    # quick sanity ping
    if cols is not None and data.size:
        print(f"[{os.path.basename(fp)}] header_cols={len(cols)} | first_row_vals={len(rows[0])}")
    return cols, data

# ------------------------
# Physics helpers
# ------------------------

def _eta_from_row(vals, cols, rho, disk_A):
    """Prop efficiency from direct column, else Pi/Pm, else (T,Q,rot)."""
    # Direct efficiency column (either form)
    i = _find_col_tokens(cols, [["propeller","efficiency"]])
    if i is not None and i < len(vals):
        v = float(vals[i])
        if np.isfinite(v):
            return float(np.clip(v, 0.0, 1.0))

    # Ideal/Mechanical power ratio
    ip = _find_col_tokens(cols, [["ideal","power"]])
    mp = _find_col_tokens(cols, [["mechanical","power"]])
    if None not in (ip, mp) and ip < len(vals) and mp < len(vals):
        Pi, Pm = float(vals[ip]), float(vals[mp])
        if np.isfinite(Pi) and np.isfinite(Pm) and Pm > 1e-12:
            return float(np.clip(Pi / Pm, 0.0, 1.0))

    # Thrust/Torque/rot fallback
    iT = _find_col_tokens(cols, [["thrust"]])
    iQ = _find_col_tokens(cols, [["torque"]])
    iR = _find_col_tokens(cols, [["rot","1s"], ["rotation","1s"]])
    if None not in (iT, iQ, iR) and max(iT, iQ, iR) < len(vals):
        T = float(vals[iT]); Q = float(vals[iQ]); rot = float(vals[iR])
        Pm = Q * (2.0 * np.pi * rot)            # rot is 1/s
        Pi = np.sqrt(max(T, 0.0)**3 / (2.0 * rho * disk_A)) if T > 0 else 0.0
        if Pm > 1e-12:
            return float(np.clip(Pi / Pm, 0.0, 1.0))

    return np.nan

def _ld_from_row(vals, cols):
    """Cruise L/D from dedicated column (several aliases)."""
    i = _find_col_tokens(cols, [["open","air","efficiency"],
                                ["l","d"],
                                ["lift","drag"],
                                ["lifttodrag"]])
    if i is not None and i < len(vals):
        v = float(vals[i])
        return v if np.isfinite(v) else np.nan
    return np.nan

def _geom_from(cols, data):
    """Extract geometry columns with unit sniffing for AOA (rad→deg)."""
    iAR  = _find_col_exact(cols, ["AR", "Aspect Ratio"])
    iLam = _find_col_exact(cols, ["lambda", "taper ratio", "taper"])
    iR   = _find_col_tokens(cols, [["aoa","root"], ["root","aoa"]])
    iT   = _find_col_tokens(cols, [["aoa","tip"],  ["tip","aoa"]])
    if None in (iAR, iLam, iR, iT):
        # best-effort fallback if header matching fails
        if data.shape[1] >= 4:
            iAR, iLam, iR, iT = 0, 1, 2, 3
        else:
            return None

    AR  = data[:, iAR]
    lam = data[:, iLam]

    def _is_rad(idx):
        name = (cols[idx] or "").lower()
        return "(rad" in name

    aoaR = data[:, iR]
    aoaT = data[:, iT]
    if _is_rad(iR): aoaR = np.degrees(aoaR)
    if _is_rad(iT): aoaT = np.degrees(aoaT)

    return pd.DataFrame({
        "AR": AR,
        "lambda": lam,
        "aoaRoot (deg)": aoaR,
        "aoaTip (deg)": aoaT
    })

# ------------------------
# Main
# ------------------------

def main():
    C = load_cfg()
    out_dir = C["paths"]["outputs_tables_dir"]; os.makedirs(out_dir, exist_ok=True)

    # Disk area for ideal power
    rho   = float(C["fluids"]["rho"])
    r_hub = float(C["geometry"]["r_hub_m"])
    span  = float(C["geometry"]["span_blade_m"])
    r_tip = r_hub + span
    disk_A = np.pi * (r_tip**2)

    vt_glob = C["paths"]["comsol_vtol_glob"]
    cr_glob = C["paths"]["comsol_cruise_glob"]

    # ---- VTOL files
    vt_frames = []
    for fp in sorted(glob.glob(vt_glob)):
        cols, data = _read_table(fp)
        if data.size == 0:
            print(f"[VTOL] No numeric rows in {os.path.basename(fp)}")
            continue
        geom = _geom_from(cols, data)
        if geom is None:
            print(f"[VTOL] Missing geometry cols in {os.path.basename(fp)}")
            continue

        # rotation rate (1/s); also compute rpm = 60*rot
        i_rot = _find_col_tokens(cols, [["rot","1s"], ["rotation","1s"]])
        rot_rps = data[:, i_rot] if (i_rot is not None and i_rot < data.shape[1]) else np.full(data.shape[0], np.nan)
        rpm = rot_rps * 60.0

        # prop efficiency
        eta = np.array([_eta_from_row(row, cols, rho, disk_A) for row in data], float)

        df = geom.copy()
        df["rot (1/s)"] = rot_rps
        df["rpm"]       = rpm
        df["eta_cfd"]   = eta
        df["mode"]      = "vtol"
        vt_frames.append(df)

    vt_all = pd.concat(vt_frames, ignore_index=True) if vt_frames else pd.DataFrame(
        columns=["AR","lambda","aoaRoot (deg)","aoaTip (deg)","rot (1/s)","rpm","eta_cfd","mode"]
    )

    # ---- CRUISE files
    cr_frames = []
    for fp in sorted(glob.glob(cr_glob)):
        cols, data = _read_table(fp)
        if data.size == 0:
            print(f"[CRUISE] No numeric rows in {os.path.basename(fp)}")
            continue
        geom = _geom_from(cols, data)
        if geom is None:
            print(f"[CRUISE] Missing geometry cols in {os.path.basename(fp)}")
            continue

        LD = np.array([_ld_from_row(row, cols) for row in data], float)

        df = geom.copy()
        df["rot (1/s)"] = np.nan
        df["rpm"]       = np.nan
        df["LD_cfd"]    = LD
        df["mode"]      = "cruise"
        cr_frames.append(df)

    cr_all = pd.concat(cr_frames, ignore_index=True) if cr_frames else pd.DataFrame(
        columns=["AR","lambda","aoaRoot (deg)","aoaTip (deg)","rot (1/s)","rpm","LD_cfd","mode"]
    )

    # ---- Write outputs
    master = pd.concat([vt_all, cr_all], ignore_index=True)
    master.to_csv(os.path.join(out_dir, "cfd_master.csv"), index=False)
    if not vt_all.empty: vt_all.to_csv(os.path.join(out_dir, "cfd_master_vtol.csv"), index=False)
    if not cr_all.empty: cr_all.to_csv(os.path.join(out_dir, "cfd_master_cruise.csv"), index=False)

    print(f"CFD rows → total={len(master)} | vtol={len(vt_all)} | cruise={len(cr_all)}")
    print("Wrote:")
    print("  -", os.path.join(out_dir, "cfd_master.csv"))
    if not vt_all.empty: print("  -", os.path.join(out_dir, "cfd_master_vtol.csv"))
    if not cr_all.empty: print("  -", os.path.join(out_dir, "cfd_master_cruise.csv"))

if __name__ == "__main__":
    main()
