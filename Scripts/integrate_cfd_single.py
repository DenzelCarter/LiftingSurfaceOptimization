# integrate_cfd_single.py
# Ingest COMSOL single-RPM results (CSV or COMSOL text table) and emit a per-geometry prior:
#   tools/cfd_single_prior.csv with columns:
#   ['AR','lambda','aoaRoot (deg)','aoaTip (deg)','rpm_cfd','eta_cfd_single']
#
# - Handles COMSOL .txt table format (lines beginning with '%' are comments).
# - Splits the wide header by 2+ spaces, preserves labels like "Propeller Efficiency (1), Propeller Efficiency".
# - Converts aoa_root/aoa_tip from radians to degrees.
# - Converts rot (1/s) to rpm (×60).
# - Rounds geometry to 4 decimals for robust joins with your datasets.
# - If multiple rows per geometry exist, keeps the first; adjust as needed.

import os, re, numpy as np, pandas as pd

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(THIS_DIR)
COMS_DIR = os.path.join(PROJ_ROOT, 'Experiment', 'comsol')
TOOLS_DIR = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

# >>>> EDIT THESE <<<<
INPUTS = [
    os.path.join(COMS_DIR, 'comsol_results_01.txt'),  # your pasted .txt
    os.path.join(COMS_DIR, 'comsol_results_02.txt'),
    os.path.join(COMS_DIR, 'comsol_results_03.txt')# you can add CSVs or more TXTs here
]
OUT = os.path.join(TOOLS_DIR, 'cfd_single_prior.csv')

KEY_COLS = ['AR','lambda','aoaRoot (deg)','aoaTip (deg)']

# ---------- parsing helpers ----------
_SPLIT_RE = re.compile(r"\s{2,}")  # split on 2+ spaces

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _find_col(headers, candidates, contains_ok=False):
    """Return index of the first header matching any candidate (by normalized string).
       If contains_ok=True, allow candidate to be a substring of header."""
    Hn = [_normalize(h) for h in headers]
    Cn = [_normalize(c) for c in candidates]
    # exact
    for ci in Cn:
        for i, hn in enumerate(Hn):
            if hn == ci:
                return i
    if contains_ok:
        for ci in Cn:
            for i, hn in enumerate(Hn):
                if ci in hn:
                    return i
    return None

def _read_comsol_txt(path):
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.strip() == "" or raw.lstrip().startswith("%"):
                continue
            lines.append(raw.rstrip("\n"))
    if not lines:
        raise SystemExit(f"No data rows in {path}")

    # header is first non-comment line
    header_line = lines[0].strip()
    headers = _SPLIT_RE.split(header_line)
    data_rows = lines[1:]

    # Build a table by splitting each data line on 2+ spaces.
    rows = []
    for ln in data_rows:
        parts = _SPLIT_RE.split(ln.strip())
        # pad/truncate to header length
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            parts = parts[:len(headers)]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=headers)

    # Try to locate required columns (robust to mild header variations)
    idx_AR = _find_col(headers, ["AR"])
    idx_L  = _find_col(headers, ["lambda"])
    idx_aR = _find_col(headers, ["aoa_root (rad)","aoa_root(rad)","aoa root (rad)"], contains_ok=True)
    idx_aT = _find_col(headers, ["aoa_tip (rad)","aoa_tip(rad)","aoa tip (rad)"], contains_ok=True)
    idx_rot= _find_col(headers, ["rot (1/s)","rotation (1/s)","rot(1/s)","rot 1/s"], contains_ok=True)
    # Prefer the explicit Propeller Efficiency column; fall back to "Open Air Efficiency" if needed.
    idx_eta= _find_col(headers, ["Propeller Efficiency (1), Propeller Efficiency","Propeller Efficiency"], contains_ok=True)

    need = [("AR", idx_AR), ("lambda", idx_L), ("aoa_root (rad)", idx_aR),
            ("aoa_tip (rad)", idx_aT), ("rot (1/s)", idx_rot), ("prop_eff", idx_eta)]
    missing = [name for name, idx in need if idx is None]
    if missing:
        raise SystemExit(f"{path}: missing required columns: {missing}")

    # Pull and convert
    def col(idx): return pd.to_numeric(df.iloc[:, idx], errors="coerce")

    AR  = col(idx_AR)
    lam = col(idx_L)
    aoa_root_deg = np.rad2deg(col(idx_aR))
    aoa_tip_deg  = np.rad2deg(col(idx_aT))
    rpm = col(idx_rot) * 60.0
    eta = pd.to_numeric(df.iloc[:, idx_eta], errors="coerce").clip(0,1)

    out = pd.DataFrame({
        "AR": AR, "lambda": lam,
        "aoaRoot (deg)": aoa_root_deg,
        "aoaTip (deg)":  aoa_tip_deg,
        "rpm_cfd": rpm,
        "eta_cfd_single": eta,
    })
    return out

def _read_csv_generic(path):
    d = pd.read_csv(path)
    # Alias columns if needed
    # Try to locate these logical fields:
    def pick(colnames, fallback=None):
        for c in colnames:
            if c in d.columns: return d[c]
        return d[fallback] if fallback and fallback in d.columns else np.nan

    AR  = pd.to_numeric(pick(['AR','ar','Ar']), errors="coerce")
    lam = pd.to_numeric(pick(['lambda','taper','Lambda']), errors="coerce")
    aR  = pick(['aoaRoot (deg)','aoa_root (deg)','root_aoa_deg'])
    aT  = pick(['aoaTip (deg)','aoa_tip (deg)','tip_aoa_deg'])
    # If radians were accidentally provided, allow detection by magnitude > ~1.5 rad
    aR = pd.to_numeric(aR, errors="coerce"); aT = pd.to_numeric(aT, errors="coerce")
    if np.nanmax(np.abs(aR)) > 1.5: aR = np.rad2deg(aR)
    if np.nanmax(np.abs(aT)) > 1.5: aT = np.rad2deg(aT)

    rpm = pd.to_numeric(pick(['rpm','RPM','rot (1/s)']), errors="coerce")
    if 'rot (1/s)' in d.columns and (rpm.isna() | (rpm==0)).all():
        rpm = pd.to_numeric(d['rot (1/s)'], errors="coerce") * 60.0

    eta = pd.to_numeric(pick(['eta_cfd_single','prop_efficiency','Propeller Efficiency','Open Air Efficiency']), errors="coerce").clip(0,1)

    out = pd.DataFrame({
        "AR": AR, "lambda": lam,
        "aoaRoot (deg)": aR, "aoaTip (deg)": aT,
        "rpm_cfd": rpm, "eta_cfd_single": eta
    })
    return out

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".dat", ".tbl"):
        return _read_comsol_txt(path)
    elif ext in (".csv",):
        return _read_csv_generic(path)
    else:
        # try txt parser first, then csv
        try:
            return _read_comsol_txt(path)
        except Exception:
            return _read_csv_generic(path)

def main():
    frames = []
    for p in INPUTS:
        if not os.path.exists(p):
            print(f"Warning: missing {p}")
            continue
        try:
            frames.append(load_any(p))
            print(f"Loaded: {p}")
        except SystemExit as e:
            raise
        except Exception as e:
            raise SystemExit(f"Failed to parse {p}: {e}")
    if not frames:
        raise SystemExit("No inputs parsed.")

    cfd = pd.concat(frames, ignore_index=True)

    # Clean up types and drop incomplete rows
    for c in ["AR","lambda","aoaRoot (deg)","aoaTip (deg)","rpm_cfd","eta_cfd_single"]:
        cfd[c] = pd.to_numeric(cfd[c], errors="coerce")
    cfd = cfd.dropna(subset=["AR","lambda","aoaRoot (deg)","aoaTip (deg)","eta_cfd_single"]).copy()
    cfd["eta_cfd_single"] = cfd["eta_cfd_single"].clip(0,1)

    # Round geometry for stable joins
    for c in ["AR","lambda","aoaRoot (deg)","aoaTip (deg)"]:
        cfd[c] = cfd[c].round(4)

    # If the same geometry appears multiple times, keep the first occurrence
    cfd = (cfd.sort_values(["AR","lambda","aoaRoot (deg)","aoaTip (deg)"])
               .drop_duplicates(subset=["AR","lambda","aoaRoot (deg)","aoaTip (deg)"], keep="first")
               .reset_index(drop=True))

    os.makedirs(TOOLS_DIR, exist_ok=True)
    cfd.to_csv(OUT, index=False)
    print(f"Wrote: {OUT} (rows={len(cfd)})")

if __name__ == "__main__":
    main()
