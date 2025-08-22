# Scripts/integrate_cfd_txt.py
# Robust COMSOL .txt ingestion (VTOL & Cruise), surrogate fitting, and DOE-aligned priors.
# Reads paths from Scripts/config.yaml via path_utils.load_cfg()

import os, glob, re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from path_utils import load_cfg

NUMERIC_RE = re.compile(r"^[\s\t]*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[\s\t]*$")

def _is_numeric_token(tok: str) -> bool:
    return bool(NUMERIC_RE.match(tok))

def _parse_comsol_txt(path: str, min_cols: int = 6, replace_commas: bool = True) -> pd.DataFrame | None:
    """
    Parse a COMSOL-exported whitespace-ish table.
    - Skips lines starting with '%' or '#'
    - Optionally replaces ',' with ' ' to kill header artifacts like "Thrust (N), Thrust"
    - Keeps only lines with >= min_cols numeric tokens
    Returns a float DataFrame of shape (n_rows, n_cols) or None.
    """
    rows = []
    max_cols = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("%") or s.startswith("#"):
                continue
            if replace_commas:
                s = s.replace(",", " ")
            toks = [t for t in s.split() if t]  # split on any whitespace
            if not toks:
                continue
            # Keep only numeric tokens; if too few, skip the line
            num_toks = []
            for t in toks:
                if _is_numeric_token(t):
                    num_toks.append(t)
                else:
                    # try a last-chance float conversion (e.g., weird unicode minus)
                    try:
                        float(t)
                        num_toks.append(t)
                    except:
                        pass
            if len(num_toks) < min_cols:
                continue
            max_cols = max(max_cols, len(num_toks))
            rows.append(num_toks)

    if not rows:
        return None

    # Normalize all rows to the same column count by truncating to max_cols
    rows = [r[:max_cols] for r in rows]
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    # Drop rows that aren't fully numeric after coercion
    df = df.dropna(how="any")
    if df.shape[1] < min_cols or df.shape[0] == 0:
        return None
    return df

def _fit_surrogate(X: np.ndarray, y: np.ndarray):
    n = X.shape[0]
    cvk = min(5, max(3, n)) if n > 2 else 3
    model = Pipeline([
        ("scl", StandardScaler()),
        ("rid", RidgeCV(alphas=np.logspace(-6, 3, 40), cv=cvk, scoring="neg_mean_absolute_error"))
    ])
    model.fit(X, y)
    return model

def _load_doe(C):
    p = C["paths"]["doe_csv"]
    g = C["geometry_cols"]
    df = pd.read_csv(p)[["filename"] + g].drop_duplicates("filename")
    for c in g:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=g).reset_index(drop=True)

def _ingest_folder(glob_pat: str, label: str, need_rows: int = 5) -> list[pd.DataFrame]:
    files = sorted(glob.glob(glob_pat))
    print(f"[{label}] glob: {glob_pat}")
    print(f"[{label}] matched files: {len(files)}")
    if files[:3]:
        for fp in files[:3]:
            print(f"  - {fp}")
    dfs = []
    for fp in files:
        df = _parse_comsol_txt(fp, min_cols=6, replace_commas=True)
        if df is None:
            print(f"  (skip) {fp}: no numeric table found")
            continue
        print(f"  parsed {fp}: {df.shape[0]} rows × {df.shape[1]} cols")
        dfs.append(df)
    if not dfs:
        return []
    # Require at least a few rows total
    total_rows = sum(d.shape[0] for d in dfs)
    print(f"[{label}] total parsed rows: {total_rows}")
    if total_rows < need_rows:
        print(f"[{label}] not enough rows (need ≥{need_rows})")
        return []
    return dfs

def main():
    C = load_cfg()
    tools = C["paths"]["tools_dir"]; os.makedirs(tools, exist_ok=True)
    gcols = C["geometry_cols"]
    doe = _load_doe(C)

    # ---------- VTOL ingest ----------
    vtol_glob = C["paths"]["comsol_vtol_glob"]
    vtol_tables = _ingest_folder(vtol_glob, "VTOL", need_rows=5)

    if vtol_tables:
        vt = pd.concat(vtol_tables, ignore_index=True)
        # Interpret: first 4 numeric cols = AR, lambda, aoa_root(rad), aoa_tip(rad)
        # last 2 numeric cols = Thrust(N), Power(W)
        geom = pd.DataFrame({
            "AR": vt.iloc[:, 0].astype(float),
            "lambda": vt.iloc[:, 1].astype(float),
            "aoaRoot (deg)": np.rad2deg(vt.iloc[:, 2].astype(float)),
            "aoaTip (deg)":  np.rad2deg(vt.iloc[:, 3].astype(float)),
        })
        thrust = vt.iloc[:, -2].astype(float)
        power  = vt.iloc[:, -1].astype(float)

        df_v = geom.assign(eta_cfd_vtol = (thrust / np.maximum(power, 1e-12)).clip(0, 1))
        # Basic sanity filters
        df_v = df_v.replace([np.inf, -np.inf], np.nan).dropna()
        before = len(df_v)
        df_v = df_v[(df_v["AR"] > 0) & (df_v["lambda"] > 0)]
        print(f"[VTOL] usable rows after filters: {len(df_v)} (from {before})")

        if len(df_v) >= 5:
            Xtr = df_v[gcols].to_numpy(float)
            ytr = df_v["eta_cfd_vtol"].to_numpy(float)
            model_v = _fit_surrogate(Xtr, ytr)
            pred = model_v.predict(doe[gcols].to_numpy(float))
            out = pd.DataFrame({"filename": doe["filename"], "eta_cfd_vtol": np.clip(pred, 0, 1)})
            out.to_csv(os.path.join(tools, "cfd_prior_vtol.csv"), index=False)
            print(f"[VTOL] wrote {os.path.join(tools, 'cfd_prior_vtol.csv')}")
        else:
            print("[VTOL] No usable VTOL rows (need ≥5). Skipping prior.")
    else:
        print("[VTOL] No usable VTOL files parsed. Skipping.")

    # ---------- Cruise ingest ----------
    cruise_glob = C["paths"]["comsol_cruise_glob"]
    cruise_tables = _ingest_folder(cruise_glob, "CRUISE", need_rows=5)

    if cruise_tables:
        cr = pd.concat(cruise_tables, ignore_index=True)
        # Interpret: first 4 numeric cols = AR, lambda, aoa_root(rad), aoa_tip(rad)
        # last 2 numeric cols = Lift(N), Drag(N)
        geom = pd.DataFrame({
            "AR": cr.iloc[:, 0].astype(float),
            "lambda": cr.iloc[:, 1].astype(float),
            "aoaRoot (deg)": np.rad2deg(cr.iloc[:, 2].astype(float)),
            "aoaTip (deg)":  np.rad2deg(cr.iloc[:, 3].astype(float)),
        })
        lift = cr.iloc[:, -2].astype(float)
        drag = cr.iloc[:, -1].astype(float)

        df_c = geom.assign(LD_cfd = (lift / np.maximum(drag, 1e-12)))
        df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna()
        before = len(df_c)
        df_c = df_c[(df_c["AR"] > 0) & (df_c["lambda"] > 0)]
        print(f"[CRUISE] usable rows after filters: {len(df_c)} (from {before})")

        if len(df_c) >= 5:
            Xtr = df_c[gcols].to_numpy(float)
            ytr = df_c["LD_cfd"].to_numpy(float)
            model_c = _fit_surrogate(Xtr, ytr)
            pred = model_c.predict(doe[gcols].to_numpy(float))
            out = pd.DataFrame({"filename": doe["filename"], "LD_cfd": pred})
            out.to_csv(os.path.join(tools, "cfd_prior_cruise.csv"), index=False)
            print(f"[CRUISE] wrote {os.path.join(tools, 'cfd_prior_cruise.csv')}")
        else:
            print("[CRUISE] No usable cruise rows (need ≥5). Skipping prior.")
    else:
        print("[CRUISE] No usable cruise files parsed. Skipping.")

if __name__ == "__main__":
    main()
