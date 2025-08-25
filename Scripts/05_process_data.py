# 05_process_data.py
# Build master_dataset.parquet from bench/tunnel CSVs (recursive).
# New: optional TARE correction using a lookup CSV (rpm_bin, thrust_tare, torque_tare).
# - If process.use_tare: true and a tare CSV is found, subtract T/PM tare vs RPM (linear interp).
# - Fallback: if no tare CSV but bench "tare" files exist, compute a constant tare (window-avg).
# - Robust filename join with DOE via canonical key (strip ext, case-insensitive).
# - Hover η computed as Pi(T_corr)/Pm_corr over rpm_window (optionally split into N bins).

import os, glob, re, numpy as np, pandas as pd
from path_utils import load_cfg

GEO = ["AR","lambda","aoaRoot (deg)","aoaTip (deg)"]
_TARE_PAT = re.compile(r"^(tare|mount|baseline)\b", re.IGNORECASE)

def _canon_name(s: str) -> str:
    b = os.path.basename(str(s)).strip()
    b = os.path.splitext(b)[0]
    b = re.sub(r"\s+", "_", b)
    return b.lower()

def _sanitize_aoa_deg(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum()==0: return s
    v = s.copy()
    if np.nanmedian(np.abs(v)) > 180:
        v1 = v/100.0
        if np.nanmedian(np.abs(v1)) < 90: v = v1
        else:
            v2 = np.mod(v, 360.0); v2[v2>180] -= 360.0; v = v2
    return v.clip(lower=-5.0, upper=45.0)

def _collect_csvs(root_dir):
    if not os.path.exists(root_dir): return []
    return sorted(set(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)))

def _infer_filename_from_path(fp: str) -> str | None:
    base = os.path.splitext(os.path.basename(fp))[0]
    if _TARE_PAT.match(base): return None
    return base

def _find_col(df: pd.DataFrame, preferred, fallbacks=()):
    cols = list(df.columns); low=[c.lower() for c in cols]
    def _lookup(cands):
        for c in cands:
            c_low=c.lower()
            for i,n in enumerate(low):
                if c_low==n or c_low in n: return cols[i]
        return None
    return _lookup(preferred) or _lookup(fallbacks)

# -------- bench column resolvers (your schema) ----------
def _resolve_rpm(df: pd.DataFrame):
    rpm_opt = _find_col(df, ["Motor Optical Speed (RPM)"])
    rpm_ele = _find_col(df, ["Motor Electrical Speed (RPM)"])
    if rpm_opt: return pd.to_numeric(df[rpm_opt], errors="coerce")
    if rpm_ele: return pd.to_numeric(df[rpm_ele], errors="coerce")
    rpm_col = _find_col(df, ["rpm"])
    if rpm_col: return pd.to_numeric(df[rpm_col], errors="coerce")
    rot_col = _find_col(df, ["rot (1/s)","rotation (1/s)","rps"])
    if rot_col: return pd.to_numeric(df[rot_col], errors="coerce")*60.0
    return pd.Series(np.nan, index=df.index)

def _resolve_thrust(df: pd.DataFrame):
    c=_find_col(df, ["Thrust (N)"], ["Thrust"])
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(np.nan, index=df.index)

def _resolve_mech_power(df: pd.DataFrame):
    c=_find_col(df, ["Mechanical Power (W)"], ["Mech Power","P_mech","Power (W)"])
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(np.nan, index=df.index)

def _resolve_torque(df: pd.DataFrame):
    c=_find_col(df, ["Torque (N·m)","Torque (N*m)"], ["Torque"])
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(np.nan, index=df.index)

# ------------- physics -------------
def _eta_rowwise(T, Pm, rho, disk_A):
    T = pd.to_numeric(T, errors="coerce"); Pm = pd.to_numeric(Pm, errors="coerce")
    Pi = np.sqrt(np.maximum(T,0.0)**3 / (2.0*rho*disk_A))
    eta = np.where(Pm>1e-12, Pi/np.maximum(Pm,1e-12), np.nan)
    return np.clip(eta, 0.0, 1.0)

def _avg_eta_over_window(rpm, eta_row, window, n_bins):
    if window is None or any(v is None for v in window):
        mask = np.isfinite(rpm) & np.isfinite(eta_row)
        bins=[mask]; centers=[float(np.nanmean(rpm[mask]))]
    else:
        lo,hi=float(window[0]), float(window[1])
        mask_all = np.isfinite(rpm) & np.isfinite(eta_row) & (rpm>=lo) & (rpm<=hi)
        if n_bins<=1:
            bins=[mask_all]; centers=[0.5*(lo+hi)]
        else:
            edges=np.linspace(lo,hi,n_bins+1); bins=[]; centers=[]
            for i in range(n_bins):
                m = mask_all & (rpm>=edges[i]) & (rpm<=(edges[i+1] if i==n_bins-1 else edges[i+1]))
                bins.append(m); centers.append(0.5*(edges[i]+edges[i+1]))
    rows=[]
    for bi,m in enumerate(bins):
        if not np.any(m): continue
        rows.append({
            "rpm_bin_index": bi,
            "rpm_bin_center": float(centers[bi]) if np.isfinite(centers[bi]) else np.nan,
            "prop_efficiency_mean": float(np.nanmean(eta_row[m])),
        })
    return rows

# ------------- TARE SUPPORT -------------
def _load_tare_lookup_from_csv(csv_path):
    """Return DataFrame with columns: rpm_bin, thrust_tare, torque_tare (all float)."""
    df = pd.read_csv(csv_path)
    need = ["rpm_bin","thrust_tare","torque_tare"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"tare CSV missing columns {miss}, found: {list(df.columns)}")
    out = pd.DataFrame({
        "rpm_bin":      pd.to_numeric(df["rpm_bin"], errors="coerce"),
        "thrust_tare":  pd.to_numeric(df["thrust_tare"], errors="coerce"),
        "torque_tare":  pd.to_numeric(df["torque_tare"], errors="coerce"),
    }).dropna()
    out = out.sort_values("rpm_bin").reset_index(drop=True)
    return out

def _make_tare_interpolator(tare_df):
    """Return callable(rpm_array)->(T_tare, Pm_tare) using linear interp. Pm = tau*omega."""
    if tare_df is None or tare_df.empty:
        return None
    r = tare_df["rpm_bin"].to_numpy(float)
    Tt = tare_df["thrust_tare"].to_numpy(float)
    Mt = tare_df["torque_tare"].to_numpy(float)
    def _interp(rpm_arr):
        rpm_arr = np.asarray(rpm_arr, float)
        if r.size==1:
            T = np.full_like(rpm_arr, Tt[0], float)
            tau = np.full_like(rpm_arr, Mt[0], float)
        else:
            T = np.interp(rpm_arr, r, Tt, left=Tt[0], right=Tt[-1])
            tau = np.interp(rpm_arr, r, Mt, left=Mt[0], right=Mt[-1])
        omega = 2*np.pi*(rpm_arr/60.0)
        Pm = tau * omega
        return T, Pm
    return _interp

def _discover_tare_csv(P):
    # Priority: explicit in config → data/processed → outputs/tables
    # You can add "tare_lookup_csv" under paths in config to point exactly.
    explicit = P.get("tare_lookup_csv", None)
    candidates = []
    if explicit: candidates.append(explicit)
    candidates.append(os.path.join(P["data_processed_dir"], "tare_lookup_01.csv"))
    candidates.append(os.path.join(P["outputs_tables_dir"], "tare_lookup_01.csv"))
    for c in candidates:
        if c and os.path.exists(c): return c
    return None

def _compute_constant_tare_from_files(files, rpm_window, rho, disk_A):
    """Very simple fallback: average thrust/torque→Pm across all tare files in window."""
    T_list=[]; Pm_list=[]; n_found=0
    for fp in files:
        base=os.path.basename(fp)
        try:
            df=pd.read_csv(fp)
        except Exception:
            continue
        if not _TARE_PAT.match(os.path.splitext(base)[0]):  # must be tare-like
            continue
        rpm=_resolve_rpm(df); T=_resolve_thrust(df)
        tau=_resolve_torque(df); Pm=_resolve_mech_power(df)
        # Prefer torque->Pm if torque present; else use mechanical power column
        if tau.notna().any():
            omega = 2*np.pi*(rpm/60.0)
            Pm_calc = tau*omega
        else:
            Pm_calc = Pm
        if rpm_window and all(v is not None for v in rpm_window):
            lo,hi=float(rpm_window[0]), float(rpm_window[1])
            m = np.isfinite(rpm) & (rpm>=lo) & (rpm<=hi)
        else:
            m = np.isfinite(rpm)
        if not np.any(m): continue
        T_list.append(float(np.nanmean(T[m])))
        Pm_list.append(float(np.nanmean(Pm_calc[m])))
        n_found += 1
    if n_found==0:
        return None
    T0 = float(np.nanmean(T_list))
    P0 = float(np.nanmean(Pm_list))
    # Build a degenerate 1-point "lookup"
    df = pd.DataFrame({"rpm_bin":[0.0], "thrust_tare":[T0], "torque_tare":[P0/(2*np.pi*1.0)]})
    return df

# ------------- main pipeline -------------
def main():
    C=load_cfg(); P=C["paths"]
    out_parquet = os.path.join(P["data_processed_dir"], "master_dataset.parquet")
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)

    # DOE geometry (truth) + canonical join key
    doe_csv=P["doe_csv"]
    if not os.path.exists(doe_csv): raise SystemExit(f"DOE CSV not found: {doe_csv}")
    doe_raw = pd.read_csv(doe_csv)
    if "filename" not in doe_raw.columns: raise SystemExit("DOE CSV missing 'filename' column.")
    doe = doe_raw[["filename"]+GEO].drop_duplicates("filename").copy()
    for c in GEO: doe[c]=pd.to_numeric(doe[c], errors="coerce")
    doe["aoaRoot (deg)"]=_sanitize_aoa_deg(doe["aoaRoot (deg)"])
    doe["aoaTip (deg)"] =_sanitize_aoa_deg(doe["aoaTip (deg)"])
    doe = doe.dropna(subset=GEO).reset_index(drop=True)
    doe["fname_key"] = doe["filename"].map(_canon_name)

    # Fluid / disk
    rho=float(C["fluids"]["rho"])
    r_tip = float(C["geometry"]["r_hub_m"]) + float(C["geometry"]["span_blade_m"])
    disk_A = np.pi*(r_tip**2)

    # Config
    proc=C.get("process", {})
    rpm_window = proc.get("rpm_window", None)  # e.g., [1000, 3000]
    n_bins     = int(proc.get("rpm_n_bins", 1))
    use_tare   = bool(proc.get("use_tare", True))
    if rpm_window: print(f"[process_data] RPM window {rpm_window} with {n_bins} bin(s)")
    print(f"[process_data] Tare correction: {'ON' if use_tare else 'OFF'}")

    # Discover bench/tunnel files
    bench_files=_collect_csvs(P["data_bench_dir"])
    tunnel_files=_collect_csvs(P["data_tunnel_dir"])
    print(f"[process_data] Bench files: {len(bench_files)} | Tunnel files: {len(tunnel_files)}")

    # ----- Prepare tare interpolator -----
    tare_interp = None
    if use_tare:
        csv_path = _discover_tare_csv(P)
        if csv_path:
            try:
                tare_df = _load_tare_lookup_from_csv(csv_path)
                tare_interp = _make_tare_interpolator(tare_df)
                print(f"[process_data] Using tare lookup CSV: {csv_path} | rows: {len(tare_df)}")
            except Exception as e:
                print(f"[process_data][warn] Bad tare CSV ({csv_path}): {e}")
                tare_interp = None
        if tare_interp is None:
            # fallback: derive constant tare from any tare files present
            tt = _compute_constant_tare_from_files(bench_files, rpm_window, rho, disk_A)
            if tt is not None:
                tare_interp = _make_tare_interpolator(tt)
                print(f"[process_data] Using constant tare from tare files (fallback).")
            else:
                print("[process_data][warn] No tare CSV and no tare files; proceeding without tare.")
                use_tare = False

    rows=[]; used_bench=used_tunnel=skipped=0

    # -------- BENCH (hover η) --------
    for fp in bench_files:
        base=os.path.basename(fp)
        # Skip tare files as bench measurements (they've been used above)
        if _TARE_PAT.match(os.path.splitext(base)[0]):
            continue
        try:
            df=pd.read_csv(fp)
        except Exception as e:
            print(f"[skip:bench] read error {base}: {e}"); skipped+=1; continue

        fn_col=_find_col(df, ["filename"])
        if fn_col is not None:
            fn_raw=str(df[fn_col].iloc[0])
        else:
            fn_raw=_infer_filename_from_path(fp)
            if fn_raw is None:
                print(f"[skip:bench] inferred tare/mount: {base}"); skipped+=1; continue

        rpm=_resolve_rpm(df); T=_resolve_thrust(df); Pm=_resolve_mech_power(df)
        if np.all(~np.isfinite(rpm)) or np.all(~np.isfinite(T)) or np.all(~np.isfinite(Pm)):
            print(f"[skip:bench] missing rpm/thrust/mech power → {base}"); skipped+=1; continue

        # --- TARE CORRECTION ---
        if use_tare and tare_interp is not None:
            T_tare, Pm_tare = tare_interp(rpm.to_numpy(float))
            T_corr  = (T.to_numpy(float)  - T_tare)
            Pm_corr = (Pm.to_numpy(float) - Pm_tare)
            # guard against negatives
            T_corr[T_corr < 0]   = np.nan
            Pm_corr[Pm_corr <= 0]= np.nan
            eta_row = _eta_rowwise(T_corr, Pm_corr, rho, disk_A)
        else:
            eta_row = _eta_rowwise(T, Pm, rho, disk_A)

        per_bins=_avg_eta_over_window(rpm, eta_row, rpm_window, n_bins)
        if not per_bins:
            print(f"[skip:bench] no rows in RPM window → {base}"); skipped+=1; continue

        for r in per_bins:
            r["filename_scanned"]=fn_raw
            r["fname_key"]=_canon_name(fn_raw)
            r["__source"]="bench"; r["__file"]=base
            rows.append(r)
        used_bench+=1

    # -------- TUNNEL (optional L/D) --------
    for fp in tunnel_files:
        base=os.path.basename(fp)
        try:
            df=pd.read_csv(fp)
        except Exception as e:
            print(f"[skip:tunnel] read error {base}: {e}"); skipped+=1; continue

        fn_col=_find_col(df, ["filename"])
        fn_raw=str(df[fn_col].iloc[0]) if fn_col else _infer_filename_from_path(fp)
        if fn_raw is None:
            print(f"[skip:tunnel] inferred tare/mount: {base}"); skipped+=1; continue

        ld_col=_find_col(df, ["ld_cruise","L/D","LD","Open Air Efficiency (1)","Open Air Efficiency"])
        if ld_col is None:
            Lc=_find_col(df, ["Lift (N)","Lift"]); Dc=_find_col(df, ["Drag (N)","Drag"])
            if not (Lc and Dc): continue
            L=pd.to_numeric(df[Lc], errors="coerce"); D=pd.to_numeric(df[Dc], errors="coerce")
            ld_val=float(np.nanmean(np.where((np.isfinite(L))&(np.isfinite(D))&(D>1e-12), L/D, np.nan)))
        else:
            ld_val=float(pd.to_numeric(df[ld_col], errors="coerce").mean())

        rows.append({
            "filename_scanned": fn_raw,
            "fname_key": _canon_name(fn_raw),
            "ld_cruise": ld_val,
            "__source":"tunnel",
            "__file": base
        })
        used_tunnel+=1

    # Nothing usable?
    if not rows:
        print("[process_data] No usable rows; writing geometry-only parquet.")
        m = doe.copy()
        m["prop_efficiency_mean"]=np.nan; m["ld_cruise"]=np.nan
        m.to_parquet(out_parquet, index=False)
        print("Wrote", out_parquet, "| rows:", len(m), "| props:", m["filename"].nunique())
        return

    raw = pd.DataFrame(rows)
    m = raw.merge(doe[["filename","fname_key"]+GEO], on="fname_key", how="left", suffixes=("","_doe"))

    miss = m["filename"].isna()
    if miss.any():
        missed = m.loc[miss, "filename_scanned"].dropna().unique().tolist()
        print(f"[process_data][warn] {len(missed)} filenames not in DOE after canonical join (first 10): {missed[:10]}")

    m = m.dropna(subset=GEO+["filename"]).reset_index(drop=True)
    m = m.drop(columns=["fname_key"], errors="ignore")

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    m.to_parquet(out_parquet, index=False)
    print(f"[process_data] Used bench: {used_bench} | Used tunnel: {used_tunnel} | Skipped: {skipped}")
    print("Wrote", out_parquet, "| rows:", len(m), "| props:", m["filename"].nunique())
    if "prop_efficiency_mean" in m.columns:
        print("Hover rows:", int(m["prop_efficiency_mean"].notna().sum()))
    if "ld_cruise" in m.columns:
        print("Cruise rows:", int(m["ld_cruise"].notna().sum()))

if __name__ == "__main__":
    main()
