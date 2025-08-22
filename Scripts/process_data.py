# Build master_dataset.parquet from bench+tunnel raw CSVs with tare subtraction.
import os, glob, numpy as np, pandas as pd
from path_utils import load_cfg

def load_doe(C):
    df = pd.read_csv(C["paths"]["doe_csv"])
    g  = C["geometry_cols"]
    out = df[["filename"]+g].drop_duplicates("filename")
    for c in g: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=g)

def nearest_subtract(meas_df, tare_df, key_col, cols_to_sub):
    if tare_df.empty: 
        return meas_df.copy()
    t = tare_df.sort_values(key_col).reset_index(drop=True)
    m = meas_df.sort_values(key_col).reset_index(drop=True)
    tk = t[key_col].to_numpy(float); mk = m[key_col].to_numpy(float)
    idx = np.searchsorted(tk, mk).clip(0, len(tk)-1)
    left = np.maximum(idx-1, 0); right = idx
    choose_left = np.abs(mk - tk[left]) <= np.abs(mk - tk[right])
    match = np.where(choose_left, left, right)
    tmatch = t.iloc[match].reset_index(drop=True)
    out = m.copy()
    for c in cols_to_sub:
        if c in out.columns and c in tmatch.columns:
            out[c] = out[c].to_numpy(float) - tmatch[c].to_numpy(float)
    return out

def prep_bench(C):
    root = os.path.join(C["paths"]["data_dir"], "bench")
    raw  = sorted(glob.glob(os.path.join(root, "raw", "*.csv")))
    tare = sorted(glob.glob(os.path.join(root, "tare", "*.csv")))
    if not raw: return None
    br = pd.concat([pd.read_csv(p) for p in raw], ignore_index=True)
    bt = pd.concat([pd.read_csv(p) for p in tare], ignore_index=True) if tare else pd.DataFrame(columns=br.columns)
    if "rpm" not in br.columns: br["rpm"] = np.nan
    if not bt.empty and "rpm" in bt.columns:
        br = nearest_subtract(br, bt, key_col="rpm", cols_to_sub=["thrust_N","torque_Nm","power_W"])
    if "prop_efficiency" in br.columns:
        br["eta_hover"] = br["prop_efficiency"].astype(float).clip(0,1)
    elif {"thrust_N","power_W"}.issubset(br.columns):
        br["eta_hover"] = (br["thrust_N"].astype(float) / br["power_W"].astype(float)).clip(lower=0)
        br["eta_hover"] = br.groupby("filename")["eta_hover"].transform(lambda s: (s - s.min())/(s.max()-s.min()+1e-12))
    else:
        return None
    agg = br.groupby("filename", as_index=False)["eta_hover"].mean()
    agg = agg.rename(columns={"eta_hover":"prop_efficiency_mean"})
    return agg

def prep_tunnel(C):
    root = os.path.join(C["paths"]["data_dir"], "tunnel")
    raw  = sorted(glob.glob(os.path.join(root, "raw", "*.csv")))
    tare = sorted(glob.glob(os.path.join(root, "tare", "*.csv")))
    if not raw: return None
    tr = pd.concat([pd.read_csv(p) for p in raw], ignore_index=True)
    tt = pd.concat([pd.read_csv(p) for p in tare], ignore_index=True) if tare else pd.DataFrame(columns=tr.columns)
    key = "airspeed_mps" if "airspeed_mps" in tr.columns else ("V_inf" if "V_inf" in tr.columns else None)
    if key and key in tt.columns:
        tr = nearest_subtract(tr, tt, key_col=key, cols_to_sub=["drag_N","lift_N","CD","CL"])
    if {"lift_N","drag_N"}.issubset(tr.columns):
        tr["ld_cruise"] = (tr["lift_N"].astype(float) / tr["drag_N"].astype(float)).replace([np.inf,-np.inf], np.nan)
    elif {"CL","CD"}.issubset(tr.columns):
        tr["ld_cruise"] = (tr["CL"].astype(float) / tr["CD"].astype(float)).replace([np.inf,-np.inf], np.nan)
    else:
        return None
    agg = tr.groupby("filename", as_index=False)["ld_cruise"].mean()
    return agg

def main():
    C = load_cfg()
    tools = C["paths"]["tools_dir"]
    os.makedirs(tools, exist_ok=True)
    doe  = load_doe(C)
    bench= prep_bench(C)
    tun  = prep_tunnel(C)
    parts = [doe]
    if bench is not None: parts.append(bench)
    if tun   is not None: parts.append(tun)
    master = parts[0]
    for p in parts[1:]:
        master = master.merge(p, on="filename", how="left")
    master.to_parquet(os.path.join(tools, "master_dataset.parquet"), index=False)
    print("Wrote master_dataset.parquet with columns:", list(master.columns))

if __name__ == "__main__":
    main()
