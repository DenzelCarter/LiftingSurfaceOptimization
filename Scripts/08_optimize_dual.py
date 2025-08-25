# 08_optimize_dual.py
# Continuous optimization over bounds with independent hover/cruise channels.
# Uses EXP models if available; else CFD surrogates; LL is last fallback for cruise.
# Outputs:
#   Experiment/outputs/tables/optimize_dual_candidates.csv
#   Experiment/outputs/tables/optimize_dual_best.csv

import os, numpy as np, pandas as pd
from math import pi
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

# ---------- helpers ----------

def z01(v):
    v=np.asarray(v,float); lo,hi=np.nanmin(v),np.nanmax(v)
    return np.zeros_like(v) if (not np.isfinite(lo) or hi-lo<1e-12) else (v-lo)/(hi-lo)

def _fit_ridge(X,y):
    n=X.shape[0]; cvk=min(5,max(3,n))
    return Pipeline([
        ("imp",SimpleImputer(strategy="mean")),
        ("scl",StandardScaler()),
        ("rid",RidgeCV(alphas=np.logspace(-6,3,40), cv=cvk, scoring="neg_mean_absolute_error"))
    ]).fit(X,y)

def _fit_hgbr(X,y):
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scl",StandardScaler()),
        ("hgb",HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42))
    ]).fit(X,y)

def _oswald(AR, lam, e0, k):
    return float(np.clip(e0 - k*(1.0-lam)**2, 0.70, 0.98))

def _LL(C, row):
    AR=float(row["AR"]); lam=float(row["lambda"])
    CLt=float(C["cruise_ll"]["target_CL"])
    CD0=float(C["cruise_ll"]["CD0"]); K2=float(C["cruise_ll"]["K2"])
    e=_oswald(AR,lam,float(C["cruise_ll"]["e_base"]), float(C["cruise_ll"]["e_taper_k"]))
    CD = CD0 + K2*CLt*CLt + CLt*CLt/(pi*AR*e)
    return float(CLt/CD) if CD>1e-12 else np.nan

def _sample(bounds: dict, n: int, seed=42):
    rng=np.random.default_rng(seed); cols=list(bounds.keys())
    X=np.column_stack([rng.uniform(bounds[c][0], bounds[c][1], size=n) for c in cols])
    return cols, X

def _refine(top_points, n_refine, sigma=0.06, bounds=None, seed=123):
    if len(top_points)==0 or n_refine<=0: return np.zeros((0, len(bounds)), float)
    rng=np.random.default_rng(seed); pts=[]
    per=max(1, n_refine//len(top_points))
    keys=list(bounds.keys())
    spans=np.array([bounds[k][1]-bounds[k][0] for k in keys], float)
    for p in top_points:
        for _ in range(per):
            noise=rng.normal(0.0, sigma, size=len(p))*np.maximum(spans, 1e-6)
            cand=p+noise
            # clamp to bounds
            for i,k in enumerate(keys):
                lo,hi=bounds[k]; cand[i]=float(np.clip(cand[i], lo, hi))
            pts.append(cand)
    return np.array(pts,float)

def _train_cfd_surrogates(C, gcols):
    """Train geometry-only CFD surrogates from cfd_master.csv if available."""
    tdir=C["paths"]["outputs_tables_dir"]
    cfd_path=os.path.join(tdir,"cfd_master.csv")
    vt_mdl=cr_mdl=None
    if not os.path.exists(cfd_path):
        print("[CFD] cfd_master.csv not found → skip CFD surrogates.")
        return vt_mdl, cr_mdl

    df=pd.read_csv(cfd_path)
    if "mode" not in df.columns:
        print("[CFD] 'mode' column missing → skip CFD surrogates.")
        return vt_mdl, cr_mdl

    mode = df["mode"].astype(str).str.lower()

    # VTOL
    vt = df.loc[mode=="vtol"].copy()
    if not vt.empty and all(c in vt.columns for c in gcols+["eta_cfd"]):
        vt = vt.dropna(subset=gcols+["eta_cfd"])
        if len(vt) >= 10:
            vt_mdl = _fit_hgbr(vt[gcols].to_numpy(float), vt["eta_cfd"].clip(0,1).to_numpy(float))
            print(f"[CFD] VTOL surrogate trained on {len(vt)} rows.")
        else:
            print(f"[CFD] VTOL rows after dropna < 10 ({len(vt)}).")
    else:
        print("[CFD] VTOL block missing required columns or empty.")

    # Cruise
    cr = df.loc[mode=="cruise"].copy()
    if not cr.empty and all(c in cr.columns for c in gcols+["LD_cfd"]):
        cr = cr.dropna(subset=gcols+["LD_cfd"])
        if len(cr) >= 10:
            cr_mdl = _fit_hgbr(cr[gcols].to_numpy(float), cr["LD_cfd"].to_numpy(float))
            print(f"[CFD] Cruise surrogate trained on {len(cr)} rows.")
        else:
            print(f"[CFD] Cruise rows after dropna < 10 ({len(cr)}).")
    else:
        print("[CFD] Cruise block missing required columns or empty.")

    return vt_mdl, cr_mdl

# ---------- main ----------

def main():
    C=load_cfg()
    out_dir=C["paths"]["outputs_tables_dir"]; os.makedirs(out_dir, exist_ok=True)
    gcols=C["geometry_cols"]; bounds={k:tuple(C["bounds"][k]) for k in gcols}

    # weights/modes
    w_hover=float(C["selection"]["w_hover"])
    use_priors=bool(C["selection"]["use_priors"])
    mode_h = C["selection"].get("opt_mode_hover","auto").lower()
    mode_c = C["selection"].get("opt_mode_cruise","auto").lower()
    N1=int(C["selection"]["samples"]["coarse"])
    N2=int(C["selection"]["samples"]["refine"])

    # EXP models (if any data)
    mpath=C["paths"]["master_parquet"]
    m=pd.read_parquet(mpath) if os.path.exists(mpath) else None
    hov_mdl=cru_mdl=None
    if m is not None:
        if "prop_efficiency_mean" in m.columns:
            htab=m.dropna(subset=gcols+["prop_efficiency_mean"])
            if htab["filename"].nunique()>=3:
                hov_mdl=_fit_ridge(htab[gcols].to_numpy(float), htab["prop_efficiency_mean"].clip(0,1).to_numpy(float))
                print(f"[EXP] Hover ridge on {htab['filename'].nunique()} props.")
        if "ld_cruise" in m.columns:
            ctab=m.dropna(subset=gcols+["ld_cruise"])
            if ctab["filename"].nunique()>=3:
                cru_mdl=_fit_ridge(ctab[gcols].to_numpy(float), ctab["ld_cruise"].to_numpy(float))
                print(f"[EXP] Cruise ridge on {ctab['filename'].nunique()} props.")

    # CFD surrogates
    vt_cfd, cr_cfd = _train_cfd_surrogates(C, gcols)

    have_hover = hov_mdl is not None
    have_cruise = cru_mdl is not None
    # effective modes when 'auto'
    mode_h_eff = "hybrid" if (mode_h=="auto" and have_hover) else ("priors_only" if mode_h=="auto" else mode_h)
    mode_c_eff = "hybrid" if (mode_c=="auto" and have_cruise) else ("priors_only" if mode_c=="auto" else mode_c)

    # -------- coarse sampling --------
    keys, X1 = _sample(bounds, N1, seed=42)
    df1 = pd.DataFrame(X1, columns=keys)

    # Hover predictions (η)
    eta = np.full(len(df1), np.nan, float)
    if mode_h_eff in ("data_only","hybrid") and have_hover:
        eta = hov_mdl.predict(df1[gcols].to_numpy(float)).clip(0,1)
    if (mode_h_eff in ("priors_only","hybrid")) and use_priors and vt_cfd is not None:
        pred = vt_cfd.predict(df1[gcols].to_numpy(float)).clip(0,1)
        if mode_h_eff=="priors_only": eta = pred
        else:
            msk = ~np.isfinite(eta)
            eta[msk] = pred[msk]

    # Cruise predictions (L/D)
    LD = np.full(len(df1), np.nan, float)
    if mode_c_eff in ("data_only","hybrid") and have_cruise:
        LD = cru_mdl.predict(df1[gcols].to_numpy(float))
    if (mode_c_eff in ("priors_only","hybrid")) and use_priors:
        if cr_cfd is not None:
            pred = cr_cfd.predict(df1[gcols].to_numpy(float))
            if mode_c_eff=="priors_only": LD = pred
            else:
                msk = ~np.isfinite(LD)
                LD[msk] = pred[msk]
        else:
            # LL fallback
            LL = np.array([_LL(C, r) for _,r in df1.iterrows()], float)
            if mode_c_eff=="priors_only": LD = LL
            else:
                msk = ~np.isfinite(LD)
                LD[msk] = LL[msk]

    s_hov = z01(eta); s_cru = z01(LD)
    dms1  = w_hover*s_hov + (1.0-w_hover)*s_cru

    # pick top seeds for refinement
    top_k = min(40, max(10, N1//200))
    idx_top = np.argsort(-dms1)[:top_k]
    seeds = df1[gcols].to_numpy(float)[idx_top]
    X2 = _refine(seeds, N2, sigma=0.06, bounds=bounds, seed=123)
    df2 = pd.DataFrame(X2, columns=gcols) if X2.size>0 else pd.DataFrame(columns=gcols)

    # -------- refinement eval --------
    if not df2.empty:
        eta2 = np.full(len(df2), np.nan, float)
        if mode_h_eff in ("data_only","hybrid") and have_hover:
            eta2 = hov_mdl.predict(df2[gcols].to_numpy(float)).clip(0,1)
        if (mode_h_eff in ("priors_only","hybrid")) and use_priors and vt_cfd is not None:
            pred = vt_cfd.predict(df2[gcols].to_numpy(float)).clip(0,1)
            if mode_h_eff=="priors_only": eta2 = pred
            else:
                msk = ~np.isfinite(eta2)
                eta2[msk] = pred[msk]

        LD2 = np.full(len(df2), np.nan, float)
        if mode_c_eff in ("data_only","hybrid") and have_cruise:
            LD2 = cru_mdl.predict(df2[gcols].to_numpy(float))
        if (mode_c_eff in ("priors_only","hybrid")) and use_priors:
            if cr_cfd is not None:
                pred = cr_cfd.predict(df2[gcols].to_numpy(float))
                if mode_c_eff=="priors_only": LD2 = pred
                else:
                    msk = ~np.isfinite(LD2)
                    LD2[msk] = pred[msk]
            else:
                LL = np.array([_LL(C, r) for _,r in df2.iterrows()], float)
                if mode_c_eff=="priors_only": LD2 = LL
                else:
                    msk = ~np.isfinite(LD2)
                    LD2[msk] = LL[msk]

        s_hov2 = z01(eta2); s_cru2 = z01(LD2)
        dms2   = w_hover*s_hov2 + (1.0-w_hover)*s_cru2

        cand1 = df1[gcols].copy(); cand1["eta_pred"]=eta;  cand1["LD_pred"]=LD;  cand1["DMS"]=dms1; cand1["stage"]="coarse"
        cand2 = df2[gcols].copy(); cand2["eta_pred"]=eta2; cand2["LD_pred"]=LD2; cand2["DMS"]=dms2; cand2["stage"]="refine"
        cand  = pd.concat([cand1, cand2], ignore_index=True)
    else:
        cand  = df1[gcols].copy(); cand["eta_pred"]=eta; cand["LD_pred"]=LD; cand["DMS"]=dms1; cand["stage"]="coarse"

    cand = cand.sort_values("DMS", ascending=False).reset_index(drop=True)
    cand.insert(0, "rank", np.arange(1, len(cand)+1))

    out_all = os.path.join(out_dir, "optimize_dual_candidates.csv")
    cand.to_csv(out_all, index=False)

    best = cand.iloc[[0]].copy()
    out_best = os.path.join(out_dir, "optimize_dual_best.csv")
    best.to_csv(out_best, index=False)

    show_cols = ["rank","DMS","eta_pred","LD_pred"] + gcols + ["stage"]
    with pd.option_context("display.max_columns", None):
        print(cand.head(12)[show_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("\nSaved:")
    print("  -", out_all)
    print("  -", out_best)
    print(f"\nBest geometry (DMS={best['DMS'].iloc[0]:.4f}):")
    print(best[gcols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

if __name__ == "__main__":
    main()
