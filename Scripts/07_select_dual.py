# 07_select_dual.py
# Rank DOE candidates by dual-mode score (exploit vs explore).
# Uses:
#   - EXP Ridge models if available
#   - Otherwise trains CFD surrogates on-the-fly from cfd_master.csv (geometry-only HGBR)
#   - LL as last fallback for cruise

import os, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

def z01(v):
    v=np.asarray(v,float)
    lo,hi=np.nanmin(v),np.nanmax(v)
    if not np.isfinite(lo) or hi-lo<1e-12: return np.zeros_like(v)
    return (v-lo)/(hi-lo)

def _fit_ridge(X,y):
    n=X.shape[0]; cvk=min(5,max(3,n))
    return Pipeline([
        ("imp",SimpleImputer(strategy="mean")),
        ("scl",StandardScaler()),
        ("rid",RidgeCV(alphas=np.logspace(-6,3,40),cv=cvk,scoring="neg_mean_absolute_error"))
    ]).fit(X,y)

def _fit_hgbr(X,y):
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scl",StandardScaler()),
        ("hgb",HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42))
    ]).fit(X,y)

def _load_or_train_cfd_surrogates(C, gcols):
    """Train lightweight geometry-only CFD surrogates from cfd_master.csv if available."""
    tdir=C["paths"]["outputs_tables_dir"]
    cfd_path=os.path.join(tdir,"cfd_master.csv")
    vt_mdl=cr_mdl=None
    if not os.path.exists(cfd_path):
        print("[CFD] cfd_master.csv not found → skipping CFD surrogates.")
        return vt_mdl, cr_mdl

    df=pd.read_csv(cfd_path)
    if "mode" not in df.columns:
        print("[CFD] 'mode' column missing in cfd_master.csv → skipping CFD surrogates.")
        return vt_mdl, cr_mdl

    # Normalize mode to lowercase strings
    mode_series = df["mode"].astype(str).str.lower()

    # VTOL surrogate (eta_cfd)
    vt_mask = (mode_series=="vtol")
    vt = df.loc[vt_mask].copy()
    if not vt.empty and all(c in vt.columns for c in gcols+["eta_cfd"]):
        vt = vt.dropna(subset=gcols+["eta_cfd"])
        if len(vt) >= 10:
            vt_mdl = _fit_hgbr(vt[gcols].to_numpy(float), vt["eta_cfd"].clip(0,1).to_numpy(float))
            print(f"[CFD] VTOL surrogate trained on {len(vt)} rows.")
        else:
            print(f"[CFD] VTOL rows after dropna < 10 ({len(vt)}). Skipping VTOL surrogate.")
    else:
        print("[CFD] VTOL block missing required columns or empty. Skipping VTOL surrogate.")

    # Cruise surrogate (LD_cfd)
    cr_mask = (mode_series=="cruise")
    cr = df.loc[cr_mask].copy()
    if not cr.empty and all(c in cr.columns for c in gcols+["LD_cfd"]):
        cr = cr.dropna(subset=gcols+["LD_cfd"])
        if len(cr) >= 10:
            cr_mdl = _fit_hgbr(cr[gcols].to_numpy(float), cr["LD_cfd"].to_numpy(float))
            print(f"[CFD] Cruise surrogate trained on {len(cr)} rows.")
        else:
            print(f"[CFD] Cruise rows after dropna < 10 ({len(cr)}). Skipping cruise surrogate.")
    else:
        print("[CFD] Cruise block missing required columns or empty. Skipping cruise surrogate.")

    return vt_mdl, cr_mdl

def main():
    C=load_cfg()
    gcols=C["geometry_cols"]
    out_dir=C["paths"]["outputs_tables_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # DOE
    doe_path=C["paths"]["doe_csv"]
    if not os.path.exists(doe_path):
        raise SystemExit(f"DOE CSV not found: {doe_path}")
    df=pd.read_csv(doe_path)[["filename"]+gcols].drop_duplicates("filename")
    for c in gcols: df[c]=pd.to_numeric(df[c], errors="coerce")
    df=df.dropna(subset=gcols).reset_index(drop=True)

    # Experimental master (if any)
    mpath=C["paths"]["master_parquet"]
    m=pd.read_parquet(mpath) if os.path.exists(mpath) else None

    # EXP ridge models (hover/cruise)
    hov_pred=np.full(len(df), np.nan, float)
    cru_pred=np.full(len(df), np.nan, float)
    if m is not None:
        # hover
        if "prop_efficiency_mean" in m.columns:
            htab=m.dropna(subset=gcols+["prop_efficiency_mean"])
            if htab["filename"].nunique()>=3:
                mdl=_fit_ridge(htab[gcols].to_numpy(float), htab["prop_efficiency_mean"].clip(0,1).to_numpy(float))
                hov_pred = mdl.predict(df[gcols].to_numpy(float)).clip(0,1)
                print(f"[EXP] Hover ridge trained on {htab['filename'].nunique()} props.")
        # cruise
        if "ld_cruise" in m.columns:
            ctab=m.dropna(subset=gcols+["ld_cruise"])
            if ctab["filename"].nunique()>=3:
                mdl=_fit_ridge(ctab[gcols].to_numpy(float), ctab["ld_cruise"].to_numpy(float))
                cru_pred = mdl.predict(df[gcols].to_numpy(float))
                print(f"[EXP] Cruise ridge trained on {ctab['filename'].nunique()} props.")

    # CFD surrogates as fallback/augment
    vt_cfd, cr_cfd = _load_or_train_cfd_surrogates(C, gcols)

    # Fallbacks
    if ~np.isfinite(hov_pred).any() and vt_cfd is not None:
        hov_pred = vt_cfd.predict(df[gcols].to_numpy(float)).clip(0,1)
        print("[SEL] Using CFD VTOL surrogate for hover predictions.")
    if ~np.isfinite(cru_pred).any():
        if cr_cfd is not None:
            cru_pred = cr_cfd.predict(df[gcols].to_numpy(float))
            print("[SEL] Using CFD cruise surrogate for L/D predictions.")
        else:
            # simple LL fallback (no polars to keep this fast)
            CLt=float(C["cruise_ll"]["target_CL"]); CD0=float(C["cruise_ll"]["CD0"]); K2=float(C["cruise_ll"]["K2"])
            e0=float(C["cruise_ll"]["e_base"]); ek=float(C["cruise_ll"]["e_taper_k"])
            def _oswald(AR,lam): return float(np.clip(e0 - ek*(1.0-lam)**2, 0.70, 0.98))
            def _LL_row(r):
                AR=float(r["AR"]); lam=float(r["lambda"]); e=_oswald(AR,lam)
                CD=CD0+K2*CLt*CLt + CLt*CLt/(np.pi*AR*e)
                return float(CLt/CD) if CD>1e-12 else np.nan
            cru_pred = df.apply(_LL_row, axis=1).to_numpy(float)
            print("[SEL] Using simple LL fallback for L/D predictions.")

    # Scores
    w_hover=float(C["selection"]["w_hover"])
    exploit = w_hover*z01(hov_pred) + (1.0-w_hover)*z01(cru_pred)

    # Explore score = distance from tested set in standardized geometry space
    if m is not None and "filename" in m.columns:
        tested_names=set(m["filename"].unique())
        tested_mask = df["filename"].isin(tested_names).to_numpy()
        Z = df[gcols].to_numpy(float)
        mu=Z.mean(axis=0); sd=Z.std(axis=0); sd[sd<1e-12]=1.0
        Z=(Z-mu)/sd
        Zt=Z[tested_mask]
        if Zt.size>0:
            dmin = cdist(Z, Zt).min(axis=1)
        else:
            dmin = np.linalg.norm(Z, axis=1)
    else:
        Z = df[gcols].to_numpy(float)
        mu=Z.mean(axis=0); sd=Z.std(axis=0); sd[sd<1e-12]=1.0
        Z=(Z-mu)/sd
        dmin=np.linalg.norm(Z,axis=1)
    explore=z01(dmin)

    wexp=float(C["selection"]["w_exploit"])
    comp = wexp*exploit + (1.0-wexp)*explore

    outdf = df.copy()
    outdf["pred_eta_mean"]=hov_pred
    outdf["pred_LD"]=cru_pred
    outdf["exploit_score"]=exploit
    outdf["explore_score"]=explore
    outdf["composite_score"]=comp
    outdf = outdf.sort_values("composite_score", ascending=False).reset_index(drop=True)
    outdf.insert(0,"rank", np.arange(1,len(outdf)+1))
    out_path=os.path.join(out_dir,"next_props_recommendations.csv")
    outdf.to_csv(out_path, index=False)

    # Console preview
    cols_show=["rank","filename","composite_score","exploit_score","explore_score","pred_eta_mean","pred_LD"]+gcols
    with pd.option_context("display.max_columns", None):
        print(outdf.head(12)[cols_show].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("\nSaved →", out_path)

if __name__ == "__main__":
    main()
