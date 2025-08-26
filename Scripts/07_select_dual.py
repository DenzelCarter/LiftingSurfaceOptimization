# 07_select_dual.py
# Rank DOE candidates by dual-mode score with calibrated priors + ML fusion.

import os, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

def z01(v):
    v=np.asarray(v,float); lo,hi=np.nanmin(v),np.nanmax(v)
    return np.zeros_like(v) if not np.isfinite(lo) or hi-lo<1e-12 else (v-lo)/(hi-lo)

def _fit_ridge(X,y):
    n=X.shape[0]; cvk=min(5,max(3,n))
    return Pipeline([("imp",SimpleImputer(strategy="mean")),
                     ("scl",StandardScaler()),
                     ("rid",RidgeCV(alphas=np.logspace(-6,3,40),cv=cvk,scoring="neg_mean_absolute_error"))]).fit(X,y)
def _fit_hgbr(X,y):
    return Pipeline([("imp",SimpleImputer(strategy="median")),
                     ("scl",StandardScaler()),
                     ("hgb",HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42))]).fit(X,y)

def _load_cal(C):
    p=C["paths"].get("prior_calibration_csv")
    if not p or not os.path.exists(p): return {}
    df=pd.read_csv(p)
    d={}
    for _,r in df.iterrows():
        d[(str(r["mode"]).lower(), str(r["model"]).lower())]=(float(r["a"]), float(r["b"]))
    return d

def _apply_cal(x, cal, key):
    if key in cal:
        a,b=cal[key]; return a*np.asarray(x,float)+b
    return np.asarray(x,float)

def _load_or_train_cfd_surrogates(C, gcols):
    tdir=C["paths"]["outputs_tables_dir"]
    cfd_path=os.path.join(tdir,"cfd_master.csv")
    vt_mdl=cr_mdl=None
    if os.path.exists(cfd_path):
        df=pd.read_csv(cfd_path)
        if "mode" in df.columns:
            vt=df[df["mode"]=="vtol"]
            cr=df[df["mode"]=="cruise"]
            if set(gcols+["eta_cfd"]).issubset(vt.columns) and len(vt.dropna(subset=gcols+["eta_cfd"]))>=10:
                tab=vt.dropna(subset=gcols+["eta_cfd"])
                vt_mdl=_fit_hgbr(tab[gcols].to_numpy(float), tab["eta_cfd"].clip(0,1).to_numpy(float))
            if set(gcols+["LD_cfd"]).issubset(cr.columns) and len(cr.dropna(subset=gcols+["LD_cfd"]))>=10:
                tab=cr.dropna(subset=gcols+["LD_cfd"])
                cr_mdl=_fit_hgbr(tab[gcols].to_numpy(float), tab["LD_cfd"].to_numpy(float))
    return vt_mdl, cr_mdl

def main():
    C=load_cfg()
    gcols=C["geometry_cols"]
    out=C["paths"]["outputs_tables_dir"]; os.makedirs(out, exist_ok=True)

    df=pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename").dropna(subset=gcols)
    for c in gcols: df[c]=pd.to_numeric(df[c], errors="coerce")

    # EXP models on tested set (if any)
    mpath=C["paths"]["master_parquet"]
    m=pd.read_parquet(mpath) if os.path.exists(mpath) else None

    hov=np.full(len(df),np.nan); cru=np.full(len(df),np.nan)
    hov_mdl=cru_mdl=None
    if m is not None:
        htab=m.dropna(subset=gcols+["prop_efficiency_mean"])
        if htab["filename"].nunique()>=3:
            hov_mdl=_fit_ridge(htab[gcols].to_numpy(float), htab["prop_efficiency_mean"].clip(0,1).to_numpy(float))
            hov=hov_mdl.predict(df[gcols].to_numpy(float)).clip(0,1)
        if "ld_cruise" in m.columns:
            ctab=m.dropna(subset=gcols+["ld_cruise"])
            if ctab["filename"].nunique()>=3:
                cru_mdl=_fit_ridge(ctab[gcols].to_numpy(float), ctab["ld_cruise"].to_numpy(float))
                cru=cru_mdl.predict(df[gcols].to_numpy(float))

    # Priors (CFD + BEMT/LL) + calibration
    cal=_load_cal(C)
    vt_cfd, cr_cfd = _load_or_train_cfd_surrogates(C, gcols)

    # BEMT avg is per-filename; bring it in
    bemt_csv=os.path.join(out,"bemt_avg_prior.csv")
    bemt=np.full(len(df), np.nan)
    if os.path.exists(bemt_csv):
        b=pd.read_csv(bemt_csv)[["filename","eta_bemt_mean"]]
        tmp=df[["filename"]].merge(b, on="filename", how="left")
        bemt=tmp["eta_bemt_mean"].to_numpy(float)

    # LL prior (filename-level)
    ll_csv=os.path.join(out,"ll_cruise_prior.csv")
    ll=np.full(len(df), np.nan)
    if os.path.exists(ll_csv):
        L=pd.read_csv(ll_csv)[["filename","LD_ll"]]
        tmp=df[["filename"]].merge(L, on="filename", how="left")
        ll=tmp["LD_ll"].to_numpy(float)

    # CFD surrogates → DOE geometry
    cfd_hov = vt_cfd.predict(df[gcols].to_numpy(float)).clip(0,1) if vt_cfd is not None else np.full(len(df), np.nan)
    cfd_cru = cr_cfd.predict(df[gcols].to_numpy(float))           if cr_cfd is not None else np.full(len(df), np.nan)

    # Apply calibration to priors
    bemt_cal = _apply_cal(bemt, cal, ("hover","bemt"))
    cfdh_cal = _apply_cal(cfd_hov, cal, ("hover","cfd"))
    ll_cal   = _apply_cal(ll,   cal, ("cruise","ll"))
    cfdc_cal = _apply_cal(cfd_cru, cal, ("cruise","cfd"))

    # Blend priors internally per channel
    pw=C["selection"]["priors_weights"]
    ph_bemt=float(pw["hover"].get("bemt",0.0)); ph_cfd=float(pw["hover"].get("cfd",1.0))
    s = ph_bemt + ph_cfd if (np.isfinite(ph_bemt)+np.isfinite(ph_cfd)) else 1.0
    ph_bemt/=s; ph_cfd/=s
    prior_hover = ph_bemt*np.asarray(bemt_cal,float) + ph_cfd*np.asarray(cfdh_cal,float)

    pc_ll=float(pw["cruise"].get("ll",0.5)); pc_cfd=float(pw["cruise"].get("cfd",0.5))
    s = pc_ll + pc_cfd if (np.isfinite(pc_ll)+np.isfinite(pc_cfd)) else 1.0
    pc_ll/=s; pc_cfd/=s
    prior_cruise = pc_ll*np.asarray(ll_cal,float) + pc_cfd*np.asarray(cfdc_cal,float)

    # Fuse ML + priors in exploit
    use_priors_in_exploit=bool(C["selection"].get("use_priors_in_exploit", True))
    wh=float(C["selection"].get("w_data_vs_priors_hover",0.85))
    wc=float(C["selection"].get("w_data_vs_priors_cruise",0.85))

    pred_hover = np.where(np.isfinite(hov),
                          wh*hov + (1.0-wh)*prior_hover if use_priors_in_exploit else hov,
                          prior_hover)
    pred_cruise = np.where(np.isfinite(cru),
                           wc*cru + (1.0-wc)*prior_cruise if use_priors_in_exploit else cru,
                           prior_cruise)

    # Scores
    w_hover=float(C["selection"]["w_hover"])
    exploit = w_hover*z01(pred_hover) + (1.0-w_hover)*z01(pred_cruise)

    # Explore = distance from tested set
    if m is not None and "filename" in m.columns:
        tested = df["filename"].isin(set(m["filename"].unique()))
        mu=df[gcols].to_numpy(float).mean(axis=0); sd=df[gcols].to_numpy(float).std(axis=0); sd[sd<1e-12]=1.0
        Z=df[gcols].to_numpy(float); Z=(Z-mu)/sd
        Zt=Z[tested.to_numpy()]
        dmin = cdist(Z, Zt).min(axis=1) if Zt.size>0 else np.linalg.norm(Z,axis=1)
    else:
        mu=df[gcols].to_numpy(float).mean(axis=0); sd=df[gcols].to_numpy(float).std(axis=0); sd[sd<1e-12]=1.0
        Z=(df[gcols].to_numpy(float)-mu)/sd
        dmin=np.linalg.norm(Z,axis=1)
    explore=z01(dmin)

    wexp=float(C["selection"]["w_exploit"])
    comp = wexp*exploit + (1.0-wexp)*explore

    outdf = df.copy()
    outdf["pred_eta_mean"]=pred_hover; outdf["pred_LD"]=pred_cruise
    outdf["exploit_score"]=exploit; outdf["explore_score"]=explore; outdf["composite_score"]=comp
    outdf = outdf.sort_values("composite_score", ascending=False).reset_index(drop=True)
    outdf.insert(0,"rank", np.arange(1,len(outdf)+1))
    outdf.to_csv(os.path.join(out,"next_props_recommendations.csv"), index=False)

    print(outdf.head(10)[["rank","filename","composite_score","exploit_score","explore_score","pred_eta_mean","pred_LD"]]
          .to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("\nSaved →", os.path.join(out,"next_props_recommendations.csv"))

if __name__ == "__main__":
    main()
