# 08_optimize_dual.py
# Continuous optimization over bounds with calibrated priors and ML fusion.

import os, numpy as np, pandas as pd
from math import pi
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
                     ("hgb",HistGradientBoostingRegressor(learning_rate=0.08, max_iter=600, random_state=42))]).fit(X,y)

def _load_cal(C):
    p=C["paths"].get("prior_calibration_csv")
    if not p or not os.path.exists(p): return {}
    df=pd.read_csv(p); d={}
    for _,r in df.iterrows():
        d[(str(r["mode"]).lower(), str(r["model"]).lower())]=(float(r["a"]), float(r["b"]))
    return d

def _apply_cal(x, cal, key):
    if key in cal:
        a,b=cal[key]; return a*np.asarray(x,float)+b
    return np.asarray(x,float)

def _sample(bounds, n, seed=42):
    rng=np.random.default_rng(seed); cols=list(bounds.keys())
    X=np.column_stack([rng.uniform(bounds[c][0], bounds[c][1], size=n) for c in cols])
    return cols, X

def _refine(top_points, n_refine, sigma=0.06, bounds=None, seed=123):
    rng=np.random.default_rng(seed); pts=[]
    per=max(1, n_refine//max(1,len(top_points)))
    for p in top_points:
        for _ in range(per):
            noise=rng.normal(0.0,sigma,size=len(p))*np.maximum(np.abs(p),1.0)
            cand=p+noise
            if bounds:
                for i,(k,(lo,hi)) in enumerate(bounds.items()):
                    cand[i]=float(np.clip(cand[i],lo,hi))
            pts.append(cand)
    return np.array(pts,float)

def _train_cfd_surrogates(C, gcols):
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

def _train_filename_surrogate(csv_path, gcols, target):
    if not os.path.exists(csv_path): return None
    df=pd.read_csv(csv_path)
    need=["filename"]+[target]
    if not set(need).issubset(df.columns): return None
    # need geometry to train; merge with DOE
    C=load_cfg()
    doe=pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename")
    for c in gcols: doe[c]=pd.to_numeric(doe[c], errors="coerce")
    tab=doe.merge(df[["filename",target]], on="filename", how="inner").dropna(subset=gcols+[target])
    if len(tab)<10: return None
    return _fit_hgbr(tab[gcols].to_numpy(float), tab[target].to_numpy(float))

def main():
    C=load_cfg()
    out=C["paths"]["outputs_tables_dir"]; os.makedirs(out, exist_ok=True)
    gcols=C["geometry_cols"]; bounds={k:tuple(C["bounds"][k]) for k in gcols}
    w_hover=float(C["selection"]["w_hover"])
    use_priors=bool(C["selection"]["use_priors"])
    mode_h=C["selection"].get("opt_mode_hover","auto").lower()
    mode_c=C["selection"].get("opt_mode_cruise","auto").lower()
    N1=int(C["selection"]["samples"]["coarse"]); N2=int(C["selection"]["samples"]["refine"])

    # Experimental models (if any)
    mpath=C["paths"]["master_parquet"]
    m=pd.read_parquet(mpath) if os.path.exists(mpath) else None
    hov_mdl=cru_mdl=None
    if m is not None:
        htab=m.dropna(subset=gcols+["prop_efficiency_mean"])
        if htab["filename"].nunique()>=3: hov_mdl=_fit_ridge(htab[gcols].to_numpy(float), htab["prop_efficiency_mean"].clip(0,1).to_numpy(float))
        if "ld_cruise" in m.columns:
            ctab=m.dropna(subset=gcols+["ld_cruise"])
            if ctab["filename"].nunique()>=3: cru_mdl=_fit_ridge(ctab[gcols].to_numpy(float), ctab["ld_cruise"].to_numpy(float))

    # Priors: CFD surrogates + (optional) filename-based BEMT/LL surrogates
    vt_cfd, cr_cfd = _train_cfd_surrogates(C, gcols)
    bemt_sur = _train_filename_surrogate(os.path.join(out,"bemt_avg_prior.csv"), gcols, "eta_bemt_mean")
    ll_sur   = _train_filename_surrogate(os.path.join(out,"ll_cruise_prior.csv"), gcols, "LD_ll")

    # Calibration
    cal=_load_cal(C)

    # sampling
    keys, X1 = _sample(bounds, N1, seed=42); df1=pd.DataFrame(X1, columns=keys)

    # Hover channel
    eta=np.full(len(df1),np.nan)
    # data channel
    if hov_mdl is not None and (mode_h in ("auto","data_only","hybrid")):
        eta = hov_mdl.predict(df1[gcols].to_numpy(float)).clip(0,1)
    # priors channel
    if use_priors and (mode_h in ("auto","priors_only","hybrid")):
        cfd = vt_cfd.predict(df1[gcols].to_numpy(float)).clip(0,1) if vt_cfd is not None else np.full(len(df1),np.nan)
        bem = bemt_sur.predict(df1[gcols].to_numpy(float)) if bemt_sur is not None else np.full(len(df1),np.nan)
        cfd_cal = _apply_cal(cfd, cal, ("hover","cfd"))
        bem_cal = _apply_cal(bem, cal, ("hover","bemt"))
        pw=C["selection"]["priors_weights"]["hover"]
        ph_bemt=float(pw.get("bemt",0.0)); ph_cfd=float(pw.get("cfd",1.0))
        s=ph_bemt+ph_cfd if (np.isfinite(ph_bemt)+np.isfinite(ph_cfd)) else 1.0
        ph_bemt/=s; ph_cfd/=s
        prior_hover = ph_bemt*bem_cal + ph_cfd*cfd_cal
        if mode_h in ("priors_only",) or eta.size==0 or ~np.isfinite(eta).any():
            eta = prior_hover
        elif mode_h in ("auto","hybrid"):
            wh=float(C["selection"].get("w_data_vs_priors_hover",0.85))
            msk=~np.isfinite(eta); eta[msk]=prior_hover[msk]
            eta=np.where(np.isfinite(eta), wh*eta+(1.0-wh)*prior_hover, prior_hover)

    # Cruise channel
    LD=np.full(len(df1),np.nan)
    if cru_mdl is not None and (mode_c in ("auto","data_only","hybrid")):
        LD=cru_mdl.predict(df1[gcols].to_numpy(float))
    if use_priors and (mode_c in ("auto","priors_only","hybrid")):
        cfd = cr_cfd.predict(df1[gcols].to_numpy(float)) if cr_cfd is not None else np.full(len(df1),np.nan)
        ll  = ll_sur.predict(df1[gcols].to_numpy(float)) if ll_sur is not None else np.full(len(df1),np.nan)
        cfd_cal = _apply_cal(cfd, cal, ("cruise","cfd"))
        ll_cal  = _apply_cal(ll,  cal, ("cruise","ll"))
        pw=C["selection"]["priors_weights"]["cruise"]
        pc_ll=float(pw.get("ll",0.5)); pc_cfd=float(pw.get("cfd",0.5))
        s=pc_ll+pc_cfd if (np.isfinite(pc_ll)+np.isfinite(pc_cfd)) else 1.0
        pc_ll/=s; pc_cfd/=s
        prior_cruise = pc_ll*ll_cal + pc_cfd*cfd_cal
        if mode_c in ("priors_only",) or LD.size==0 or ~np.isfinite(LD).any():
            LD = prior_cruise
        elif mode_c in ("auto","hybrid"):
            wc=float(C["selection"].get("w_data_vs_priors_cruise",0.85))
            msk=~np.isfinite(LD); LD[msk]=prior_cruise[msk]
            LD=np.where(np.isfinite(LD), wc*LD+(1.0-wc)*prior_cruise, prior_cruise)

    # Dual-mode score
    score = w_hover*z01(eta) + (1.0-w_hover)*z01(LD)

    # refine sampling around top K
    K=max(20, int(0.01*N1))
    top_idx = np.argsort(score)[::-1][:K]
    X2 = _refine(df1[gcols].to_numpy(float)[top_idx,:], N2, sigma=0.05, bounds=bounds)
    df2=pd.DataFrame(X2, columns=gcols)

    # evaluate refined set with same pipeline
    def _eval_set(df):
        # hover
        e=np.full(len(df),np.nan)
        if hov_mdl is not None and (mode_h in ("auto","data_only","hybrid")):
            e = hov_mdl.predict(df[gcols].to_numpy(float)).clip(0,1)
        if use_priors and (mode_h in ("auto","priors_only","hybrid")):
            cf = vt_cfd.predict(df[gcols].to_numpy(float)).clip(0,1) if vt_cfd is not None else np.full(len(df),np.nan)
            bm = bemt_sur.predict(df[gcols].to_numpy(float)) if bemt_sur is not None else np.full(len(df),np.nan)
            cf=_apply_cal(cf, cal, ("hover","cfd")); bm=_apply_cal(bm, cal, ("hover","bemt"))
            pw=C["selection"]["priors_weights"]["hover"]; s=pw["bemt"]+pw["cfd"]
            phb=pw["bemt"]/s; phc=pw["cfd"]/s
            ph = phb*bm + phc*cf
            if mode_h=="priors_only" or e.size==0 or ~np.isfinite(e).any():
                e=ph
            elif mode_h in ("auto","hybrid"):
                wh=float(C["selection"].get("w_data_vs_priors_hover",0.85))
                msk=~np.isfinite(e); e[msk]=ph[msk]
                e=np.where(np.isfinite(e), wh*e+(1.0-wh)*ph, ph)
        # cruise
        L=np.full(len(df),np.nan)
        if cru_mdl is not None and (mode_c in ("auto","data_only","hybrid")):
            L = cru_mdl.predict(df[gcols].to_numpy(float))
        if use_priors and (mode_c in ("auto","priors_only","hybrid")):
            cf = cr_cfd.predict(df[gcols].to_numpy(float)) if cr_cfd is not None else np.full(len(df),np.nan)
            ll = ll_sur.predict(df[gcols].to_numpy(float)) if ll_sur is not None else np.full(len(df),np.nan)
            cf=_apply_cal(cf, cal, ("cruise","cfd")); ll=_apply_cal(ll, cal, ("cruise","ll"))
            pw=C["selection"]["priors_weights"]["cruise"]; s=pw["ll"]+pw["cfd"]
            pll=pw["ll"]/s; pcf=pw["cfd"]/s
            pc = pll*ll + pcf*cf
            if mode_c=="priors_only" or L.size==0 or ~np.isfinite(L).any():
                L=pc
            elif mode_c in ("auto","hybrid"):
                wc=float(C["selection"].get("w_data_vs_priors_cruise",0.85))
                msk=~np.isfinite(L); L[msk]=pc[msk]
                L=np.where(np.isfinite(L), wc*L+(1.0-wc)*pc, pc)
        sc = w_hover*z01(e) + (1.0-w_hover)*z01(L)
        return e,L,sc

    eta2, LD2, sc2 = _eval_set(df2)
    best_idx = int(np.argmax(sc2))
    best = df2.iloc[[best_idx]].copy()
    best["pred_eta"]=float(eta2[best_idx]); best["pred_LD"]=float(LD2[best_idx]); best["dual_score"]=float(sc2[best_idx])

    save_path=os.path.join(out,"optimize_dual_best.csv")
    best.to_csv(save_path, index=False)
    print("\nBest candidate:\n")
    print(best[gcols+["pred_eta","pred_LD","dual_score"]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("\nSaved →", save_path)

if __name__ == "__main__":
    main()
