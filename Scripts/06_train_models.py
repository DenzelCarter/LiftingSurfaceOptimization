# 06_train_models.py
# Train geometry-only Ridge models:
#   - hover:   prop_efficiency_mean
#   - cruise:  ld_cruise
# PLUS: fit affine calibrations that map priors → experimental (and save CSV).
# Writes to Experiment/outputs/tables:
#   - hover_predictions.csv, prop_hover_metrics.csv
#   - cruise_predictions.csv, wing_cruise_metrics.csv
#   - prior_calibration.csv

import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from path_utils import load_cfg

def _fit_ridge(X,y):
    n=X.shape[0]; cvk=min(5,max(3,n))
    return Pipeline([
        ("imp",SimpleImputer(strategy="mean")),
        ("scl",StandardScaler()),
        ("rid",RidgeCV(alphas=np.logspace(-6,3,40), cv=cvk, scoring="neg_mean_absolute_error"))
    ]).fit(X,y)

def _lopo(df, gcols, target):
    files = df["filename"].unique().tolist()
    rows=[]; yhat=[]; ytrue=[]
    for f in files:
        te=df[df.filename==f]; tr=df[df.filename!=f]
        if tr[target].notna().sum()<3: continue
        mdl=_fit_ridge(tr[gcols].to_numpy(float), tr[target].to_numpy(float))
        p = mdl.predict(te[gcols].to_numpy(float))
        rows.append({"filename":f,"y_true":float(te[target].mean()),"y_pred":float(p.mean())})
        ytrue.append(te[target].mean()); yhat.append(p.mean())
    if not rows:
        return np.nan, np.nan, np.nan, pd.DataFrame(columns=["filename","y_true","y_pred"])
    y=np.array(ytrue,float); yb=np.array(yhat,float)
    r2 = 1 - np.sum((y-yb)**2)/(np.sum((y-y.mean())**2)+1e-12)
    mae=float(np.mean(np.abs(y-yb))); rho=float(spearmanr(y,yb).statistic)
    return r2, mae, rho, pd.DataFrame(rows).sort_values("filename")

# ---------- prior calibration helpers ----------
def _fit_affine(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return None
    X=np.column_stack([x[m], np.ones(m.sum())])
    a,b = np.linalg.lstsq(X, y[m], rcond=None)[0]
    yhat = a*x[m] + b
    ss_res = np.sum((y[m]-yhat)**2)
    ss_tot = np.sum((y[m]-y[m].mean())**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    mae = float(np.mean(np.abs(y[m]-yhat)))
    return float(a), float(b), float(r2), float(mae), int(m.sum())

def _per_prop_mean(m, col):
    g = m.dropna(subset=["filename", col]).groupby("filename")[col].mean().reset_index()
    return g.rename(columns={col: f"{col}_mean"})

def _fit_hgbr(X,y):
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scl",StandardScaler()),
        ("hgb",HistGradientBoostingRegressor(learning_rate=0.08, max_iter=500, random_state=42))
    ]).fit(X,y)

def main():
    C=load_cfg()
    out=C["paths"]["outputs_tables_dir"]; os.makedirs(out, exist_ok=True)
    gcols=C["geometry_cols"]
    mpath=C["paths"]["master_parquet"]
    if not os.path.exists(mpath): raise SystemExit(f"master dataset not found: {mpath}")
    m=pd.read_parquet(mpath)

    # ---------------- Hover LOPO ----------------
    hov=m.dropna(subset=gcols+["prop_efficiency_mean"])
    if hov["filename"].nunique()>=3:
        r2, mae, rho, per=_lopo(hov, gcols, "prop_efficiency_mean")
        per.to_csv(os.path.join(out,"hover_predictions.csv"), index=False)
        pd.DataFrame([{"global_r2":r2,"avg_mae":mae,"spearman_rho":rho,"n_props":len(per)}]) \
          .to_csv(os.path.join(out,"prop_hover_metrics.csv"), index=False)
        print(f"Hover LOPO: R²={r2:.4f} | MAE={mae:.4f} | ρ={rho:.3f}")
    else:
        per=pd.DataFrame(columns=["filename","y_true","y_pred"])
        print("Hover: not enough props for LOPO (need ≥3).")

    # ---------------- Cruise LOPO ----------------
    have_cruise=("ld_cruise" in m.columns)
    if have_cruise:
        cr=m.dropna(subset=gcols+["ld_cruise"])
        if cr["filename"].nunique()>=3:
            r2c, maec, rhoc, perc=_lopo(cr, gcols, "ld_cruise")
            perc.to_csv(os.path.join(out,"cruise_predictions.csv"), index=False)
            pd.DataFrame([{"global_r2":r2c,"avg_mae":maec,"spearman_rho":rhoc,"n_props":len(perc)}]) \
              .to_csv(os.path.join(out,"wing_cruise_metrics.csv"), index=False)
            print(f"Cruise LOPO: R²={r2c:.4f} | MAE={maec:.4f} | ρ={rhoc:.3f}")
        else:
            print("Cruise: not enough props for LOPO (need ≥3).")

    # =======================
    # PRIOR CALIBRATIONS
    # =======================
    cal_rows=[]

    # experimental references (per prop means)
    exp_eta = _per_prop_mean(m, "prop_efficiency_mean") if "prop_efficiency_mean" in m.columns else None
    exp_ld  = _per_prop_mean(m, "ld_cruise") if have_cruise else None

    # --- Hover: BEMT avg vs experiment ---
    bemt_csv = os.path.join(out,"bemt_avg_prior.csv")
    if exp_eta is not None and os.path.exists(bemt_csv):
        b = pd.read_csv(bemt_csv)
        tab = b.merge(exp_eta.rename(columns={"prop_efficiency_mean_mean":"eta_exp"}),
                      on="filename", how="inner")
        fit = _fit_affine(tab["eta_bemt_mean"], tab["eta_exp"])
        if fit:
            a,bias,r2h,maeh,n = fit
            cal_rows.append({"mode":"hover","model":"bemt","a":a,"b":bias,"r2":r2h,"mae":maeh,"n":n})

    # --- Hover: CFD surrogate vs experiment ---
    cfd_path = os.path.join(out,"cfd_master.csv")
    if exp_eta is not None and os.path.exists(cfd_path):
        doe = pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename").dropna(subset=gcols)
        for c in gcols: doe[c]=pd.to_numeric(doe[c], errors="coerce")
        tested = doe.merge(exp_eta.rename(columns={"prop_efficiency_mean_mean":"eta_exp"}),
                           on="filename", how="inner")
        cfd = pd.read_csv(cfd_path)
        need = set(gcols+["eta_cfd"])
        if "mode" in cfd.columns and need.issubset(cfd.columns):
            vt = cfd[cfd["mode"]=="vtol"].dropna(subset=gcols+["eta_cfd"])
            if len(vt)>=10:
                mdl=_fit_hgbr(vt[gcols].to_numpy(float), vt["eta_cfd"].clip(0,1).to_numpy(float))
                pred = mdl.predict(tested[gcols].to_numpy(float))
                fit = _fit_affine(pred, tested["eta_exp"].to_numpy(float))
                if fit:
                    a,bias,r2h,maeh,n = fit
                    cal_rows.append({"mode":"hover","model":"cfd","a":a,"b":bias,"r2":r2h,"mae":maeh,"n":n})

    # --- Cruise: LL prior vs experiment ---
    if have_cruise and exp_ld is not None:
        ll_csv = os.path.join(out,"ll_cruise_prior.csv")
        if os.path.exists(ll_csv):
            ll = pd.read_csv(ll_csv)
            tab = ll.merge(exp_ld.rename(columns={"ld_cruise_mean":"ld_exp"}), on="filename", how="inner")
            fit = _fit_affine(tab["LD_ll"], tab["ld_exp"])
            if fit:
                a,bias,r2c,maec,n = fit
                cal_rows.append({"mode":"cruise","model":"ll","a":a,"b":bias,"r2":r2c,"mae":maec,"n":n})

    # --- Cruise: CFD surrogate vs experiment ---
    if have_cruise and exp_ld is not None and os.path.exists(cfd_path):
        doe = pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename").dropna(subset=gcols)
        for c in gcols: doe[c]=pd.to_numeric(doe[c], errors="coerce")
        tested = doe.merge(exp_ld.rename(columns={"ld_cruise_mean":"ld_exp"}), on="filename", how="inner")
        cfd = pd.read_csv(cfd_path)
        need = set(gcols+["LD_cfd"])
        if "mode" in cfd.columns and need.issubset(cfd.columns):
            cr = cfd[cfd["mode"]=="cruise"].dropna(subset=gcols+["LD_cfd"])
            if len(cr)>=10:
                mdl=_fit_hgbr(cr[gcols].to_numpy(float), cr["LD_cfd"].to_numpy(float))
                pred = mdl.predict(tested[gcols].to_numpy(float))
                fit = _fit_affine(pred, tested["ld_exp"].to_numpy(float))
                if fit:
                    a,bias,r2c,maec,n = fit
                    cal_rows.append({"mode":"cruise","model":"cfd","a":a,"b":bias,"r2":r2c,"mae":maec,"n":n})

    cal_df = pd.DataFrame(cal_rows, columns=["mode","model","a","b","r2","mae","n"])
    cal_path = C["paths"].get("prior_calibration_csv", os.path.join(out,"prior_calibration.csv"))
    cal_df.to_csv(cal_path, index=False)
    print("Wrote prior calibration →", cal_path)
    if not cal_df.empty:
        print(cal_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

if __name__ == "__main__":
    main()
