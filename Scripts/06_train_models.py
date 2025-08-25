# Train geometry-only Ridge models:
#   - hover:   prop_efficiency_mean
#   - cruise:  ld_cruise
# Saves LOPO parity + metrics into Experiment/outputs/tables/

import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from path_utils import load_cfg

def _fit(X,y):
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
        mdl=_fit(tr[gcols].to_numpy(float), tr[target].to_numpy(float))
        p = mdl.predict(te[gcols].to_numpy(float))
        rows.append({"filename":f,"y_true":float(te[target].mean()),"y_pred":float(p.mean())})
        ytrue.append(te[target].mean()); yhat.append(p.mean())
    if not rows: return np.nan, np.nan, np.nan, pd.DataFrame(columns=["filename","y_true","y_pred"])
    y=np.array(ytrue,float); yb=np.array(yhat,float)
    r2 = 1 - np.sum((y-yb)**2)/(np.sum((y-y.mean())**2)+1e-12)
    mae=float(np.mean(np.abs(y-yb))); rho=float(spearmanr(y,yb).statistic)
    return r2, mae, rho, pd.DataFrame(rows).sort_values("filename")

def main():
    C=load_cfg()
    out=C["paths"]["outputs_tables_dir"]; os.makedirs(out, exist_ok=True)
    gcols=C["geometry_cols"]
    mpath=C["paths"]["master_parquet"]
    if not os.path.exists(mpath): raise SystemExit(f"master dataset not found: {mpath}")
    m=pd.read_parquet(mpath)

    # Hover
    hov=m.dropna(subset=gcols+["prop_efficiency_mean"])
    if hov["filename"].nunique()>=3:
        r2, mae, rho, per=_lopo(hov, gcols, "prop_efficiency_mean")
        per.to_csv(os.path.join(out,"hover_predictions.csv"), index=False)
        pd.DataFrame([{"global_r2":r2,"avg_mae":mae,"spearman_rho":rho,"n_props":len(per)}]) \
          .to_csv(os.path.join(out,"prop_hover_metrics.csv"), index=False)
        print(f"Hover LOPO: R²={r2:.4f} | MAE={mae:.4f} | ρ={rho:.3f}")
    else:
        print("Hover: not enough props for LOPO (need ≥3).")

    # Cruise
    if "ld_cruise" in m.columns:
        cr=m.dropna(subset=gcols+["ld_cruise"])
        if cr["filename"].nunique()>=3:
            r2, mae, rho, per=_lopo(cr, gcols, "ld_cruise")
            per.to_csv(os.path.join(out,"cruise_predictions.csv"), index=False)
            pd.DataFrame([{"global_r2":r2,"avg_mae":mae,"spearman_rho":rho,"n_props":len(per)}]) \
              .to_csv(os.path.join(out,"wing_cruise_metrics.csv"), index=False)
            print(f"Cruise LOPO: R²={r2:.4f} | MAE={mae:.4f} | ρ={rho:.3f}")
        else:
            print("Cruise: not enough props for LOPO (need ≥3).")
    else:
        print("Cruise: ld_cruise not in master.")

if __name__ == "__main__":
    main()
