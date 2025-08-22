# Train geometry-only models: hover η̄ (VTOL) and cruise L/D from experiments only.
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from path_utils import load_cfg

def fit_model(X, y):
    cvk = min(5, max(3, len(y))) if len(y)>2 else 3
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler()),
        ("rid", RidgeCV(alphas=np.logspace(-6,3,40), cv=cvk, scoring="neg_mean_absolute_error"))
    ])
    pipe.fit(X, y); return pipe

def lopo(table, gcols, ycol):
    rows=[]; yT=[]; yP=[]
    for fname, grp in table.groupby("filename"):
        tr = table.loc[table["filename"]!=fname]
        te = table.loc[table["filename"]==fname]
        if len(tr)<3: continue
        m = fit_model(tr[gcols].to_numpy(float), tr[ycol].to_numpy(float))
        yh = float(np.clip(m.predict(te[gcols].to_numpy(float))[0], 0, 1)) if ycol=="eta_hover" else float(m.predict(te[gcols].to_numpy(float))[0])
        y  = float(te[ycol].iloc[0])
        rows.append({"filename":fname, "y_true":y, "y_pred":yh, "abs_err":abs(y-yh)})
        yT.append(y); yP.append(yh)
    if not yT: return np.nan, np.nan, pd.DataFrame()
    return r2_score(yT,yP), mean_absolute_error(yT,yP), pd.DataFrame(rows)

def main():
    C = load_cfg(); tools=C["paths"]["tools_dir"]; gcols=C["geometry_cols"]
    mpath = os.path.join(tools, "master_dataset.parquet")
    if not os.path.exists(mpath): raise SystemExit("Run process_data.py first.")
    master = pd.read_parquet(mpath)

    # Hover (η̄)
    if "prop_efficiency_mean" in master.columns:
        tab = master.dropna(subset=gcols+["prop_efficiency_mean"]).rename(columns={"prop_efficiency_mean":"eta_hover"})
        r2, mae, per = lopo(tab, gcols, "eta_hover")
        per.to_csv(os.path.join(tools,"prop_hover_predictions.csv"), index=False)
        pd.DataFrame([{"global_r2":r2,"avg_mae":mae,"n":len(per)}]).to_csv(os.path.join(tools,"prop_hover_metrics.csv"), index=False)
        print(f"Hover LOPO: R2={r2:.3f} MAE={mae:.4f} N={len(per)}")
    else:
        print("No prop_efficiency_mean in master; skip hover training.")

    # Cruise (L/D)
    if "ld_cruise" in master.columns:
        tab = master.dropna(subset=gcols+["ld_cruise"])
        model = fit_model(tab[gcols].to_numpy(float), tab["ld_cruise"].to_numpy(float))
        yh = model.predict(tab[gcols].to_numpy(float))
        r2 = r2_score(tab["ld_cruise"], yh) if len(tab)>=2 else np.nan
        mae= mean_absolute_error(tab["ld_cruise"], yh) if len(tab)>=1 else np.nan
        out = tab[["filename"]].copy(); out["ld_true"]=tab["ld_cruise"]; out["ld_pred"]=yh
        out.to_csv(os.path.join(tools,"wing_cruise_predictions.csv"), index=False)
        pd.DataFrame([{"global_r2":r2,"avg_mae":mae,"n":len(out)}]).to_csv(os.path.join(tools,"wing_cruise_metrics.csv"), index=False)
        print(f"Cruise fit: R2={r2:.3f} MAE={mae:.4f} N={len(out)}")
    else:
        print("No ld_cruise in master; skip cruise training.")

if __name__ == "__main__":
    main()
