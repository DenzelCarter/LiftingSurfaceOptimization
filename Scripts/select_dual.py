# Candidate selection for dual-mode LS: blend hover (ML) + cruise (priors) with exploit/explore.
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from path_utils import load_cfg

def ridge_pred(Xtr, ytr, Xte):
    n = Xtr.shape[0]; cvk = min(5, max(3, n)) if n>2 else 3
    model = Pipeline([("scl", StandardScaler()), ("rid", RidgeCV(alphas=np.logspace(-6,3,20), cv=cvk, scoring="neg_mean_absolute_error"))])
    model.fit(Xtr, ytr); return model.predict(Xte)

def z01(v):
    v = np.asarray(v,float); lo,hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi-lo<1e-12: return np.zeros_like(v)
    return (v-lo)/(hi-lo)

def main():
    C = load_cfg(); tools=C["paths"]["tools_dir"]; gcols=C["geometry_cols"]; os.makedirs(tools, exist_ok=True)
    candidates = pd.read_csv(C["paths"]["doe_csv"])[["filename"]+gcols].drop_duplicates("filename")
    for c in gcols: candidates[c] = pd.to_numeric(candidates[c], errors="coerce")
    candidates = candidates.dropna(subset=gcols).reset_index(drop=True)

    master_path = os.path.join(tools, "master_dataset.parquet")
    master = pd.read_parquet(master_path) if os.path.exists(master_path) else None

    # Hover predictions from experiments-only model
    pred_hover = np.full(len(candidates), np.nan)
    tested_set = set()
    if master is not None and "prop_efficiency_mean" in master.columns:
        train = master.dropna(subset=gcols+["prop_efficiency_mean"]).rename(columns={"prop_efficiency_mean":"eta_hover"})
        if len(train)>=3:
            pred_hover[:] = ridge_pred(train[gcols].to_numpy(float), train["eta_hover"].to_numpy(float), candidates[gcols].to_numpy(float))
            tested_set = set(train["filename"].astype(str).tolist())

    # Cruise priors
    ll_csv  = os.path.join(tools,"ll_cruise_prior.csv")
    cfd_cru = os.path.join(tools,"cfd_prior_cruise.csv")
    ld_ll = candidates[["filename"]].merge(pd.read_csv(ll_csv), on="filename", how="left")["LD_ll"].to_numpy(float) if os.path.exists(ll_csv) else np.full(len(candidates), np.nan)
    ld_cf = candidates[["filename"]].merge(pd.read_csv(cfd_cru), on="filename", how="left")["LD_cfd"].to_numpy(float) if os.path.exists(cfd_cru) else np.full(len(candidates), np.nan)

    # Hover priors (optional)
    bemt_csv = os.path.join(tools,"bemt_avg_prior.csv")
    cfd_vtol = os.path.join(tools,"cfd_prior_vtol.csv")
    eta_bemt = candidates[["filename"]].merge(pd.read_csv(bemt_csv), on="filename", how="left")["eta_bemt_mean"].to_numpy(float) if os.path.exists(bemt_csv) else np.full(len(candidates), np.nan)
    eta_cfd  = candidates[["filename"]].merge(pd.read_csv(cfd_vtol), on="filename", how="left")["eta_cfd_vtol"].to_numpy(float) if os.path.exists(cfd_vtol) else np.full(len(candidates), np.nan)

    # Weights
    w_hover   = float(C["selection"]["w_hover"]); w_cruise = 1.0 - w_hover
    use_priors= bool(C["selection"]["use_priors"]); pw = C["selection"]["priors_weights"]

    # Cruise raw score
    cruise_raw = np.zeros(len(candidates))
    for i in range(len(candidates)):
        s=w=0.0
        if np.isfinite(ld_ll[i]):  s += pw["cruise"]["ll"]  * ld_ll[i];  w += pw["cruise"]["ll"]
        if np.isfinite(ld_cf[i]):  s += pw["cruise"]["cfd"] * ld_cf[i];  w += pw["cruise"]["cfd"]
        cruise_raw[i] = s/max(w,1e-9) if (use_priors and w>0) else (ld_ll[i] if np.isfinite(ld_ll[i]) else ld_cf[i])

    # Hover raw score
    hover_raw = np.zeros(len(candidates))
    for i in range(len(candidates)):
        if np.isfinite(pred_hover[i]): hover_raw[i] = pred_hover[i]
        elif use_priors and (np.isfinite(eta_bemt[i]) or np.isfinite(eta_cfd[i])):
            s=w=0.0
            if np.isfinite(eta_bemt[i]): s += pw["hover"]["bemt"]*eta_bemt[i]; w += pw["hover"]["bemt"]
            if np.isfinite(eta_cfd[i]):  s += pw["hover"]["cfd"] *eta_cfd[i];  w += pw["hover"]["cfd"]
            hover_raw[i] = s/max(w,1e-9) if w>0 else np.nan
        else:
            hover_raw[i] = np.nan

    hover_s  = z01(hover_raw)
    cruise_s = z01(cruise_raw)
    dms = np.where(np.isfinite(hover_s) & np.isfinite(cruise_s),
                   w_hover*hover_s + w_cruise*cruise_s,
                   np.where(np.isfinite(hover_s), hover_s, np.where(np.isfinite(cruise_s), cruise_s, 0.0)))

    # Exploration distance
    X = candidates[gcols].to_numpy(float)
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd<1e-12]=1.0
    Xz = (X-mu)/sd
    tested = list(tested_set)
    explore = np.zeros(len(candidates))
    if tested and master is not None:
        tested_df = candidates[candidates["filename"].astype(str).isin(tested)]
        Xt = tested_df[gcols].to_numpy(float)
        Xt = (Xt - mu)/sd
        for i in range(len(candidates)):
            d = np.sqrt(((Xt - Xz[i,:])**2).sum(axis=1)) if len(Xt) else np.linalg.norm(Xz[i,:])
            explore[i] = d.min() if len(Xt) else d
    explore_s = z01(explore)

    comp = float(C["selection"]["w_exploit"])*z01(dms) + (1.0-float(C["selection"]["w_exploit"]))*explore_s

    # spacing
    chosen=[]
    min_sep = float(C["selection"]["min_dist_std"])
    for i in np.argsort(-comp):
        if len(chosen) >= int(C["selection"]["K"]): break
        ok=True
        for j in chosen:
            if np.linalg.norm(Xz[i,:] - Xz[j,:]) < min_sep:
                ok=False; break
        if ok: chosen.append(i)

    rows=[]
    for rank,i in enumerate(chosen, start=1):
        row = {"rank":rank, "filename": candidates["filename"].iloc[i],
               "composite_score": float(comp[i]), "dual_mode_score": float(dms[i]),
               "hover_score": float(hover_s[i]), "cruise_score": float(cruise_s[i]),
               "explore_score": float(explore_s[i])}
        for c in gcols: row[c] = float(candidates[c].iloc[i])
        row.update({
            "pred_eta_hover": float(pred_hover[i]) if np.isfinite(pred_hover[i]) else np.nan,
            "LD_ll": float(ld_ll[i]) if np.isfinite(ld_ll[i]) else np.nan,
            "LD_cfd": float(ld_cf[i]) if np.isfinite(ld_cf[i]) else np.nan
        })
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("rank")
    out.to_csv(os.path.join(tools,"next_props_recommendations.csv"), index=False)

    # pretty print
    view = ["rank","filename","composite_score","hover_score","cruise_score","explore_score","dual_mode_score"] + gcols
    with pd.option_context("display.max_columns", None):
        print(out[view].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved recommendations to: {os.path.join(tools,'next_props_recommendations.csv')}")

if __name__ == "__main__":
    main()
