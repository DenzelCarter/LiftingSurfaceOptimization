# Continuous optimization of geometry within config bounds for a dual-mode LS.
# Objective: Dual Mode Score (DMS) = w_hover*z(eta_hover) + (1-w_hover)*z(L/D_cruise)
# Hover: experiments-only Ridge model on geometry.
# Cruise: lifting-line closed-form (same params as make_priors.py).
#
# Writes: Experiment/outputs/tables/optimal_dual_params.csv

import os, numpy as np, pandas as pd
from math import pi
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from path_utils import load_cfg

# ----------------- lifting-line helpers (mirrors models/analytical/calculate_lifting_line.py) -----------------
def _oswald_from_taper(AR, lam, e_base, k):
    # modest penalty away from lam=1; clamp to plausible range
    e = e_base - k * (1.0 - lam)**2
    return float(np.clip(e, 0.70, 0.98))

def _CDp_from_params(CL, CD0, K2):
    # simple profile drag model if polars aren't used here
    return float(CD0 + K2 * CL**2)

def ll_LD(AR, lam, target_CL, CD0, K2, e_base, e_taper_k):
    e   = _oswald_from_taper(AR, lam, e_base, e_taper_k)
    CDp = _CDp_from_params(target_CL, CD0, K2)
    CD  = CDp + target_CL**2/(pi*AR*e)
    return float(target_CL/CD) if CD > 1e-12 else np.nan

# ----------------- hover model -----------------
def fit_hover_model(master_df, gcols):
    # uses prop_efficiency_mean (η̄) as target
    tab = master_df.dropna(subset=gcols + ["prop_efficiency_mean"]).copy()
    if len(tab) < 3:
        return None
    y = tab["prop_efficiency_mean"].to_numpy(float).clip(0, 1)
    X = tab[gcols].to_numpy(float)
    cvk = min(5, max(3, len(tab)))
    model = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler()),
        ("rid", RidgeCV(alphas=np.logspace(-6, 3, 40), cv=cvk, scoring="neg_mean_absolute_error"))
    ])
    model.fit(X, y)
    return model

# ----------------- sampling -----------------
def sample_box(bounds, n):
    rng = np.random.default_rng(42)
    cols = list(bounds.keys())
    X = np.zeros((n, len(cols)), float)
    for j, c in enumerate(cols):
        lo, hi = bounds[c]
        X[:, j] = rng.uniform(lo, hi, size=n)
    return cols, X

def refine_around(top_points, n_refine, sigma_frac=0.05, bounds=None):
    rng = np.random.default_rng(123)
    pts=[]
    for p in top_points:
        for _ in range(n_refine // len(top_points)):
            noise = rng.normal(0.0, sigma_frac, size=len(p)) * np.maximum(np.abs(p), 1.0)
            cand = p + noise
            # clamp to bounds
            if bounds:
                for i, (k,(lo,hi)) in enumerate(bounds.items()):
                    cand[i] = float(np.clip(cand[i], lo, hi))
            pts.append(cand)
    return np.array(pts, float)

def z01(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)

# ----------------- main -----------------
def main():
    C = load_cfg()
    tools = C["paths"]["tools_dir"]; os.makedirs(tools, exist_ok=True)
    gcols = C["geometry_cols"]  # ["AR","lambda","aoaRoot (deg)","aoaTip (deg)"]

    # bounds from config
    B = C.get("bounds", {})
    for k in gcols:
        if k not in B:
            raise SystemExit(f"Missing bound for '{k}' in config.yaml -> bounds")

    # weights from config
    w_hover = float(C["selection"]["w_hover"])
    w_cruise = 1.0 - w_hover

    # lifting-line params
    LL = C["cruise_ll"]
    target_CL = float(LL["target_CL"])
    CD0 = float(LL["CD0"]); K2 = float(LL["K2"])
    e_base = float(LL["e_base"]); e_taper_k = float(LL["e_taper_k"])

    # load experiments
    mpath = os.path.join(tools, "master_dataset.parquet")
    if not os.path.exists(mpath):
        raise SystemExit("Run Scripts/process_data.py first (master_dataset.parquet missing).")
    master = pd.read_parquet(mpath)

    # train hover model
    hover_model = fit_hover_model(master, gcols)
    if hover_model is None:
        raise SystemExit("Not enough experimental hover data to fit a model (need ≥3 rows).")

    # coarse sampling
    N1 = int(os.environ.get("DUAL_OPT_COARSE", "5000"))
    keys, X1 = sample_box({k: B[k] for k in gcols}, N1)
    df1 = pd.DataFrame(X1, columns=keys)

    # evaluate hover
    eta_pred = hover_model.predict(df1[gcols].to_numpy(float)).clip(0, 1)
    # evaluate cruise (LL)
    LD_pred = np.array([
        ll_LD(float(r["AR"]), float(r["lambda"]), target_CL, CD0, K2, e_base, e_taper_k)
        for _, r in df1.iterrows()
    ])

    # score
    s_hover  = z01(eta_pred)
    s_cruise = z01(LD_pred)
    dms1 = w_hover * s_hover + w_cruise * s_cruise

    # pick top seeds
    Kseed = min(20, max(5, N1 // 250))
    idx_top = np.argsort(-dms1)[:Kseed]
    seeds = X1[idx_top, :]

    # refine around seeds
    N2 = int(os.environ.get("DUAL_OPT_REFINE", "2000"))
    X2 = refine_around(seeds, n_refine=N2, sigma_frac=0.05, bounds={k: B[k] for k in gcols})
    df2 = pd.DataFrame(X2, columns=keys)

    eta2 = hover_model.predict(df2[gcols].to_numpy(float)).clip(0, 1)
    LD2  = np.array([
        ll_LD(float(r["AR"]), float(r["lambda"]), target_CL, CD0, K2, e_base, e_taper_k)
        for _, r in df2.iterrows()
    ])

    s_hover2  = z01(eta2)
    s_cruise2 = z01(LD2)
    dms2 = w_hover * s_hover2 + w_cruise * s_cruise2

    # combine pools and re-rank
    allX   = np.vstack([X1, X2])
    all_eta= np.concatenate([eta_pred, eta2])
    all_LD = np.concatenate([LD_pred, LD2])
    all_dms= np.concatenate([dms1, dms2])

    order = np.argsort(-all_dms)
    TOP = 25  # report top designs
    rows=[]
    for rank, i in enumerate(order[:TOP], start=1):
        rows.append({
            "rank": rank,
            "AR":       float(allX[i, 0]),
            "lambda":   float(allX[i, 1]),
            "aoaRoot (deg)": float(allX[i, 2]),
            "aoaTip (deg)":  float(allX[i, 3]),
            "pred_eta_hover": float(np.clip(all_eta[i], 0, 1)),
            "pred_LD_cruise": float(all_LD[i]),
            "dual_mode_score": float(all_dms[i]),
            "w_hover": w_hover,
        })
    out = pd.DataFrame(rows)

    out_path = os.path.join(tools, "optimal_dual_params.csv")
    out.to_csv(out_path, index=False)

    # Pretty print top-5
    view_cols = ["rank","dual_mode_score","pred_eta_hover","pred_LD_cruise","AR","lambda","aoaRoot (deg)","aoaTip (deg)"]
    with pd.option_context("display.max_columns", None):
        print(out.head(5)[view_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved top-{TOP} designs → {out_path}")
    # also echo the single best design clearly
    best = out.iloc[0]
    print("\nBest (within bounds):")
    print(f"  AR={best['AR']:.3f}, lambda={best['lambda']:.3f}, aoaRoot={best['aoaRoot (deg)']:.3f}°, aoaTip={best['aoaTip (deg)']:.3f}°")
    print(f"  Pred η̄={best['pred_eta_hover']:.3f}, Pred L/D={best['pred_LD_cruise']:.2f}, DMS={best['dual_mode_score']:.3f} (w_hover={w_hover:.2f})")

if __name__ == "__main__":
    main()
