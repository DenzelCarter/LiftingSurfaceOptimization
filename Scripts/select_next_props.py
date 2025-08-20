# select_next_props.py
# Recommend next DOE props to test using exploit + explore.
# ▶️ Run directly (no CLI args). Paths are relative to this file.

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# CONFIG
# =========================
K_RECOMMEND = 6

# Selection blend: composite = w_exploit * score_pred + w_explore * score_dist
W_EXPLOIT = 0.6
W_EXPLORE = 0.4

# Use BEMT prior (averaged across rpm bins) and calibrate it on tested props?
USE_BEMT_PRIOR = True
BEMT_WEIGHT    = 0.5   # how much BEMT contributes inside the exploit score (0..1)

# Which DOE rows to allow:
PREFER_PLAN_02 = True
INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS = True

PLAN_02_BOUNDS = {
    'AR':            (6.0, 10.0),
    'lambda':        (0.5, 1.0),
    'aoaRoot (deg)': (10.0, 20.0),
    'aoaTip (deg)':  (3.0, 8.0),
}

# Feature weights in standardized space
FEATURE_WEIGHTS = {
    'AR': 1.0, 'lambda': 1.0, 'aoaRoot (deg)': 1.0, 'aoaTip (deg)': 1.0,
    'flexMod (GPA)': 1.0, 'twist_slope': 1.0,
}

# Diversity constraint (optional): enforce a minimum std-space distance between chosen recs
MIN_DIST_BETWEEN_RECS = 0.0   # set e.g. 0.5 to avoid near-duplicates among the K picks

# =========================
# Paths (relative to this file)
# =========================
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

DOE_FILES = [
    os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv'),
    os.path.join(TOOLS_DIR, 'doe_test_plan_02.csv'),
]
MASTER_PARQUET = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
BEMT_CSV       = os.path.join(TOOLS_DIR, 'bemt_predictions_final.csv')
OUTPUT_RECS    = os.path.join(TOOLS_DIR, 'next_props_recommendations.csv')

GEOM_COLS = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'flexMod (GPA)']

# =========================
# Helpers
# =========================
def load_doe(paths):
    frames = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['__plan'] = os.path.basename(p)
            frames.append(df)
        else:
            print(f"Warning: DOE file not found: {p}")
    if not frames:
        raise SystemExit("No DOE CSVs found.")
    df_all = pd.concat(frames, ignore_index=True)
    if 'twist_slope' not in df_all.columns:
        df_all['twist_slope'] = df_all['aoaTip (deg)'] - df_all['aoaRoot (deg)']
    return df_all

def within_bounds(row, bounds):
    for c, (lo, hi) in bounds.items():
        v = float(row[c])
        if not (lo <= v <= hi): return False
    return True

def filter_doe(df, prefer_plan_02=True, include_plan01_if_in_box=True, bounds=None):
    if prefer_plan_02:
        df2 = df[df['__plan'].str.contains('doe_test_plan_02', case=False, regex=False)].copy()
    else:
        df2 = pd.DataFrame(columns=df.columns)
    if include_plan01_if_in_box and bounds is not None:
        df1 = df[df['__plan'].str.contains('doe_test_plan_01', case=False, regex=False)].copy()
        df1 = df1[df1.apply(lambda r: within_bounds(r, bounds), axis=1)]
        df_out = pd.concat([df2, df1], ignore_index=True)
    else:
        df_out = df2 if prefer_plan_02 else df.copy()
    return df_out.drop_duplicates(subset=['filename']).reset_index(drop=True)

def load_tested_props(master_parquet):
    if not os.path.exists(master_parquet):
        print(f"Note: master parquet not found at {master_parquet}. Assuming none tested yet.")
        return set(), None
    m = pd.read_parquet(master_parquet)
    if 'filename' not in m.columns or 'prop_efficiency_mean' not in m.columns:
        print("Note: master parquet missing needed columns. Assuming none tested yet.")
        return set(), None
    # Build prop-level avg η from master (mirrors train_prop_eff.py)
    g = m.dropna(subset=['prop_efficiency_mean']).groupby('filename')
    table = g['prop_efficiency_mean'].mean().to_frame('eta_mean').reset_index()
    geom = m[['filename'] + GEOM_COLS].drop_duplicates('filename')
    table = table.merge(geom, on='filename', how='left').dropna(subset=GEOM_COLS)
    table['eta_mean'] = table['eta_mean'].clip(0,1)
    return set(map(str, table['filename'].unique())), table

def standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd

def z01(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)

def pairwise_min_dist(x, Y):
    if Y.size == 0:
        return float(np.linalg.norm(x, ord=2))
    d = np.sqrt(np.sum((Y - x[None, :])**2, axis=1))
    return float(np.min(d))

def greedy_maximin_indices(Xw, idx_tested, idx_pool, k, min_sep=0.0):
    selected = []
    X_tested = Xw[idx_tested, :] if len(idx_tested) > 0 else np.zeros((0, Xw.shape[1]))
    pool = list(idx_pool)
    for _ in range(k):
        if not pool: break
        best_i, best_score = None, -1.0
        for i in pool:
            dmin = pairwise_min_dist(Xw[i, :], X_tested)
            if dmin > best_score:
                best_i, best_score = i, dmin
        if best_i is None: break
        selected.append((best_i, best_score))
        X_tested = Xw[best_i, :][None, :] if X_tested.size == 0 else np.vstack([X_tested, Xw[best_i, :]])
        pool.remove(best_i)
        if min_sep > 0.0:
            pool = [j for j in pool if np.linalg.norm(Xw[j,:] - Xw[best_i,:]) >= min_sep]
    return selected

def ridge_pred(train_X, train_y, test_X):
    n_train = train_X.shape[0]
    cv_k = min(5, max(3, n_train)) if n_train > 2 else 3
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  RidgeCV(alphas=np.logspace(-6, 3, 20), cv=cv_k, scoring='neg_mean_absolute_error'))
    ])
    model.fit(train_X, train_y)
    return model.predict(test_X)

def bemt_avg_by_prop(bemt_csv):
    if not os.path.exists(bemt_csv):
        return None
    b = pd.read_csv(bemt_csv)
    if 'filename' not in b.columns or 'prop_eff_bemt' not in b.columns:
        return None
    # average across rpm bins
    t = b.groupby('filename')['prop_eff_bemt'].mean().clip(0,1).to_frame('eta_bemt_mean').reset_index()
    return t

def calibrate_bemt_on_tested(bemt_table, tested_table):
    # linear fit y_true = a * y_bemt + b  (robust enough for small N)
    x = []
    y = []
    merged = tested_table[['filename','eta_mean']].merge(bemt_table, on='filename', how='inner')
    if merged.empty:
        return None
    x = merged['eta_bemt_mean'].to_numpy(float)
    y = merged['eta_mean'].to_numpy(float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def main():
    # 1) Load DOE & filter
    doe_all = load_doe(DOE_FILES)
    doe = filter_doe(
        doe_all,
        prefer_plan_02=PREFER_PLAN_02,
        include_plan01_if_in_box=INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS,
        bounds=PLAN_02_BOUNDS
    )

    # 2) Tested props + their avg η table
    tested_set, tested_table = load_tested_props(MASTER_PARQUET)

    # 3) Geometry matrix (+ twist_slope), standardize & weight
    cols = GEOM_COLS + ['twist_slope']
    for c in cols:
        if c not in doe.columns:
            raise SystemExit(f"DOE missing required column: '{c}'")
    X = doe[cols].astype(float).to_numpy()
    Xz, mu, sd = standardize(X)
    w = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in cols], float)
    Xw = Xz * w[None, :]

    filenames = doe['filename'].astype(str).tolist()
    idx_all = np.arange(len(filenames))
    idx_tested = [i for i in idx_all if filenames[i] in tested_set]
    idx_pool   = [i for i in idx_all if filenames[i] not in tested_set]

    if len(idx_pool) == 0:
        print("All DOE candidates (after filtering) appear to be tested. Nothing to recommend.")
        return

    # 4) Exploitation score (predicted avg-η)
    pred_eta = np.full(len(doe), np.nan, float)

    if tested_table is not None and len(tested_table) >= 3:
        # train ridge on tested props (geometry -> avg η)
        Xt = tested_table[GEOM_COLS].to_numpy(float)
        yt = tested_table['eta_mean'].to_numpy(float)
        # predict for ALL rows in DOE (including tested; we'll only use pool)
        pred_eta[:] = ridge_pred(Xt, yt, doe[GEOM_COLS].to_numpy(float))
    else:
        print("Not enough tested props to train a model; exploitation will rely on BEMT or be disabled.")

    # Optional BEMT prior (mean across rpm) + calibration on tested props
    eta_bemt = np.full(len(doe), np.nan, float)
    if USE_BEMT_PRIOR:
        bemt_tbl = bemt_avg_by_prop(BEMT_CSV)
        if bemt_tbl is not None:
            tmp = doe[['filename']].merge(bemt_tbl, on='filename', how='left')
            eta_bemt = tmp['eta_bemt_mean'].to_numpy(float)
            # calibrate on tested props if possible
            if tested_table is not None and len(tested_table) >= 3:
                fit = calibrate_bemt_on_tested(bemt_tbl, tested_table)
                if fit is not None:
                    a, b = fit
                    eta_bemt = a * eta_bemt + b
            eta_bemt = np.clip(eta_bemt, 0.0, 1.0)

    # Combine model + BEMT inside the exploit score (when both exist)
    exploit_raw = np.where(
        np.isfinite(pred_eta) & np.isfinite(eta_bemt),
        (1.0 - BEMT_WEIGHT) * pred_eta + BEMT_WEIGHT * eta_bemt,
        np.where(np.isfinite(pred_eta), pred_eta,
                 np.where(np.isfinite(eta_bemt), eta_bemt, 0.5))
    )
    exploit_score = z01(exploit_raw)

    # 5) Exploration score (distance to tested set in std-weighted space)
    X_tested = Xw[idx_tested, :] if len(idx_tested) > 0 else np.zeros((0, Xw.shape[1]))
    dist_to_tested = np.array([pairwise_min_dist(Xw[i,:], X_tested) for i in idx_all], float)
    explore_score  = z01(dist_to_tested)

    # 6) Composite score on POOL only
    composite = W_EXPLOIT * exploit_score + W_EXPLORE * explore_score

    # 7) Greedy pick with optional min separation among the K
    pool_sorted = sorted(idx_pool, key=lambda i: composite[i], reverse=True)
    chosen = []
    for i in pool_sorted:
        if len(chosen) >= K_RECOMMEND: break
        if MIN_DIST_BETWEEN_RECS <= 0:
            chosen.append(i); continue
        ok = True
        for j in chosen:
            if np.linalg.norm(Xw[i,:] - Xw[j,:]) < MIN_DIST_BETWEEN_RECS:
                ok = False; break
        if ok: chosen.append(i)

    # If we enforced spacing and got fewer than K, top up by pure maximin
    if len(chosen) < K_RECOMMEND:
        remaining = [i for i in idx_pool if i not in chosen]
        added = greedy_maximin_indices(Xw, idx_tested + chosen, remaining, K_RECOMMEND - len(chosen), min_sep=0.0)
        chosen += [i for (i, _) in added]

    # 8) Build output
    out_rows = []
    for rank, i in enumerate(chosen, start=1):
        row = {
            'rank': rank,
            'filename': filenames[i],
            'source_plan': doe.loc[i, '__plan'],
            'pred_eta_mean': float(np.clip(pred_eta[i], 0, 1)) if np.isfinite(pred_eta[i]) else np.nan,
            'bemt_eta_mean_cal': float(eta_bemt[i]) if np.isfinite(eta_bemt[i]) else np.nan,
            'exploit_score': float(exploit_score[i]),
            'explore_score': float(explore_score[i]),
            'composite_score': float(composite[i]),
            'min_dist_std_space': float(dist_to_tested[i]),
        }
        for c in GEOM_COLS + ['twist_slope']:
            row[c] = float(doe.loc[i, c])
        out_rows.append(row)
    out = pd.DataFrame(out_rows).sort_values('rank').reset_index(drop=True)

    # 9) Save & print
    os.makedirs(TOOLS_DIR, exist_ok=True)
    out.to_csv(OUTPUT_RECS, index=False)

    print("\nRecommended next props to test (exploit+explore):")
    view_cols = ['rank','filename','source_plan','composite_score','exploit_score','explore_score','min_dist_std_space','pred_eta_mean','bemt_eta_mean_cal'] + GEOM_COLS + ['twist_slope']
    with pd.option_context('display.max_columns', None):
        print(out[view_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved recommendations to: {OUTPUT_RECS}")

if __name__ == "__main__":
    main()
