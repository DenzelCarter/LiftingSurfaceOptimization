# select_next_props.py
# Recommend next DOE props to test using greedy maximin in geometry space.
# Run directly with the VS Code "Run Python File" ▶️ (no command-line args required).

import os
import numpy as np
import pandas as pd

# =========================
# CONFIG — edit as needed
# =========================
K_RECOMMEND = 6  # how many props to recommend

# Which DOE rows to allow:
PREFER_PLAN_02 = True
INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS = True  # only include plan_01 if inside plan_02 box

# Plan-02 bounds (your print-friendly constraints)
PLAN_02_BOUNDS = {
    'AR':        (6.0, 10.0),
    'lambda':    (0.5, 1.0),
    'aoaRoot (deg)': (10.0, 20.0),
    'aoaTip (deg)':  (3.0, 8.0),
}

# Optional feature weights (in standardized space). 1.0 = neutral.
FEATURE_WEIGHTS = {
    'AR': 1.0,
    'lambda': 1.0,
    'aoaRoot (deg)': 1.0,
    'aoaTip (deg)': 1.0,
    'flexMod (GPA)': 1.0,
    'twist_slope': 1.0,  # tip-root AoA
}

# =========================
# Paths (relative to this file, so ▶️ works)
# =========================
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)              # .../ICRA_2026_Optimization
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')

DOE_FILES = [
    os.path.join(TOOLS_DIR, 'doe_test_plan_01.csv'),
    os.path.join(TOOLS_DIR, 'doe_test_plan_02.csv'),
]
MASTER_PARQUET = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
OUTPUT_RECS    = os.path.join(TOOLS_DIR, 'next_props_recommendations.csv')

# =========================
# Columns / helpers
# =========================
GEOM_COLS = ['AR', 'lambda', 'aoaRoot (deg)', 'aoaTip (deg)', 'flexMod (GPA)']

def load_doe(paths):
    frames = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['__plan'] = os.path.basename(p)  # track source plan
            frames.append(df)
        else:
            print(f"Warning: DOE file not found: {p}")
    if not frames:
        raise SystemExit("No DOE CSVs found.")
    df_all = pd.concat(frames, ignore_index=True)

    # add twist_slope feature for diversity
    if 'twist_slope' not in df_all.columns:
        if all(c in df_all.columns for c in ['aoaRoot (deg)', 'aoaTip (deg)']):
            df_all['twist_slope'] = df_all['aoaTip (deg)'] - df_all['aoaRoot (deg)']
        else:
            raise SystemExit("DOE is missing 'aoaRoot (deg)'/'aoaTip (deg)' columns.")
    return df_all

def within_bounds(row, bounds):
    for c, (lo, hi) in bounds.items():
        v = float(row[c])
        if not (lo <= v <= hi):
            return False
    return True

def filter_doe(df, prefer_plan_02=True, include_plan01_if_in_box=True, bounds=None):
    """Keep plan_02 rows. Optionally include plan_01 rows only if inside the plan_02 bounds."""
    if prefer_plan_02:
        df2 = df[df['__plan'].str.contains('doe_test_plan_02', case=False, regex=False)].copy()
    else:
        df2 = pd.DataFrame(columns=df.columns)

    if include_plan01_if_in_box and bounds is not None:
        df1 = df[df['__plan'].str.contains('doe_test_plan_01', case=False, regex=False)].copy()
        mask = df1.apply(lambda r: within_bounds(r, bounds), axis=1)
        df1_in = df1.loc[mask]
        df_out = pd.concat([df2, df1_in], ignore_index=True)
    else:
        df_out = df2 if prefer_plan_02 else df.copy()

    # Deduplicate by filename if the same shows up (shouldn't, but just in case)
    df_out = df_out.drop_duplicates(subset=['filename'], keep='first').reset_index(drop=True)
    return df_out

def load_tested_props(master_parquet):
    if not os.path.exists(master_parquet):
        print(f"Note: master parquet not found at {master_parquet}. Assuming none tested yet.")
        return set()
    m = pd.read_parquet(master_parquet)
    if 'filename' not in m.columns:
        print("Note: master parquet has no 'filename' column. Assuming none tested yet.")
        return set()
    return set(map(str, m['filename'].unique()))

def standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd

def pairwise_min_dist(x, Y):
    if Y.size == 0:
        return float(np.linalg.norm(x, ord=2))
    d = np.sqrt(np.sum((Y - x[None, :])**2, axis=1))
    return float(np.min(d))

def greedy_maximin(X_all, idx_tested, idx_pool, k):
    selected = []
    X_tested = X_all[idx_tested, :] if len(idx_tested) > 0 else np.zeros((0, X_all.shape[1]))
    pool = list(idx_pool)

    for _ in range(k):
        if not pool:
            break
        best_i, best_score = None, -1.0
        for i in pool:
            s = pairwise_min_dist(X_all[i, :], X_tested)
            if s > best_score:
                best_i, best_score = i, s
        if best_i is None:
            break
        selected.append((best_i, best_score))
        # update tested set
        X_tested = X_all[best_i, :][None, :] if X_tested.size == 0 else np.vstack([X_tested, X_all[best_i, :]])
        pool.remove(best_i)
    return selected

def main():
    # 1) Load & filter DOE
    doe_all = load_doe(DOE_FILES)
    doe = filter_doe(
        doe_all,
        prefer_plan_02=PREFER_PLAN_02,
        include_plan01_if_in_box=INCLUDE_PLAN_01_IF_WITHIN_PLAN_02_BOUNDS,
        bounds=PLAN_02_BOUNDS
    )

    # 2) Detect tested props from master dataset
    tested = load_tested_props(MASTER_PARQUET)

    # 3) Geometry matrix (with twist_slope)
    cols = GEOM_COLS + ['twist_slope']
    for c in cols:
        if c not in doe.columns:
            raise SystemExit(f"DOE missing required column: '{c}'")
    X = doe[cols].astype(float).to_numpy()
    Xz, mu, sd = standardize(X)

    # 4) Apply feature weights in standardized space
    w = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in cols], dtype=float)
    Xw = Xz * w[None, :]

    # 5) Split tested vs pool
    filenames = list(map(str, doe['filename'].tolist()))
    idx_all = np.arange(len(filenames))
    idx_tested = [i for i in idx_all if filenames[i] in tested]
    idx_pool   = [i for i in idx_all if filenames[i] not in tested]

    if len(idx_pool) == 0:
        print("All DOE candidates (after filtering) appear to be tested. Nothing to recommend.")
        return

    # 6) Greedy maximin selection
    recs = greedy_maximin(Xw, idx_tested, idx_pool, K_RECOMMEND)

    # 7) Build output table
    out_rows = []
    for rank, (i, dist) in enumerate(recs, start=1):
        row = {
            'rank': rank,
            'filename': filenames[i],
            'source_plan': doe.loc[i, '__plan'],
            'min_dist_std_space': dist,
        }
        for c in cols:
            row[c] = float(doe.loc[i, c])
        out_rows.append(row)
    out = pd.DataFrame(out_rows)

    # 8) Save & print
    os.makedirs(TOOLS_DIR, exist_ok=True)
    out.to_csv(OUTPUT_RECS, index=False)

    print("\nRecommended next props to test (greedy maximin):")
    # Pretty print a compact view
    view_cols = ['rank','filename','source_plan','min_dist_std_space'] + cols
    print(out[view_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved recommendations to: {OUTPUT_RECS}")

if __name__ == "__main__":
    main()
