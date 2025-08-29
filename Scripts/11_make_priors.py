# 04_make_priors.py
# Convenience wrapper:
#   - runs 02_calculate_bemt.py and 03_calculate_lifting_line.py
#   - then writes a joined priors catalog for quick inspection:
#       Experiment/outputs/tables/priors_catalog.csv

import os, sys, subprocess, pandas as pd
from path_utils import load_cfg

HERE = os.path.dirname(os.path.abspath(__file__))
SCR_BEMT = os.path.join(HERE, "02_calculate_bemt.py")
SCR_LL   = os.path.join(HERE, "03_calculate_lifting_line.py")

def _run(pyfile):
    if not os.path.exists(pyfile):
        raise SystemExit(f"Missing script: {pyfile}")
    print(f"→ Running {os.path.basename(pyfile)}")
    subprocess.run([sys.executable, pyfile], check=True)

def main():
    C = load_cfg()
    tables_dir = C["paths"]["outputs_tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    # Run the two generators
    _run(SCR_BEMT)
    _run(SCR_LL)

    # Collect outputs if present
    p_bemt = os.path.join(tables_dir, "bemt_avg_prior.csv")
    p_ll   = os.path.join(tables_dir, "ll_cruise_prior.csv")

    dfb = pd.read_csv(p_bemt) if os.path.exists(p_bemt) else pd.DataFrame(columns=["filename","eta_bemt_mean"])
    dfl = pd.read_csv(p_ll)   if os.path.exists(p_ll)   else pd.DataFrame(columns=["filename","LD_ll"])

    # Join on filename if both exist; otherwise just save whichever exists
    if not dfb.empty and not dfl.empty:
        cat = dfb.merge(dfl, on="filename", how="outer")
    else:
        cat = dfb if not dfb.empty else dfl

    out_cat = os.path.join(tables_dir, "priors_catalog.csv")
    cat.to_csv(out_cat, index=False)
    print(f"✓ Wrote priors_catalog.csv with {len(cat)} rows → {out_cat}")

if __name__ == "__main__":
    main()
