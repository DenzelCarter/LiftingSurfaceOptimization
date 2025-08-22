# Unified driver: analytical priors for hover (BEMT avg) + cruise (Lifting-Line).
import os, sys, pandas as pd
from path_utils import load_cfg

# allow importing models.*
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path: sys.path.append(PROJ_ROOT)

from models.analytical.calculate_bemt import bemt_avg_for_doe
from models.analytical.calculate_lifting_line import ll_prior_for_doe

def load_doe(C):
    p = C["paths"]["doe_csv"]; g = C["geometry_cols"]
    df = pd.read_csv(p)[["filename"]+g].drop_duplicates("filename")
    for c in g: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=g).reset_index(drop=True)

def main():
    C      = load_cfg()
    tools  = C["paths"]["tools_dir"]
    os.makedirs(tools, exist_ok=True)
    doe    = load_doe(C)
    llcfg  = C["cruise_ll"]
    # Cruise prior (LL)
    ll = ll_prior_for_doe(
        doe_df=doe,
        target_CL=float(llcfg["target_CL"]),
        airfoil_dir=C["paths"]["airfoil_dir"] if bool(llcfg.get("use_airfoil_polars", True)) else None,
        CD0=float(llcfg["CD0"]),
        K2=float(llcfg["K2"]),
        e_base=float(llcfg["e_base"]),
        e_taper_k=float(llcfg["e_taper_k"]),
    )
    ll.to_csv(os.path.join(tools, "ll_cruise_prior.csv"), index=False)
    print("Wrote ll_cruise_prior.csv")
    # Hover prior (BEMT avg)
    try:
        bemt = bemt_avg_for_doe(doe_df=doe, airfoil_dir=C["paths"]["airfoil_dir"])
        bemt["eta_bemt_mean"] = bemt["eta_bemt_mean"].clip(0,1)
        bemt.to_csv(os.path.join(tools, "bemt_avg_prior.csv"), index=False)
        print("Wrote bemt_avg_prior.csv")
    except Exception as e:
        print(f"Skip BEMT avg prior (reason: {e})")

if __name__ == "__main__":
    main()
