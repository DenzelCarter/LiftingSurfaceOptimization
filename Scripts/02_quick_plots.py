#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from path_utils import load_cfg

def pick_cols(df: pd.DataFrame):
    """Robustly finds the required column names for plotting."""
    rpm_candidates = ["rpm_bin_center", "rpm"]
    eff_candidates = ["prop_efficiency", "prop_efficiency_mean"]
    prop_candidates = ["filename", "prop_name"]

    def _find(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    rpm_col = _find(rpm_candidates)
    eff_col = _find(eff_candidates)
    prop_col = _find(prop_candidates)

    if not all([rpm_col, eff_col, prop_col]):
        raise KeyError(f"Could not find all required columns. Found: RPM='{rpm_col}', Eff='{eff_col}', Prop='{prop_col}'")
    return rpm_col, eff_col, prop_col

def main():
    C = load_cfg()
    P = C["paths"]
    OUTDIR = P["outputs_plots_dir"]
    
    input_path = P["master_parquet"]
    if not os.path.exists(input_path):
        raise SystemExit(f"Error: Master dataset not found at '{input_path}'")
        
    print(f"Reading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    rpm_col, eff_col, prop_col = pick_cols(df)
        
    df = df.sort_values([prop_col, rpm_col])
    os.makedirs(OUTDIR, exist_ok=True)

    # --- Define a set of unique visual identifiers ---
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linestyles = ['-', '--', ':', '-.']

    # --- Generate Combined Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use enumerate to get a unique index for each propeller
    for i, (name, group) in enumerate(df.groupby(prop_col)):
        # Cycle through markers and linestyles to create unique visuals
        marker = markers[i % len(markers)]
        linestyle = linestyles[(i // len(markers)) % len(linestyles)]
        
        ax.plot(
            group[rpm_col], 
            group[eff_col], 
            label=name, 
            marker=marker, 
            linestyle=linestyle,
            markersize=5, 
            alpha=0.8
        )
    
    ax.set_xlabel("RPM Bin Center")
    ax.set_ylabel("Propeller Efficiency")
    ax.set_title("Propeller Efficiency vs. RPM")
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    ax.legend(title="Propeller", bbox_to_anchor=(1.04, 1), loc="upper left")
    
    out_all_path = os.path.join(OUTDIR, "rpm_vs_efficiency_all_props.pdf")
    # Adjust layout to make space for the legend
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig.savefig(out_all_path)
    plt.close(fig)
    print(f"Wrote combined plot to: {out_all_path}")

if __name__ == "__main__":
    main()