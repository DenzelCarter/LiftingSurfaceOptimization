# Scripts/02_quick_plots.py
# Generates sanity-check plots for both hover (eta vs. DL) and
# cruise (L/D vs. alpha) performance from the processed master dataset.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_utils import load_cfg

def generate_performance_plot(df, xlabel, ylabel, output_filename, title):
    """
    Creates and saves a performance plot for a given flight mode.
    """
    if df.empty:
        print(f"Skipping plot '{title}' because no data was found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort data to ensure lines are drawn correctly
    df = df.sort_values(['filename', 'op_point'])
    
    for filename, group in df.groupby('filename'):
        ax.plot(
            group['op_point'], 
            group['performance'], 
            label=filename.replace('.csv', ''), 
            marker='o', 
            linestyle='-',
            markersize=4, 
            alpha=0.8
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Optional: Keep title for quick checks, but it's not needed for the paper
    # ax.set_title(title)
    
    # Place the legend outside the plot area to avoid clutter.
    ax.legend(title="Lifting Surface", bbox_to_anchor=(1.04, 1), loc="upper left")
    
    fig.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust for external legend

    fig.savefig(output_filename)
    plt.close(fig)
    print(f"Wrote plot to: {output_filename}")

def main():
    C = load_cfg()
    P = C["paths"]
    
    input_path = P["master_parquet"]
    if not os.path.exists(input_path):
        raise SystemExit(f"Error: Master dataset not found at '{input_path}'")
        
    print(f"Reading data from: {input_path}")
    df_all = pd.read_parquet(input_path)
    
    os.makedirs(P["outputs_plots"], exist_ok=True)

    # --- 1. Split data by flight mode ---
    df_hover = df_all[df_all['flight_mode'] == 'hover'].copy()
    df_cruise = df_all[df_all['flight_mode'] == 'cruise'].copy()

    # --- 2. Generate Hover Plot (Efficiency vs. Disk Loading) ---
    hover_plot_path = os.path.join(P["outputs_plots"], "02_hover_performance.pdf")
    generate_performance_plot(
        df=df_hover,
        title="Hover Performance",
        xlabel="Disk Loading ($T/A$, N/m$^2$)",
        ylabel="Hover Efficiency ($\eta_{hover}$)",
        output_filename=hover_plot_path
    )

    # --- 3. Generate Cruise Plot (L/D vs. Angle of Attack) ---
    cruise_plot_path = os.path.join(P["outputs_plots"], "02_cruise_performance.pdf")
    generate_performance_plot(
        df=df_cruise,
        title="Cruise Performance",
        xlabel="Angle of Attack ($\alpha_r$, deg)",
        ylabel="Lift-to-Drag Ratio (L/D)",
        output_filename=cruise_plot_path
    )

if __name__ == "__main__":
    main()