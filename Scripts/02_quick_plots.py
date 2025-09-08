# Scripts/02_quick_plots.py
# Generates a performance summary with aligned x-axes for clear comparison.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

# --- Configuration Loading ---
def load_config() -> dict:
    """Loads config.yaml from the same directory as the script."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file not found. Ensure 'config.yaml' is in the same folder as this script.")

def main():
    C = load_config()
    P = C["paths"]
    script_dir = Path(__file__).parent

    # --- 1. Load Processed Data ---
    master_parquet_path = (script_dir / P["master_parquet"]).resolve()
    if not master_parquet_path.exists():
        raise SystemExit(f"Error: Master dataset not found at '{master_parquet_path}'. Please run 01_process_data.py first.")
        
    df_full = pd.read_parquet(master_parquet_path)
    
    # --- 2. Separate Data by Flight Mode ---
    df_hover = df_full[df_full['flight_mode'] == 'hover'].copy()
    df_cruise = df_full[df_full['flight_mode'] == 'cruise'].copy()

    if df_hover.empty and df_cruise.empty:
        raise SystemExit("No hover or cruise data found in the master dataset to plot.")

    # --- 3. Create Combined Figure with Two Subplots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- MODIFIED: Added sharex=True to align the x-axes ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Performance Summary of Initial Designs', fontsize=18, y=0.99)

    # --- Plot 1: Hover Performance ---
    if not df_hover.empty:
        print(f"Plotting hover performance for {len(df_hover)} data points.")
        
        scatter = sns.scatterplot(
            data=df_hover, x='aoa_root (deg)', y='performance',
            hue='op_speed', palette='viridis', alpha=0.7, s=50,
            ax=ax1, legend=False
        )
        ax1.set_title('Hover Performance vs. Geometry and RPM', fontsize=14)
        ax1.set_ylabel('Hover Efficiency (η_hover)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        norm = plt.Normalize(df_hover['op_speed'].min(), df_hover['op_speed'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.set_label('RPM', rotation=270, labelpad=15, fontsize=12)
        
        min_rpm = df_hover['op_speed'].min()
        median_rpm = df_hover['op_speed'].median()
        max_rpm = df_hover['op_speed'].max()
        cbar.set_ticks([min_rpm, median_rpm, max_rpm])
        cbar.set_ticklabels([f'{min_rpm:.0f} (Min)', f'{median_rpm:.0f} (Median)', f'{max_rpm:.0f} (Max)'])
        
    else:
        ax1.text(0.5, 0.5, "No Hover Data Found", ha='center', va='center', fontsize=14, alpha=0.5)
        ax1.set_title('Hover Performance', fontsize=14)

    # --- Plot 2: Cruise Performance ---
    if not df_cruise.empty:
        print(f"Plotting cruise performance for {len(df_cruise)} COMSOL data points.")
        sns.scatterplot(
            data=df_cruise, x='aoa_root (deg)', y='performance', 
            alpha=0.6, s=50, ax=ax2, color='seagreen'
        )
        ax2.set_title('Cruise Performance vs. Operation', fontsize=14)
        ax2.set_xlabel('Root Angle of Attack (deg)', fontsize=12)
        ax2.set_ylabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No Cruise Data Found", ha='center', va='center', fontsize=14, alpha=0.5)
        ax2.set_title('Cruise Performance', fontsize=14)

    # --- 4. Customize and Save the Plot ---
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "02_performance_summary.pdf"
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"\nSuccessfully generated performance summary plot.")
    print(f"Output saved to: '{plot_path}'")

if __name__ == "__main__":
    main()