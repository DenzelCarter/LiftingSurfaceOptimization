# Scripts/02_quick_plots.py
# Generates a comprehensive 2x2 performance summary plot for hover and cruise modes.

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

    # --- 3. Create a 2x2 Figure ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Performance Summary of Initial Designs', fontsize=20, y=0.98)

    # --- Plot 1 (Top-Left): Hover Performance vs. Root AoA ---
    ax = axes[0, 0]
    if not df_hover.empty:
        scatter1 = sns.scatterplot(data=df_hover, x='aoa_root (deg)', y='performance', hue='op_speed',
                                   palette='viridis', alpha=0.7, s=40, ax=ax, legend=False)
        ax.set_title('Hover η vs. Root AoA (Colored by RPM)', fontsize=14)
        ax.set_xlabel('Root Angle of Attack (deg)', fontsize=12)
        ax.set_ylabel('Hover Efficiency (η_hover)', fontsize=12)
        
        norm = plt.Normalize(df_hover['op_speed'].min(), df_hover['op_speed'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('RPM', rotation=270, labelpad=15, fontsize=12)
        min_rpm, med_rpm, max_rpm = df_hover['op_speed'].min(), df_hover['op_speed'].median(), df_hover['op_speed'].max()
        cbar.set_ticks([min_rpm, med_rpm, max_rpm])
        cbar.set_ticklabels([f'{min_rpm:.0f}', f'{med_rpm:.0f}', f'{max_rpm:.0f}'])
    else:
        ax.text(0.5, 0.5, "No Hover Data", ha='center', va='center')

    # --- Plot 2 (Top-Right): Hover Performance vs. RPM ---
    ax = axes[0, 1]
    if not df_hover.empty:
        sns.lineplot(data=df_hover, x='op_speed', y='performance', hue='filename',
                     marker='o', markersize=4, style='filename', ax=ax, legend=True)
        ax.set_title('Hover η vs. RPM (Grouped by LS Design)', fontsize=14)
        ax.set_xlabel('Operational Speed (RPM)', fontsize=12)
        ax.set_ylabel('Hover Efficiency (η_hover)', fontsize=12)
        ax.legend(title='Lifting Surface', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.text(0.5, 0.5, "No Hover Data", ha='center', va='center')

    # --- Plot 3 (Bottom-Left): Cruise Performance vs. Root AoA ---
    ax = axes[1, 0]
    if not df_cruise.empty:
        sns.scatterplot(data=df_cruise, x='aoa_root (deg)', y='performance', 
                        alpha=0.6, s=40, ax=ax, color='seagreen')
        ax.set_title('Cruise L/D vs. Root AoA', fontsize=14)
        ax.set_xlabel('Root Angle of Attack (deg)', fontsize=12)
        ax.set_ylabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
    else:
        ax.text(0.5, 0.5, "No Cruise Data", ha='center', va='center')

    # --- Plot 4 (Bottom-Right): Cruise Performance vs. Cruise Speed ---
    ax = axes[1, 1]
    if not df_cruise.empty:
        sns.scatterplot(data=df_cruise, x='op_speed', y='performance', 
                        alpha=0.6, s=40, ax=ax, color='darkorange')
        ax.set_title('Cruise L/D vs. Cruise Speed', fontsize=14)
        ax.set_xlabel('Operational Speed (m/s)', fontsize=12)
        ax.set_ylabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
    else:
        ax.text(0.5, 0.5, "No Cruise Data", ha='center', va='center')

    # --- 4. Customize and Save the Plot ---
    for ax in axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0, 0.9, 0.97]) # Adjust layout for legend

    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "02_performance_summary.pdf"
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"\nSuccessfully generated comprehensive performance summary plot.")
    print(f"Output saved to: '{plot_path}'")

if __name__ == "__main__":
    main()