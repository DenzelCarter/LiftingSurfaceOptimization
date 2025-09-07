# Scripts/02_quick_plots.py
# Generates plots for both hover (eta vs. disk loading) and cruise (L/D vs. AoA) data.

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    fig.suptitle('Performance Summary', fontsize=18, y=0.99)

    # --- Plot 1: Hover Performance (Efficiency vs. Disk Loading) ---
    if not df_hover.empty:
        print(f"Plotting hover performance for {df_hover['filename'].nunique()} unique propellers.")
        sns.lineplot(data=df_hover, x='op_point', y='performance', hue='filename',
                     marker='o', markersize=5, legend='full', ax=ax1)
        
        ax1.set_title('Hover Performance', fontsize=14)
        ax1.set_xlabel('Disk Loading (T/A, N/m²)', fontsize=12)
        ax1.set_ylabel('Hover Efficiency (η_hover)', fontsize=12)
        
        handles, labels = ax1.get_legend_handles_labels()
        cleaned_labels = [os.path.splitext(os.path.basename(label))[0] for label in labels]
        ax1.legend(handles, cleaned_labels, title='Lifting Surface', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.text(0.5, 0.5, "No Hover Data Found", ha='center', va='center', fontsize=14, alpha=0.5)
        ax1.set_title('Hover Performance', fontsize=14)


    # --- Plot 2: Cruise Performance (L/D vs. Root AoA) ---
    if not df_cruise.empty:
        print(f"Plotting cruise performance for {len(df_cruise)} COMSOL data points.")
        # --- MODIFIED: Simplified scatter plot with no legend ---
        sns.scatterplot(data=df_cruise, x='op_point', y='performance', 
                        color='royalblue', alpha=0.7, s=50, ax=ax2)
        
        ax2.set_title('Cruise Performance (from COMSOL)', fontsize=14)
        ax2.set_xlabel('Root Angle of Attack (deg)', fontsize=12)
        ax2.set_ylabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability
    else:
        ax2.text(0.5, 0.5, "No Cruise Data Found", ha='center', va='center', fontsize=14, alpha=0.5)
        ax2.set_title('Cruise Performance', fontsize=14)

    # --- 4. Customize and Save the Plot ---
    plt.tight_layout(rect=[0, 0, 0.85, 0.98]) # Adjust layout to make room for hover legend

    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "02_performance_summary.pdf"
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"\nSuccessfully generated performance summary plot.")
    print(f"Output saved to: '{plot_path}'")

if __name__ == "__main__":
    main()