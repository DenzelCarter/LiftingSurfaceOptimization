# Scripts/02_quick_plots.py
# Generates a quick plot of hover efficiency vs. disk loading for all tested propellers.

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
    
    # --- MODIFIED: Filter for hover data only ---
    df_hover = df_full[df_full['flight_mode'] == 'hover'].copy()
    if df_hover.empty:
        raise SystemExit("No hover data found in the master dataset to plot.")

    print(f"Plotting hover performance for {df_hover['filename'].nunique()} unique propellers.")

    # --- 2. Create Hover Performance Plot (Efficiency vs. Disk Loading) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use seaborn to plot, creating a distinct line for each propeller (filename)
    sns.lineplot(
        data=df_hover,
        x='op_point',
        y='performance',
        hue='filename',
        marker='o',
        markersize=5,
        legend='full',
        ax=ax
    )

    # --- 3. Customize and Save the Plot ---
    ax.set_title('Hover Performance: Efficiency vs. Disk Loading', fontsize=16)
    ax.set_xlabel('Disk Loading (T/A, N/m²)', fontsize=12)
    ax.set_ylabel('Hover Efficiency (η_hover)', fontsize=12)
    
    # Improve legend readability
    handles, labels = ax.get_legend_handles_labels()
    # Clean up labels if they are long file paths
    cleaned_labels = [os.path.splitext(os.path.basename(label))[0] for label in labels]
    ax.legend(handles, cleaned_labels, title='Lifting Surface', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    # Save the figure
    plots_dir = (script_dir / P["outputs_plots"]).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "02_hover_performance_vs_DL.pdf"
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"\nSuccessfully generated hover performance plot.")
    print(f"Output saved to: '{plot_path}'")

if __name__ == "__main__":
    main()