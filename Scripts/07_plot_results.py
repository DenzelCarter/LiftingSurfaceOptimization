# Scripts/07_plot_results.py
# This script loads the pre-computed optimization results and generates the
# final plots for analysis. It runs quickly for iterative formatting.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import path_utils

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # --- 1. Load Pre-computed Data ---
    print("--- Loading pre-computed optimization and design data... ---")
    tables_dir = Path(P["outputs_tables"])
    
    results_path = tables_dir / "05_optimization_results.csv"
    initial_opt_path = tables_dir / "06_initial_designs_optimized.csv"

    if not results_path.exists() or not initial_opt_path.exists():
        raise SystemExit("Error: Required data files not found. Please run 05 and 06 first.")

    df_results = pd.read_csv(results_path)
    df_initial_optimized = pd.read_csv(initial_opt_path)

    # --- 2. Create the Main Pareto Front Plot ---
    print("\n--- Generating Final Pareto Front Plot ---")
    plots_dir = Path(P["outputs_plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "07_pareto_front_final.pdf"

    with plt.rc_context({'font.size': 12}):
        fig, ax1 = plt.subplots(figsize=(13, 10))
        ax1.set_title('Fully Optimized Designs vs. Optimized Initial Designs', fontsize=18, pad=20)
        ax1.set_xlabel('Cruise Performance (L/D)', fontsize=14)
        ax1.set_ylabel('Hover Performance (η)', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot the optimized performance of all initial geometries
        ax1.scatter(df_initial_optimized['ld_opt'], df_initial_optimized['eta_opt'], 
                    c='gray', s=60, label='Initial Geometries (at Optimal Operation)', alpha=0.8, zorder=3)
        
        # Plot Optimized Designs
        scatter = ax1.scatter(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], 
                              c=df_results['w_hover'], cmap='viridis', s=250, edgecolors='k', 
                              zorder=10, label='Fully Optimized Designs')
        ax1.plot(df_results['peak_cruise_ld'], df_results['peak_hover_eta'], 'k--', alpha=0.6, zorder=5)

        ax1.legend(loc='best')
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Hover Weight ($w_{hover}$)', rotation=270, labelpad=20, fontsize=12)
        
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

    print(f"Successfully generated combined Pareto front plot: {plot_path}")

    # --- 3. Create Parameter Trend Plot ---
    print("--- Generating Parameter Trend Plot ---")
    param_plot_path = plots_dir / "07_parameters_vs_weight.pdf"
    
    param_vars = [
        'AR', 'lambda', 'twist (deg)', 
        'aoa_root_hover (deg)', 'op_speed_hover (rpm)',
        'aoa_root_cruise (deg)', 'op_speed_cruise (m/s)'
    ]
    
    fig, axes = plt.subplots(len(param_vars), 1, figsize=(10, 20), sharex=True)
    fig.suptitle('Optimized Parameters vs. Hover Weight', fontsize=16, y=0.99)

    for ax, var in zip(axes, param_vars):
        ax.plot(df_results['w_hover'], df_results[var], marker='o', linestyle='-')
        ax.set_ylabel(var)
        ax.grid(True, linestyle='--')

    axes[-1].set_xlabel('Hover Weight ($w_{hover}$)')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(param_plot_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully generated parameter trend plot: {param_plot_path}")

if __name__ == "__main__":
    main()