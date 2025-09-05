# Scripts/00_plot_intro_tradeoff.py
# Creates the introductory figure (Fig. 1) for the paper, illustrating the
# performance trade-off space with a full-span conceptual Pareto front.

import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # --- 1. Define Illustrative Performance Data ---
    # Data points are (Cruise L/D, Hover Efficiency)
    aircraft_performance = {
        'Quadcopter': (2, 0.75),
        'Helicopter': (5, 0.65),
        'Tilt-Rotor': (12, 0.50),
        'Fixed-Wing UAV': (16, 0.05),
    }
    
    # --- 2. Define Conceptual Ideal Pareto Front ---
    # MODIFIED: The curve now spans the entire x-axis of the plot (0 to 20).
    ideal_ld = np.linspace(0, 20, 200)
    
    # MODIFIED: Adjusted the equation to ensure it starts high and ends near zero.
    ideal_efficiency = 0.82 - 0.0025 * (ideal_ld)**2
    ideal_efficiency = np.maximum(ideal_efficiency, 0.0) # Clip at zero efficiency.
    
    # --- 3. Create the Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the Ideal Performance Frontier and the now complete Desirable Region
    ax.plot(ideal_ld, ideal_efficiency, 'k--', linewidth=2, alpha=0.7)
    ax.fill_between(ideal_ld, ideal_efficiency, 1.0, color='gray', alpha=0.1)
    ax.text(14, 0.8, 'Desirable\nRegion', ha='center', fontsize=12, color='black', alpha=0.6)

    # Plot each aircraft type
    for name, (ld, eff) in aircraft_performance.items():
        ax.plot(ld, eff, 'o', markersize=12) 
        ax.text(ld + 0.3, eff, name, fontsize=11, verticalalignment='center')
            
    # Add a specific target/goal point
    ax.plot(13, 0.65, '*', color='crimson', markersize=20)
    ax.text(13.3, 0.65, 'Stop-Rotor Target', fontsize=11, color='crimson', verticalalignment='center')

    # --- 4. Final Touches for Publication Quality ---
    ax.set_xlabel("Cruise Performance (Lift-to-Drag Ratio, L/D)")
    ax.set_ylabel("Hover Performance (Hover Efficiency, $\eta_{hover}$)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.05, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    fig.tight_layout()

    # --- 5. Save the Figure ---
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Experiment', 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "00_intro_tradeoff_plot.pdf")
    
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved introductory trade-off plot to: {output_path}")

if __name__ == "__main__":
    main()