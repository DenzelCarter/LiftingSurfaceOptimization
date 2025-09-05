# Scripts/07_plot_pareto.py
# Creates a publication-ready Pareto front plot with Cruise L/D on the x-axis.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from path_utils import load_cfg
import matplotlib.cm as cm

def find_pareto_front(points):
    """Finds the Pareto front from a set of 2D points to be maximized."""
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] > p, axis=1)
            is_efficient[i] = True
    return is_efficient

def main():
    C = load_cfg()
    P = C["paths"]
    HOVER_CFG = C.get("hover_process", {})
    CRUISE_CFG = C.get("cruise_process", {})

    # --- 1. Load Model and Data ---
    model_path = os.path.join(P["outputs_models"], "xgboost_dual_model.json")
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(model_path)
    print("Loaded dual-output XGBoost model.")

    df_doe = pd.read_csv(P["doe_csv"])
    df_optimal = pd.read_csv(os.path.join(P["outputs_tables"], "optimal_designs.csv"))
    print("Loaded DOE and optimal designs data.")

    # --- 2. Predict Performance for All Original DOE Designs ---
    hover_op_point = np.mean(HOVER_CFG.get("rpm_window", [800, 1600]))
    cruise_op_point = np.mean(CRUISE_CFG.get("airspeed_window", [15, 25]))
    
    materials = df_doe['material'].unique()
    all_doe_preds = []

    for material in materials:
        if not isinstance(material, str): continue
        df_doe_mat = df_doe[df_doe['material'] == material]
        
        for _, row in df_doe_mat.iterrows():
            hover_data = {'AR': [row['AR']], 'lambda': [row['lambda']], 'aoaRoot (deg)': [row['aoaRoot (deg)']], 
                          'aoaTip (deg)': [row['aoaTip (deg)']], 'material': [material], 
                          'flight_mode': ['hover'], 'op_point': [hover_op_point]}
            cruise_data = {'AR': [row['AR']], 'lambda': [row['lambda']], 'aoaRoot (deg)': [row['aoaRoot (deg)']], 
                           'aoaTip (deg)': [row['aoaTip (deg)']], 'material': [material], 
                           'flight_mode': ['cruise'], 'op_point': [cruise_op_point]}
            
            X_pred_raw = pd.concat([pd.DataFrame(hover_data), pd.DataFrame(cruise_data)], ignore_index=True)
            X_pred = pd.get_dummies(X_pred_raw, columns=['material', 'flight_mode'])
            model_cols = xgb_model.get_booster().feature_names
            for col in model_cols:
                if col not in X_pred.columns: X_pred[col] = 0
            
            preds = xgb_model.predict(X_pred[model_cols])
            all_doe_preds.append({
                'material': material, 'filename': row['filename'],
                'predicted_hover_eff': preds[0], 'predicted_cruise_LD': preds[1]
            })
            
    df_doe_preds = pd.DataFrame(all_doe_preds)

    # --- 3. Generate a Plot for Each Material ---
    for material in materials:
        if not isinstance(material, str): continue
        
        df_doe_plot = df_doe_preds[df_doe_preds['material'] == material]
        df_opt_plot = df_optimal[df_optimal['material'] == material]
        
        if df_doe_plot.empty: continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # --- MODIFIED: Swapped axes for consistency ---
        ax.scatter(df_doe_plot['predicted_cruise_LD'], df_doe_plot['predicted_hover_eff'],
                   c='black', alpha=0.3, label='DOE Designs', s=50)

        # Identify and plot the Pareto front of the DOE designs
        doe_points = df_doe_plot[['predicted_cruise_LD', 'predicted_hover_eff']].values
        pareto_mask = find_pareto_front(doe_points)
        pareto_points = doe_points[pareto_mask]
        pareto_points = pareto_points[pareto_points[:, 0].argsort()] # Sort by x-axis (L/D)
        
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k--', alpha=0.8, linewidth=2, label='DOE Pareto Front')

        # Plot the new, optimized designs with unique colors
        w_hover_values = sorted(df_opt_plot['w_hover'].unique())
        colors = cm.viridis(np.linspace(0, 1, len(w_hover_values)))

        for i, w_val in enumerate(w_hover_values):
            df_subset = df_opt_plot[df_opt_plot['w_hover'] == w_val]
            ax.scatter(df_subset['predicted_cruise_LD'], df_subset['predicted_hover_eff'],
                       c=[colors[i]], marker='*', s=350, edgecolor='black', zorder=10)
            
            for _, row in df_subset.iterrows():
                ax.text(row['predicted_cruise_LD'], row['predicted_hover_eff'] + 0.015,
                        f"$w_h={row['w_hover']}$", ha='center', fontsize=11, fontweight='bold')
        
        # --- MODIFIED: Swapped axis labels ---
        ax.set_xlabel("Predicted Cruise L/D Ratio")
        ax.set_ylabel("Predicted Hover Efficiency, $\eta_{hover}$")
        
        ax.legend(loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        fig.tight_layout()
        
        output_path = os.path.join(P["outputs_plots"], f"pareto_front_{material.lower()}.pdf")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved publication-ready Pareto front plot to: {output_path}")

if __name__ == "__main__":
    main()