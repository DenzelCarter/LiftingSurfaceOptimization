# Paper-ready plots: parity (hover/cruise), coverage, and recommendation scores.
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from path_utils import load_cfg

def savefig(p): plt.tight_layout(); plt.savefig(p, dpi=300); print("Saved", p); plt.close()

def main():
    C = load_cfg(); tools=C["paths"]["tools_dir"]; plots=C["paths"]["plots_dir"]; os.makedirs(plots, exist_ok=True)

    # Hover parity
    ph = os.path.join(tools,"prop_hover_predictions.csv")
    mh = os.path.join(tools,"prop_hover_metrics.csv")
    if os.path.exists(ph):
        d = pd.read_csv(ph); m = pd.read_csv(mh) if os.path.exists(mh) else None
        fig,ax=plt.subplots(figsize=(4.8,4.4))
        ax.scatter(d["y_true"], d["y_pred"], s=36, edgecolor="none")
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Measured η̄ (VTOL)"); ax.set_ylabel("Predicted η̄")
        if m is not None:
            ax.text(0.04,0.96, f"R²={m['global_r2'].iloc[0]:.3f}\nMAE={m['avg_mae'].iloc[0]:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="gray"))
        savefig(os.path.join(plots,"parity_hover.pdf"))

    # Cruise parity
    pc = os.path.join(tools,"wing_cruise_predictions.csv")
    mc = os.path.join(tools,"wing_cruise_metrics.csv")
    if os.path.exists(pc):
        d = pd.read_csv(pc); m = pd.read_csv(mc) if os.path.exists(mc) else None
        fig,ax=plt.subplots(figsize=(4.8,4.4))
        ax.scatter(d["ld_true"], d["ld_pred"], s=36, edgecolor="none")
        lo=min(d["ld_true"].min(), d["ld_pred"].min()); hi=max(d["ld_true"].max(), d["ld_pred"].max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel("Measured L/D (cruise)"); ax.set_ylabel("Predicted L/D")
        if m is not None:
            ax.text(0.04,0.96, f"R²={m['global_r2'].iloc[0]:.3f}\nMAE={m['avg_mae'].iloc[0]:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="gray"))
        savefig(os.path.join(plots,"parity_cruise.pdf"))

    # Coverage AR–λ colored by η̄ (if available)
    master_path = os.path.join(tools, "master_dataset.parquet")
    if os.path.exists(master_path):
        master = pd.read_parquet(master_path)
        if {"AR","lambda","prop_efficiency_mean"}.issubset(master.columns):
            fig,ax=plt.subplots(figsize=(5.0,4.4))
            sc=ax.scatter(master["AR"], master["lambda"], c=master["prop_efficiency_mean"], cmap="viridis", s=60, edgecolor="k", lw=0.3)
            plt.colorbar(sc, ax=ax, label="η̄ (VTOL)")
            ax.set_xlabel("AR"); ax.set_ylabel("λ")
            savefig(os.path.join(plots,"coverage_AR_lambda_hover.pdf"))

    # Recommendation bars
    rec = os.path.join(tools,"next_props_recommendations.csv")
    if os.path.exists(rec):
        r = pd.read_csv(rec).sort_values("rank")
        fig,ax=plt.subplots(figsize=(max(6,0.8*len(r)),4.0))
        x = np.arange(len(r))
        ax.bar(x-0.2, r["hover_score"], width=0.4, label="Hover")
        ax.bar(x+0.2, r["cruise_score"], width=0.4, label="Cruise")
        ax.plot(x, r["composite_score"], "k.-", label="Composite")
        ax.set_xticks(x); ax.set_xticklabels(r["filename"], rotation=35, ha="right")
        ax.set_ylabel("Score (0–1)"); ax.legend(); ax.set_title("Dual-mode selection scores")
        savefig(os.path.join(plots,"recommendations_dual.png"))

if __name__ == "__main__":
    main()
