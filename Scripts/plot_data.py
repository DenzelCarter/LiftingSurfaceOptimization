# plot_data.py
# Visualize efficiencies vs RPM for all tested props from master_dataset.parquet.
# - Uses rpm_bin and *_mean columns
# - Hampel outlier detection per prop vs RPM
# - Optional moving-average smoothing for display
# Run with the VS Code ▶️ button.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- paths (relative to this file) ----------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
TOOLS_DIR  = os.path.join(PROJ_ROOT, 'Experiment', 'tools')
PLOTS_DIR  = os.path.join(PROJ_ROOT, 'Plots')

MASTER_DATASET_PATH   = os.path.join(TOOLS_DIR, 'master_dataset.parquet')
OUTLIER_REPORT_CSV    = os.path.join(TOOLS_DIR, 'plot_outlier_bins.csv')

# ---------------- config ----------------
# Outlier detection (Hampel: rolling median ± K * 1.4826 * MAD)
RUN_OUTLIER_FILTER    = True
HAMPEL_WINDOW         = 2      # neighbors on each side → 2 → window size = 5 points
HAMPEL_K              = 3.0    # threshold in MAD units
DROP_OUTLIERS_IN_LINE = True   # remove outliers from line paths
SHOW_OUTLIERS_AS_X    = True   # overlay outliers as red '×'

# Display smoothing (for the line only; never used in data files)
SMOOTH_DISPLAY        = True
SMOOTH_WINDOW         = 3      # simple moving average window (bins)

# Column names used by the pipeline
RPM_RAW_COL   = 'Motor Electrical Speed (RPM)'
RPM_BIN_COL   = 'rpm_bin'  # preferred for plotting if present

def pick_col(df: pd.DataFrame, base: str) -> str:
    """Use '<base>_mean' if present (preferred), else fall back to '<base>'."""
    mean_col = f"{base}_mean"
    if mean_col in df.columns: return mean_col
    if base in df.columns:     return base
    raise KeyError(f"Neither '{mean_col}' nor '{base}' found in dataset.")

def hampel_flags(y, k=3.0, win=2):
    """
    Return boolean mask of outliers using Hampel filter on a 1D array.
    win is neighbors on each side (total window = 2*win+1).
    """
    y = np.asarray(y, float)
    n  = len(y)
    m  = np.zeros(n, dtype=bool)
    if n == 0: return m

    # rolling median and MAD
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        window = y[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        sigma = 1.4826 * mad  # Gaussian consistency
        if sigma < 1e-12:
            # if flat window, treat nothing as outlier
            m[i] = False
        else:
            m[i] = np.abs(y[i] - med) > k * sigma
    return m

def moving_average(y, w=3):
    if w <= 1: return y
    y = np.asarray(y, float)
    if len(y) < w: return y
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')

def main():
    print(f"Loading master dataset from '{MASTER_DATASET_PATH}'...")
    if not os.path.exists(MASTER_DATASET_PATH):
        print(f"Error: Master dataset not found at '{MASTER_DATASET_PATH}'. Run process_data.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(MASTER_DATASET_PATH)
    print(f"Loaded {len(df)} rows across {df['filename'].nunique()} props.")

    # Decide which RPM column to plot against
    rpm_col = RPM_BIN_COL if RPM_BIN_COL in df.columns else RPM_RAW_COL
    if rpm_col == RPM_BIN_COL:
        print("Plotting vs. 'rpm_bin' (already binned).")
    else:
        print("Warning: 'rpm_bin' not found; plotting vs raw RPM (may be noisier).")

    # Pick efficiency columns (prefer *_mean if present)
    sys_eff_col   = pick_col(df, 'system_efficiency')
    prop_eff_col  = pick_col(df, 'prop_efficiency')
    motor_eff_col = pick_col(df, 'motor_efficiency')

    # Aggregate to unique (filename, rpm_col) safely (don't average rpm itself)
    eff_cols = [sys_eff_col, prop_eff_col, motor_eff_col]
    group_cols = ['filename', rpm_col]
    df_plot = (
        df.groupby(group_cols, as_index=False)[eff_cols]
          .mean()
          .sort_values(['filename', rpm_col])
          .reset_index(drop=True)
    )

    # Ensure numeric RPM for plotting
    if not pd.api.types.is_numeric_dtype(df_plot[rpm_col]):
        df_plot[rpm_col] = pd.to_numeric(df_plot[rpm_col], errors='coerce')

    # Outlier detection per prop across RPM
    if RUN_OUTLIER_FILTER:
        flags = []
        for prop, g in df_plot.groupby('filename'):
            g = g.sort_values(rpm_col)
            y = g[prop_eff_col].to_numpy()
            m = hampel_flags(y, k=HAMPEL_K, win=HAMPEL_WINDOW)
            f = pd.DataFrame({
                'filename': prop,
                rpm_col: g[rpm_col].to_numpy(),
                'is_outlier': m,
                'value': y
            })
            flags.append(f)
        outlier_df = pd.concat(flags, ignore_index=True)
        # join back
        df_plot = df_plot.merge(outlier_df[['filename', rpm_col, 'is_outlier']], on=['filename', rpm_col], how='left')
        df_plot['is_outlier'] = df_plot['is_outlier'].fillna(False)
        # write a small report
        rep = df_plot[df_plot['is_outlier']][['filename', rpm_col, prop_eff_col, sys_eff_col, motor_eff_col]]
        rep.to_csv(OUTLIER_REPORT_CSV, index=False)
        print(f"Flagged {len(rep)} outlier bins via Hampel; wrote {OUTLIER_REPORT_CSV}")
    else:
        df_plot['is_outlier'] = False

    # Palette per prop
    props = df_plot['filename'].unique().tolist()
    palette = sns.color_palette("husl", n_colors=len(props))
    color_map = {p: c for p, c in zip(props, palette)}

    # Ensure output folder
    os.makedirs(PLOTS_DIR, exist_ok=True)

    def make_plot(y_col: str, title: str, y_label: str, fname: str):
        fig, ax = plt.subplots(figsize=(12, 9))
        for prop, g in df_plot.groupby('filename'):
            g = g.sort_values(rpm_col)
            color = color_map[prop]

            # choose data for the line
            if DROP_OUTLIERS_IN_LINE:
                g_line = g.loc[~g['is_outlier']].copy()
            else:
                g_line = g.copy()

            # optional smoothing (on y only)
            x = g_line[rpm_col].to_numpy()
            y = g_line[y_col].to_numpy()
            if SMOOTH_DISPLAY and len(y) >= SMOOTH_WINDOW:
                y_disp = moving_average(y, w=SMOOTH_WINDOW)
                # keep x the same length (moving_average preserves length here)
                ax.plot(x, y_disp, '-', linewidth=1.6, color=color, label=prop.replace('.csv',''))
            else:
                ax.plot(x, y, '-', linewidth=1.6, color=color, label=prop.replace('.csv',''))

            # show the non-outlier points
            ax.plot(g_line[rpm_col], g_line[y_col], 'o', markersize=3.5, color=color)

            # overlay outliers (if any)
            if SHOW_OUTLIERS_AS_X:
                g_out = g.loc[g['is_outlier']]
                if not g_out.empty:
                    ax.plot(g_out[rpm_col], g_out[y_col], 'x', color='red', markersize=6, mew=1.5, linestyle='none')

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('RPM (bin)' if rpm_col == RPM_BIN_COL else 'RPM', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        # Keep legend readable
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Propeller", fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        out_path = os.path.join(PLOTS_DIR, fname)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Plot saved to '{out_path}'")

    print(f"\nGenerating plots → {PLOTS_DIR}/")
    make_plot(sys_eff_col,
              'System Efficiency vs. RPM (all props)',
              'System Efficiency (Ideal Power / Electrical Power)',
              'System_Efficiency_vs_RPM.pdf')

    make_plot(prop_eff_col,
              'Propeller Efficiency vs. RPM (all props)',
              'Propeller Efficiency (Ideal Power / Mechanical Power)',
              'Propeller_Efficiency_vs_RPM.pdf')

    make_plot(motor_eff_col,
              'Motor Efficiency vs. RPM (all props)',
              'Motor Efficiency (Mechanical Power / Electrical Power)',
              'Motor_Efficiency_vs_RPM.pdf')

    # Helpful console summary
    summary = (
        df_plot.groupby('filename')[rpm_col]
               .agg(['min','max','nunique'])
               .rename(columns={'nunique':'n_bins'})
               .reset_index()
    )
    print("\nPer-prop RPM coverage (plotted bins):")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
