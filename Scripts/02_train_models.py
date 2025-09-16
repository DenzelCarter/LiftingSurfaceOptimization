# Scripts/02_train_models.py
# Trains GPR models for hover (heteroscedastic) and cruise (near-deterministic).
# Uses per-step bootstrap variance directly as alpha for hover. No plotting here.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import make_pipeline
from joblib import dump
from pathlib import Path
import warnings
import path_utils

def _prepare_xy(df, features, target, need_alpha: bool):
    X = df[features].copy()
    y = df[target].astype(float).copy()
    if need_alpha:
        if 'performance_variance' not in df.columns:
            raise ValueError("Missing 'performance_variance' for heteroscedastic training.")
        alpha = df['performance_variance'].astype(float).to_numpy()
        alpha = np.where(np.isfinite(alpha) & (alpha >= 0.0), alpha, np.nan)
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1) & np.isfinite(alpha)
        X, y, alpha = X.loc[mask], y.loc[mask], alpha[mask]
        alpha = np.maximum(alpha, 1e-12)  # numerical floor
        return X, y, alpha
    else:
        mask = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
        return X.loc[mask], y.loc[mask], None

def _report_training_header(name, y, alpha=None):
    print(f"\n--- Training {name} GPR Model ---")
    print(f"y stats → min={np.nanmin(y):.4f}, median={np.nanmedian(y):.4f}, max={np.nanmax(y):.4f}, N={len(y)}")
    if alpha is not None:
        print(f"alpha (Var) stats → min={np.nanmin(alpha):.3e}, p50={np.nanpercentile(alpha,50):.3e}, "
              f"p90={np.nanpercentile(alpha,90):.3e}, max={np.nanmax(alpha):.3e}")

def train_gpr_hover(df, features, target, random_state=42):
    """Hover: heteroscedastic noise provided via alpha; kernel excludes WhiteKernel."""
    X, y, alpha = _prepare_xy(df, features, target, need_alpha=True)
    _report_training_header("Hover", y, alpha)

    D = X.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))

    gpr_pipeline = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,              # heteroscedastic noise (per-sample)
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=random_state
        )
    )
    print("Fitting Hover GPR...")
    gpr_pipeline.fit(X, y)
    print("Fit complete.")
    gpr = gpr_pipeline.named_steps['gaussianprocessregressor']
    print(f"Learned kernel (hover): {gpr.kernel_}")
    return gpr_pipeline

def train_gpr_cruise(df, features, target, random_state=42):
    """Cruise: near-deterministic; small WhiteKernel + tiny alpha regularizer."""
    X, y, _ = _prepare_xy(df, features, target, need_alpha=False)
    _report_training_header("Cruise", y, alpha=None)

    D = X.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))

    gpr_pipeline = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,            # small numeric stabilizer; homoscedastic part via WhiteKernel
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=random_state
        )
    )
    print("Fitting Cruise GPR...")
    gpr_pipeline.fit(X, y)
    print("Fit complete.")
    gpr = gpr_pipeline.named_steps['gaussianprocessregressor']
    print(f"Learned kernel (cruise): {gpr.kernel_}")
    return gpr_pipeline

def _coerce_feature_names(df):
    """
    Ensure master dataset matches config input_cols names.
    Allows backward compatibility with older columns.
    """
    rename_map = {}
    if "twist (deg)" in df.columns and "twist" not in df.columns:
        rename_map["twist (deg)"] = "twist"
    if "aoa_root (deg)" in df.columns and "alpha" not in df.columns:
        rename_map["aoa_root (deg)"] = "alpha"
    if "op_speed" in df.columns and "v" not in df.columns:
        rename_map["op_speed"] = "v"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def main():
    cfg = path_utils.load_cfg()
    P = cfg["paths"]

    # Features (new convention): a single list in config
    features = cfg.get("input_cols", ["AR","lambda","twist","alpha","v"])
    target   = 'performance'

    # Load processed dataset
    print("--- Loading Master Dataset ---")
    master_parquet_path = Path(P["master_parquet"])
    if not master_parquet_path.exists():
        raise FileNotFoundError(f"Master parquet not found at {master_parquet_path}. Run 01_process_data.py first.")
    df_full = pd.read_parquet(master_parquet_path)
    df_full = _coerce_feature_names(df_full)

    # Validate features present
    missing = [c for c in features if c not in df_full.columns]
    if missing:
        raise ValueError(f"Master dataset missing required feature columns: {missing}. "
                         f"Check 01_process_data.py and config.input_cols.")

    # Split by mode
    df_hover  = df_full[df_full['flight_mode'] == 'hover'].copy()
    df_cruise = df_full[df_full['flight_mode'] == 'cruise'].copy()

    # Output dir
    models_dir = Path(P["outputs_models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train & save hover
    if not df_hover.empty:
        gpr_hover = train_gpr_hover(df_hover, features, target)
        dump(gpr_hover, models_dir / "gpr_hover_model.joblib")
        print(f"Hover model saved to {models_dir / 'gpr_hover_model.joblib'}")
    else:
        warnings.warn("No hover data found; skipping hover model training.")

    # Train & save cruise
    if not df_cruise.empty:
        gpr_cruise = train_gpr_cruise(df_cruise, features, target)
        dump(gpr_cruise, models_dir / "gpr_cruise_model.joblib")
        print(f"Cruise model saved to {models_dir / 'gpr_cruise_model.joblib'}")
    else:
        warnings.warn("No cruise data found; skipping cruise model training.")

if __name__ == "__main__":
    main()
