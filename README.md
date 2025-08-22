# ICRA_2026_Optimization

A compact, script-driven workflow to **design, analyze, and select** a dual-function **morphing lifting surface (LS)** optimized for **VTOL (hover)** and **cruise**.  
It integrates:

- **Experimental** data (bench VTOL on the 1585 rig + wind-tunnel cruise),
- **Analytical priors** (BEMT for hover; Lifting-Line for cruise),
- **COMSOL CFD** surrogates (VTOL + cruise), and
- **Geometry-only ML models** trained on experiments.

---

## Folder layout

ICRA_2026_Optimization/
├─ Experiment/
│ ├─ doe/
│ │ └─ doe_test_plan_02.csv
│ ├─ data/ # RAW inputs only (never edited by scripts)
│ │ ├─ bench/
│ │ │ ├─ raw/ # 1585 rig logs with LS
│ │ │ └─ tare/ # 1585 rig logs, mount-only
│ │ ├─ tunnel/
│ │ │ ├─ raw/ # wind tunnel with LS
│ │ │ └─ tare/ # tunnel tare, mount-only
│ │ ├─ comsol/
│ │ │ ├─ vtol/ # COMSOL .txt (hover)
│ │ │ └─ cruise/ # COMSOL .txt (cruise)
│ │ └─ airfoils/ # polars (e.g., naca0012_all_re.csv)
│ ├─ outputs/ # DERIVED artifacts (scripts write here)
│ │ ├─ tables/ # “tools” directory target for CSV/Parquet
│ │ ├─ models/ # (optional) serialized ML models
│ │ └─ plots/ # paper figures
├─ models/
│ └─ analytical/
│ ├─ calculate_bemt.py
│ └─ calculate_lifting_line.py
└─ Scripts/
├─ config.yaml
├─ path_utils.py
├─ process_data.py
├─ make_priors.py
├─ integrate_cfd_txt.py
├─ train_models.py
├─ select_dual.py
└─ make_plots.py


> **Terminology:** We refer to the hardware as an **LS (lifting surface)** throughout, but column/filenames may retain “prop” for backward compatibility (e.g., `prop_efficiency_mean` = hover η̄).

---

## Requirements

- **Python** 3.10–3.13
- Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`, `scipy`

Install:
```bash
pip install numpy pandas scikit-learn matplotlib pyyaml scipy

