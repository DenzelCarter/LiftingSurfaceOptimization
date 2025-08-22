# Scripts/path_utils.py
import os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.yaml")

def load_cfg():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    base = os.path.dirname(CFG_PATH)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    # resolve everything under cfg["paths"]
    for k, v in cfg.get("paths", {}).items():
        if isinstance(v, str):
            cfg["paths"][k] = _abs(v)
    return cfg
