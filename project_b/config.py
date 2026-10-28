"""Load the Project B config file so every script reads the same assumptions."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "project_b.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Turn relative paths into absolute ones (relative to the repo root)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = REPO_ROOT / value
    return cfg
