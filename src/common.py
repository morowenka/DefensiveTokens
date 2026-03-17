"""Shared utilities for pipeline scripts."""

import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_script(log_name):
    """Common script setup: add project root to sys.path, configure logging, load config.

    Returns the parsed config dict.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{log_name}.log"),
            logging.StreamHandler(),
        ],
    )

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in config against PROJECT_ROOT
    _resolve_paths(config)

    return config


def _resolve_paths(config):
    """Resolve relative paths in config to absolute paths based on PROJECT_ROOT."""
    path_keys = {
        ("dataset", "self_labeled_path"),
        ("dataset", "defensive_path"),
        ("training", "output_dir"),
        ("evaluation", "alpaca_farm_path"),
        ("evaluation", "results_dir"),
    }
    for section, key in path_keys:
        if section in config and key in config[section]:
            p = Path(config[section][key])
            if not p.is_absolute():
                config[section][key] = str(PROJECT_ROOT / p)
