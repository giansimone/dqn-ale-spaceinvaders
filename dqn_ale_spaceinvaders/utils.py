"""
Module for utility functions and configuration file handling.
"""
import random
from pathlib import Path

import yaml

import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the available device (CPU, CUDA, or MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(filepath: Path) -> dict:
    """Load the YAML configuration file into a standard Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["device"] = get_device()

    return config


def save_config(config: dict, filepath: Path) -> None:
    """Save a standard Python dictionary into a YAML configuration file."""
    config.pop("device", None)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
