import random
from datetime import datetime

import yaml

import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(yaml_path: str = "base_config.yaml") -> dict:
    """Load the YAML configuration file into a standard Python dictionary."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config["device"] = get_device()

    return config


def generate_run_name(config: dict) -> str:
    dueling_prefix = "Dueling_" if config['dueling'] else ""
    double_prefix = "Double_" if config['double_dqn'] else ""
    base_name = f"{dueling_prefix}{double_prefix}DQN_"

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    run_name = base_name + timestamp
    return run_name
