import random
from collections import deque
from typing import Tuple

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


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(self, buffer_size: int, seed: int = None):
        self.buffer = deque(maxlen=buffer_size)
        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
