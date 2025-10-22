"""
Module for utility functions and configuration file handling.
"""
import random
from pathlib import Path

import gymnasium as gym
import torch
import numpy as np
import yaml
import imageio.v3 as iio

from agent import Agent
from environment import make_env, get_env_dims


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


def load_artifact(artifact_path: Path, render_mode: str) -> tuple[dict, gym.Env, Agent]:
    """Load the configuration, environment, and agent from the specified artifact path."""
    config_path = artifact_path.parent / "config.yaml"
    config = load_config(config_path)

    env = make_env(config, render_mode=render_mode)
    state_size, action_size = get_env_dims(env)

    agent = Agent(state_size, action_size, config)
    agent.load_model(artifact_path)

    return config, env, agent


def record_movie(env: gym.Env, agent: Agent, filepath: Path, fps: int = 60) -> None:
    """Record a movie of the agent interacting with the environment."""
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)

    while not done:
        action = agent.act(state, epsilon=0.0)

        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

        img = env.render()
        images.append(img)

    iio.imwrite(
        filepath,
        images,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
    print("done")

