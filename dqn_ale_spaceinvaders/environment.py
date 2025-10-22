"""
Module to create and manage the Atari Space Invaders environment.
"""
from typing import Tuple

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def make_env(config: dict, render_mode=None) -> gym.Env:
    "Create and wrap environment."
    env = gym.make(config["env_id"], frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        frame_skip=config["frame_skip"],
        screen_size=config["resized_frame"],
    )
    env = FrameStackObservation(env, config["frame_stack"])
    return env


def get_env_dims(env: gym.Env) -> Tuple:
    """Get state and action sizes from Gymnasiym environment."""
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    return state_size, action_size
