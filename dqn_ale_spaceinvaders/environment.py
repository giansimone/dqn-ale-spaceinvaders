import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def make_env(config: dict, render_mode=None) -> gym.Env:
    "Create and wrap environment."
    env = gym.make(config["env_id"], render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        frame_skip=config["frame_skip"],
        screen_size=config["resized_frame"],
    )
    env = FrameStackObservation(env, config["frame_stack"])
    return env
