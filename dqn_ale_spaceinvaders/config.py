"""
Configuration parameters for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

CONFIG = {
    #Environment
    "env_id": "ALE/SpaceInvaders-v5",
    "frame_skip": 1,
    "frame_stack": 4,
    "resized_frame": 84,

    #Training
    "n_episodes": 2_000,
    "n_eval_episodes": 10,
    "max_n_step": 10_000,

    #Exploration
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9995,

    #Replay Buffer
    "buffer_size": 100_000,
    "batch_size": 32,

    #Learning
    "gamma": 0.99,
    "tau": 0.001,
    "lr": 0.00025,
    "max_norm_grad": 10.,
    "update_every": 4,

    #Logging and Checkpointing
    "max_len_window": 100,
    "eval_every": 50,
    "checkpoint_dir": "checkpoints/",

    #DQN Variants
    "double_dqn": True,
    "dueling": True,
    "clip_rewards": True,
    "use_huber_loss": True,

    #System
    "device": get_device(),
    "seed": 42,
}
