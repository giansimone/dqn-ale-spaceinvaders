"""
Training script for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
import random
from collections import deque
from pathlib import Path

import torch
import numpy as np
import wandb

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from agent import Agent
from config import CONFIG


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate_agent(agent: Agent, config: dict, n_episodes: int = 10) -> float:
    eval_env = make_env(config)
    eval_scores = []

    for episode in range(n_episodes):
        state, _ = eval_env.reset(seed=config["seed"] + episode)
        score = 0.
        done = False

        while not done:
            action = agent.act(state, epsilon=0.)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            score += float(reward)

        eval_scores.append(score)

    eval_env.close()

    return np.mean(eval_scores)


def train():
    run_name = f"{'Dueling_' if CONFIG['dueling'] else ''}{'Double_' if CONFIG['double_dqn'] else''}DQN"
    wandb.init(
        project="DQN-SpaceInvaders-v5",
        name=run_name,
        config=CONFIG,
    )

    set_seed(CONFIG["seed"])

    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)


    env = make_env(CONFIG)

    state_size = env.observation_space.shape
    action_size = env.action_space.n

    print(f"Device: {CONFIG['device']}")
    print(f"Action Space: {action_size}")
    print(f"State Shape: {state_size}")

    agent = Agent(state_size, action_size, CONFIG)

    losses_window = deque(maxlen=CONFIG["max_len_window"])
    scores_window = deque(maxlen=CONFIG["max_len_window"])
    epsilon = CONFIG["epsilon_start"]
    best_eval_score = -np.inf

    for episode in range(1, CONFIG["n_episodes"] + 1):
        state, _ = env.reset(seed=CONFIG["seed"] + episode)
        score = 0.
        done = False
        episode_steps = 0

        while not done:
            episode_steps += 1

            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses_window.append(loss)

            state = next_state
            score += float(reward)

        scores_window.append(score)

        epsilon = max(CONFIG["epsilon_end"], CONFIG["epsilon_decay"] * epsilon)

        avg_score = np.mean(scores_window)
        std_score = np.std(scores_window)
        avg_loss = np.mean(losses_window)
        std_loss = np.std(losses_window)

        wandb.log({
            "Episode Reward": score,
            "Average Reward": avg_score,
            "Std Reward": std_score,
            "Average Loss": avg_loss,
            "Std Loss": std_loss,
            "Epsilon": epsilon,
            "Episode Steps": episode_steps,
        }, step=episode)

        if episode % CONFIG["eval_every"] == 0:
            eval_score = evaluate_agent(agent, CONFIG, CONFIG["n_eval_episodes"])
            wandb.log({"Evaluation Score": eval_score}, step=episode)

            print(f"\n| Episode {episode} / {CONFIG['n_episodes']}")
            print(f"| Average Score: {avg_score:.2f} | Evaluation Score: {eval_score:.2f}")
            print(f"| Average Loss: {avg_loss:.4f}")
            print(f"| Epsilon: {epsilon:.4f} | Steps: {agent.get_n_steps()}")

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                try:
                    agent.save_model(checkpoint_dir / "best_model.pth")
                    print(f"|--> New best model saved with eval score: {eval_score:.2f}")
                except Exception as e:
                    print(f"|--> Error saving model: {e}")

    try:
        agent.save_model(checkpoint_dir / "final_model.pth")
    except Exception as e:
        print(f"|--> Error saving model: {e}")

    print("\nTraining complete!")
    print(f"Best evaluation score: {best_eval_score:.2f}")

    wandb.finish()
    env.close()


if __name__ == "__main__":
    train()
