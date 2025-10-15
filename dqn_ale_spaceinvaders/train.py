"""
Training script for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
import random
from collections import deque
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import wandb

from agent import Agent
from environment import make_env
from utils import set_seed, load_config


def evaluate_agent(agent: Agent, config: dict) -> float:
    eval_env = make_env(config)
    eval_scores = []

    for episode in range(config["n_eval_episodes"]):
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


def train(yaml_config_path: str = "base_config.yaml") ->  None:
    config = load_config(yaml_config_path)

    dueling_prefix = "Dueling_" if config['dueling'] else ""
    double_prefix = "Double_" if config['double_dqn'] else ""
    base_name = f"{dueling_prefix}{double_prefix}DQN_"

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    run_name = base_name + timestamp
    wandb.init(
        project="DQN-SpaceInvaders-v5",
        name=run_name,
        config=config,
    )

    set_seed(config["seed"])

    base_dir = Path(config["base_dir"])
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config)

    state_size = env.observation_space.shape
    action_size = env.action_space.n

    print(f"Device: {config['device']}")
    print(f"Action Space: {action_size}")
    print(f"State Shape: {state_size}")

    agent = Agent(state_size, action_size, config)

    losses_window = deque(maxlen=config["max_len_window"])
    scores_window = deque(maxlen=config["max_len_window"])
    epsilon = config["epsilon_start"]
    best_eval_score = -np.inf

    for episode in range(1, config["n_episodes"] + 1):
        state, _ = env.reset(seed=config["seed"] + episode)
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

        epsilon = max(config["epsilon_end"], config["epsilon_decay"] * epsilon)

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
            "Total Steps": agent.get_n_steps(),
        }, step=episode)

        if episode % config["eval_every"] == 0:
            eval_score = evaluate_agent(agent, config)
            wandb.log({"Evaluation Score": eval_score}, step=episode)

            print(f"\n| Episode {episode} / {config['n_episodes']}")
            print(f"| Average Score: {avg_score:.2f} | Evaluation Score: {eval_score:.2f}")
            print(f"| Average Loss: {avg_loss:.4f}")
            print(f"| Epsilon: {epsilon:.4f} | Steps: {agent.get_n_steps()}")

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                try:
                    agent.save_model(run_dir / "best_model.pth")
                    print(f"|--> New best model saved with eval score: {eval_score:.2f}")
                except Exception as e:
                    print(f"|--> Error saving model: {e}")

    try:
        agent.save_model(run_dir / "final_model.pth")
    except Exception as e:
        print(f"|--> Error saving model: {e}")

    print("\nTraining complete!")
    print(f"Best evaluation score: {best_eval_score:.2f}")

    wandb.finish()
    env.close()


if __name__ == "__main__":
    train()
