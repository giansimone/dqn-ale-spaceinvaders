"""
Training script for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import wandb

from agent import Agent
from environment import make_env, get_env_dims
from utils import set_seed, load_config, save_config


def evaluate_agent(agent: Agent, config: dict) -> np.float64:
    """Evaluate the agent over a number of episodes."""
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

    return np.mean(eval_scores, dtype=np.float64)


def calculate_epsilon(step: int, config: dict):
    if step >= config["anneal_steps"]:
        return config["epsilon_end"]
    return config["epsilon_start"] - step * (config["epsilon_start"] - config["epsilon_end"]) / config["anneal_steps"]


def train(config_filename: Path = Path("config.yaml")) ->  None:
    """Train a DQN agent to play Atari Space Invaders."""
    config = load_config(config_filename)

    set_seed(config["seed"])

    run_name = "dqn_" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    run = wandb.init(
        project="DQN-SpaceInvaders-v5",
        name=run_name,
        config=config,
    )


    log_dir = Path(config["log_dir"])
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.copy(), run_dir / "config.yaml")

    env = make_env(config)
    state_size, action_size = get_env_dims(env)

    print(f"Device: {config['device']}")
    print(f"Action Space: {action_size}")
    print(f"State Shape: {state_size}")

    agent = Agent(state_size, action_size, config)

    episode = 0
    losses_window = deque(maxlen=config["max_len_window"])
    scores_window = deque(maxlen=config["max_len_window"]) 
    best_eval_score = -np.inf

    while agent.n_step < config["training_steps"]:
        state, _ = env.reset(seed=config["seed"] + agent.n_step)
        score = 0.
        done = False
        episode += 1

        while not done:
            epsilon = calculate_epsilon(agent.n_step, config)

            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, float(reward), next_state, done)
            if loss is not None:
                losses_window.append(loss)

            state = next_state
            score += float(reward)

        scores_window.append(score)

        run.log({
            "episode": episode,
            "score": score,
            "avg_score": np.mean(scores_window),
            "std_score": np.std(scores_window),
            "avg_loss": np.mean(losses_window),
            "std_loss": np.std(losses_window),
            "epsilon": epsilon,
        }, step=agent.n_step)

        if episode % config["eval_every"] == 0:
            eval_score = evaluate_agent(agent, config)
            run.log({"eval_score": eval_score}, step=agent.n_step)

            print(f"\n| Step {agent.n_step} / {config['training_steps']}"
                  f"| Evaluation Score: {eval_score:.2f}"
            )

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save_model(run_dir / "best_model.pth")
                print(f"|--> New best model saved with eval score: {eval_score:.2f}")

    agent.save_model(run_dir / "final_model.pth")

    print("\nTraining complete!")
    print(f"Best evaluation score: {best_eval_score:.2f}")

    run.finish()
    env.close()


if __name__ == "__main__":
    train()
