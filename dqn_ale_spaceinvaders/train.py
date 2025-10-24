"""
Training script for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent import Agent, evaluate_agent
from environment import make_env, get_env_dims
from utils import set_seed, load_config, save_config


def calculate_epsilon(step: int, config: dict) -> float:
    """Calculate the epsilon value for epsilon-greedy action selection."""
    if step < config["warmup_steps"]:
        return 1.0
    step -= config["warmup_steps"]
    if step >= config["anneal_steps"]:
        return config["epsilon_end"]
    return config["epsilon_start"] - step * (config["epsilon_start"] - config["epsilon_end"]) / config["anneal_steps"]


def train(config_filename: Path = Path("config.yaml")) ->  None:
    """Train a DQN agent to play Atari Space Invaders."""
    config = load_config(config_filename)

    set_seed(config["seed"])

    run_name = "dqn_" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    log_dir = Path(config["log_dir"])
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.copy(), run_dir / "config.yaml")

    writer = SummaryWriter(log_dir=run_dir)

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

        if agent.n_step% config["eval_every"] == 0:
            avg_eval_score, std_eval_score = evaluate_agent(agent, config)

            writer.add_scalar("episode", episode, global_step=agent.n_step)
            writer.add_scalar("metrics/training_score", np.mean(scores_window), global_step=agent.n_step)
            writer.add_scalar("metrics/training_score_std", np.std(scores_window), global_step=agent.n_step)
            writer.add_scalar("metrics/evaluation_score", avg_eval_score, global_step=agent.n_step)
            writer.add_scalar("metrics/evaluation_score_std", std_eval_score, global_step=agent.n_step)
            writer.add_scalar("hyperparameters/epsilon", epsilon, global_step=agent.n_step)

            print(f"\n| Step {agent.n_step} / {config['training_steps']} | Episode {episode}"
                  f"| Evaluation Score: {avg_eval_score:.2f} +/- {std_eval_score:.2f}"
            )

            if avg_eval_score > best_eval_score:
                best_eval_score = avg_eval_score
                agent.save_model(run_dir / "best_model.pt")
                print(f"|--> New best model saved with eval score: {avg_eval_score:.2f} +/- {std_eval_score:.2f}")

    agent.save_model(run_dir / "final_model.pt")

    print("\nTraining complete!")
    print(f"Best evaluation score: {best_eval_score:.2f}")

    writer.close()
    env.close()


if __name__ == "__main__":
    train()
