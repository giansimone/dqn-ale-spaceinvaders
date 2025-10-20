"""
Module to enjoy a trained DQN agent playing Atari Space Invaders.
"""
import argparse
from pathlib import Path

from agent import Agent
from environment import make_env, get_env_dims
from utils import load_config


def enjoy(run_dir: Path, num_episodes: int, model_name: str) -> None:
    """Enjoy a trained DQN agent playing Atari Space Invaders."""
    config_path = run_dir / "config.yaml"
    config = load_config(config_path)

    model_path = run_dir / model_name

    env = make_env(config, render_mode="human")
    state_size, action_size = get_env_dims(env)

    agent = Agent(state_size, action_size, config)
    agent.load_model(model_path)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0.

        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            score += float(reward)

        print(f"Atari Space Invaders Episode {episode} |--> Score: {score:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        "-r",
        type=str,
        required=True,
        help="The path to the saved agent model to play Atari Space Invaders.",
    )
    parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=10,
        help="The number of Atari Space Invaders episodes to enjoy.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="best_model.pth",
        help="The model name to play Atari Space Invaders.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    num_episodes = args.num_episodes
    model_name = args.model_name

    enjoy(run_dir=run_dir, num_episodes=num_episodes, model_name=model_name)
