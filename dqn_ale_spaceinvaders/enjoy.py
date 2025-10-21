"""
Module to enjoy a trained DQN agent playing Atari Space Invaders.
"""
import argparse
from pathlib import Path

from utils import load_artifact


def enjoy(artifact_path: Path, n_episodes: int) -> None:
    """Enjoy a trained DQN agent playing Atari Space Invaders."""
    _, env, agent = load_artifact(artifact_path, "human")

    for episode in range(1, n_episodes + 1):
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
        "--artifact",
        "-a",
        type=str,
        required=True,
        help="The artifact to play Atari Space Invaders.",
    )
    parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=10,
        help="The number of Atari Space Invaders episodes to enjoy.",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    num_episodes = args.num_episodes

    enjoy(artifact_path=artifact, n_episodes=num_episodes)
