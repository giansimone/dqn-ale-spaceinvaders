"""
Push a trained DQN agent to the Hugging Face Hub.
"""
import argparse
from pathlib import Path
from datetime import datetime
import json
import tempfile

import torch
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from agent import evaluate_agent
from utils import load_artifact, record_movie


def push_to_hub(
    username: str,
    repo_name: str,
    artifact_path: Path,
    movie_fps=12,
    ):
    """Push a trained DQN agent to the Hugging Face Hub."""
    api = HfApi()

    repo_id = f"{username}/{repo_name}"
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
    print(f"Repository created at {repo_url}")

    config, env, agent = load_artifact(artifact_path, "rgb_array")

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        torch.save(agent.policy_net.state_dict(), local_directory / "model.pt")

        with open(local_directory / "config.json", "w", encoding="utf-8") as f:
            device = config.pop("device", None)
            json.dump(config, f)
            config["device"] = device

        mean_reward, std_reward = evaluate_agent(agent, config)

        evaluate_data = {
            "env_id": config["env_id"],
            "mean_reward": mean_reward,
            "n_eval_episodes": config["n_eval_episodes"],
            "eval_datetime": datetime.now().isoformat(),
        }

        with open(local_directory / "results.json", "w", encoding="utf-8") as f:
            json.dump(evaluate_data, f)

        env_name = config["env_id"]

        metadata = {
            "tags": [
                env_name,
                "reinforcement-learning",
                "dqn",
                "atari",
                "gymnasium",
                "pytorch",
            ]
        }

        eval_metadata = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )

        metadata = {**metadata, **eval_metadata}

        model_card = f"""
# Deep Q-Network (DQN) Agent playing {env_name}

This is a trained Deep Q-Network (DQN) agent for the Atari game {env_name}.

The model was trained using the code available [here](https://github.com/giansimone/dqn-ale-spaceinvaders/).

## Usage
To load and use this model for inference:

```python
import torch
import json

from model import DQN
from agent import Agent
from environment import make_env, get_env_dims

#Load the configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Create environment. Get action and space dimensions
env = make_env(config)
state_size, action_size = get_env_dims(env)

# Instantiate the agent and load the trained policy network
agent = Agent(state_size, action_size, config)
agent.policy_net.load_state_dict(torch.load("model.pt"))
agent.policy_net.eval()

# Enjoy the agent!
state, _ = env.reset()
done = False
while not done:
    action = agent.act(state, epsilon=0.0) # Act greedily
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
```
"""
        readme_path = local_directory / "README.md"
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(model_card)

        metadata_save(readme_path, metadata)

        print("Recording movie...")
        movie_path = local_directory / "replay.mp4"
        record_movie(env, agent, movie_path, movie_fps)

        print("Uploading to Hugging Face Hub...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_directory,
            path_in_repo=".",
        )

        print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a trained DQN agent to Hugging Face Hub.")
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        required=True,
        help="Your Hugging Face username.",
    )
    parser.add_argument(
        "--repo-name",
        "-r",
        type=str,
        required=True,
        help="The name of the repository to create on the Hub.",
    )
    parser.add_argument(
        "--artifact-path",
        "-a",
        type=str,
        required=True,
        help="Path to the trained model artifact (.pt or .pth file).",
    )
    parser.add_argument(
        "--movie-fps",
        "-f",
        type=int,
        default=12,
        help="The fps value to record the movie.",
    )

    args = parser.parse_args()

    push_to_hub(
        username=args.username,
        repo_name=args.repo_name,
        artifact_path=Path(args.artifact_path),
        movie_fps=args.movie_fps,
    )
