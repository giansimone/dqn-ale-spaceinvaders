[![Python](https://img.shields.io/pypi/pyversions/dqn-ale-spaceinvaders.svg)](https://badge.fury.io/py/dqn-ale-spaceinvaders)
[![PyPI](https://badge.fury.io/py/dqn-ale-spaceinvaders.svg)](https://badge.fury.io/py/dqn-ale-spaceinvaders)
[![License](https://img.shields.io/github/license/giansimone/dqn-ale-spaceinvaders)](https://github.com/giansimone/dqn-ale-spaceinvaders/blob/main/LICENSE)

# Deep Q-Network (DQN) for Atari Space Invaders

A PyTorch implementation of Deep Q-Learning Network (DQN) trained to play Atari Space Invaders using the Arcade Learning Environment (ALE).

## Features

- **Vanilla DQN** and **Dueling DQN** architectures.
- **Double DQN** support for improved stability.
- **Replay Buffer** for experience replay.
- **Epsilon-greedy exploration** with annealing (i.e., linear decay).
- **TensorBoard integration** for training visualisation.
- **Hugging Face Hub integration** for model sharing.
- **Video recording** of agent gameplay.

## Requirements

Install via pip:

```bash
pip install dqn-ale-spaceinvaders
```

Or clone the repository and install dependencies:

```bash
git clone https://github.com/giansimone/dqn-ale-spaceinvaders.git
cd dqn-ale-spaceinvaders
poetry install
```

## Project Structure

```
dqn-ale-spaceinvaders/
├── agent.py           # DQN agent implementation
├── buffer.py          # Experience replay buffer
├── config.yaml        # Agent configuration
├── environment.py     # Environment setup and wrappers
├── model.py           # Deep learning architectures
├── train.py           # Training script
├── enjoy.py           # Play with trained agent
├── export.py          # Export model to Hugging Face Hub
└── utils.py           # Utility functions
```

## Quick Start

### Training

Train a DQN agent with the default configuration:

```bash
python -m train
```

The training script will:
- Create a timestamped run directory in `runs/`.
- Save the configuration, checkpoints, and TensorBoard logs.
- Periodically evaluate the agent and save the best model.

### Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Environment
env_id: ALE/SpaceInvaders-v5
frame_skip: 5
frame_stack: 4
resized_frame: 84

# Training
training_steps: 10000000
n_eval_episodes: 10

# Exploration
warmup_steps: 100000
epsilon_start: 1.0
epsilon_end: 0.1
anneal_steps: 1000000

# Replay Buffer
buffer_size: 200000
batch_size: 32

# Learning
gamma: 0.99
lr: 0.00025
update_every: 25000
target_update_every: 10000

# DQN Variants
double_dqn: False    # Enable Double DQN
dueling: False       # Enable Dueling DQN
clip_rewards: False  # Clip rewards to [-1, 1]
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir runs/dqn_YYYY-MM-DD_HHhMMmSSs/
```

### Testing a Trained Agent

Watch your trained agent play:

```bash
python -m enjoy --artifact runs/dqn_YYYY-MM-DD_HHhMMmSSs/final_model.pt --num-episodes 5
```

### Exporting to Hugging Face Hub

Share your trained model:

```bash
python -m export \
    --username YOUR_HF_USERNAME \
    --repo-name dqn-spaceinvaders \
    --artifact-path runs/dqn_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 12
```

This will:
- Create a repository on Hugging Face Hub.
- Upload the model weights, configuration, and evaluation results.
- Generate and upload a replay movie.
- Create a model card with usage instructions.

## Algorithm Details

### DQN Architecture

The network consists of:
- 3 convolutional layers for feature extraction.
- 2 fully connected layers for Q-value estimation.
- Input: 4 stacked 84×84 grayscale frames.
- Output: Q-values for each action.

### Dueling DQN Architecture

Separates state value and action advantages:
- Shared convolutional backbone.
- Value stream: estimates state value V(s).
- Advantage stream: estimates action advantages A(s,a).
- Q(s,a) = V(s) + (A(s,a) - mean(A(s,a))).

### Training Process

1. **Warmup**: Random exploration for initial experiences.
2. **Epsilon Annealing**: Gradual reduction from exploration to exploitation.
3. **Experience Replay**: Sample random mini-batches from replay buffer.
4. **Target Network**: Separate network updated periodically for stability.
5. **Double DQN** (optional): Reduces overestimation by decoupling action selection and evaluation.

## Advanced Usage

### Loading a Trained Model

```python
import torch
from pathlib import Path
from utils import load_artifact

# Load model
config, env, agent = load_artifact(
    Path("runs/dqn_YYYY-MM-DD_HHhMMmSSs/final_model.pt"),
    render_mode="human"
)

# Use the agent
state, _ = env.reset()
action = agent.act(state, epsilon=0.0)
```

## License

This project is available under the MIT License.
