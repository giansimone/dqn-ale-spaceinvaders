"""
Agent module for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

from model import DQN, DuelingDQN
from buffer import ReplayBuffer
from environment import make_env


class Agent():
    """Agent that interacts with and learns from the environment."""

    def __init__(
        self,
        state_size: Tuple[int, ...],
        action_size: int,
        config: dict
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        network = DuelingDQN if config["dueling"] else DQN

        self.policy_net = network(state_size, action_size).to(self.config["device"])
        self.target_net = network(state_size, action_size).to(self.config["device"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(
            self.policy_net.parameters(),
            lr=config["lr"],
        )
        self.memory = ReplayBuffer(config["buffer_size"], seed=config["seed"])
        self._n_step = 0

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Save experience in replay memory and learn every update_every time steps."""
        self.memory.push(state, action, reward, next_state, done)
        self._n_step += 1

        if (self._n_step % self.config["update_every"] != 0 or
        len(self.memory) < self.config["batch_size"]):
            return None

        experiences = self.memory.sample(self.config["batch_size"])
        loss = self.learn(experiences)
        return loss

    def act(self, state: np.ndarray, epsilon: float = 0.) -> int:
        """Return actions for given state as per current policy."""
        if random.random() <= epsilon:
            return random.randrange(self.action_size)

        state = np.array(state, dtype=np.float32)
        tensor_state = torch.from_numpy(state).float().unsqueeze(0).to(self.config["device"]) / 255.

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(tensor_state)
        self.policy_net.train()

        return action_values.argmax(1).item()

    def learn(self, experiences: Tuple[np.ndarray, ...]) -> float:
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.config["device"]) / 255.
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.config["device"])
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.config["device"])
        next_states = torch.from_numpy(next_states).float().to(self.config["device"]) / 255.
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.config["device"])

        if self.config["clip_rewards"]:
            rewards = torch.clamp(rewards, -1., 1.)

        with torch.no_grad():
            if self.config["double_dqn"]:
                best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                q_targets_next = self.target_net(next_states).gather(1, best_actions)
            else:
                q_targets_next = self.target_net(next_states).max(1)[0].unsqueeze(1)

            q_targets = rewards + (self.config["gamma"] * q_targets_next * (1 - dones))

        q_expected = self.policy_net(states).gather(1, actions)

        loss = F.smooth_l1_loss(q_expected, q_targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if self._n_step % self.config["target_update_every"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        return loss.item()

    def save_model(self, filepath: Path) -> None:
        """Save the model to the specified filepath."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimiser": self.optimiser.state_dict(),
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"|--> Error saving model at {filepath}: {e}")

    def load_model(self, filepath: Path) -> None:
        """Load the model from the specified filepath."""
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.config["device"])
            policy_state = checkpoint.get("policy_net")
            target_state = checkpoint.get("target_net")
            optimiser_state = checkpoint.get("optimiser")

            self.policy_net.load_state_dict(policy_state, strict=False)
            self.target_net.load_state_dict(target_state, strict=False)
            self.optimiser.load_state_dict(optimiser_state)
            print(f"Model loaded successfully from {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")

    @property
    def n_step(self):
        """Get the current step count."""
        return self._n_step


def evaluate_agent(agent: Agent, config: dict) -> list[np.float64, np.float64]:
    """Evaluate the agent over a number of episodes."""
    env = make_env(config)
    scores = []

    for episode in range(config["n_eval_episodes"]):
        state, _ = env.reset(seed=config["seed"] + episode)
        score = 0.
        done = False

        while not done:
            action = agent.act(state, epsilon=0.)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += float(reward)

        scores.append(score)

    env.close()

    avg_score = np.mean(scores, dtype=np.float64)
    std_score = np.std(scores, dtype=np.float64)

    return avg_score, std_score



