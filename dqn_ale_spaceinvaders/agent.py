"""
Agent module for Deep Q-Learning Network (DQN) on ALE Space Invaders.
"""
import random
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from model import DQN, DuelingDQN
from buffer import ReplayBuffer


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
        self.device = config["device"]

        Network = DuelingDQN if config["dueling"] else DQN

        self.policy_net = Network(state_size, action_size).to(self.device)
        self.target_net = Network(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(
            self.policy_net.parameters(),
            lr=config["lr"],
        )
        self.memory = ReplayBuffer(config["buffer_size"], seed=config["seed"])
        self.n_steps = 0

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        self.memory.push(state, action, reward, next_state, done)
        self.n_steps += 1

        if (self.n_steps % self.config["update_every"] != 0 or
        len(self.memory) < self.config["batch_size"]):
            return None

        self.n_steps = 0
        experiences = self.memory.sample(self.config["batch_size"])
        loss = self.learn(experiences)
        return loss

    def act(self, state: np.ndarray, epsilon: float = 0.) -> int:
        if random.random() <= epsilon:
            return random.randrange(self.action_size)

        state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        return action_values.argmax(1).item()

    def learn(self, experiences: Tuple[np.ndarray, ...]) -> float:
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.device) / 255.
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device) / 255.
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

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

        if self.config["use_huber_loss"]:
            loss = F.smooth_l1_loss(q_expected, q_targets)
        else:
            loss = F.mse_loss(q_expected, q_targets)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.config["max_norm_grad"],
        )
        self.optimiser.step()

        return loss.item()

    def save_model(self, filepath: str) -> None:
        try:
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimiser": self.optimiser.state_dict(),
            }, filepath)
        except Exception as e:
            print(f"|--> Error saving model: {e}")

    def load_model(self, filepath: str) -> None:
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
        except Exception as e:
            print(f"|--> Error loading model: {e}")
