"""
DQN and Dueling DQN model definitions for ALE Space Invaders environment.
"""
from typing import Tuple

import torch
from torch import nn


class DQN(nn.Module):
    """Vanilla Deep Q-Network (DQN) Architecture."""

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            action_size: int,
            hidden_dim: int = 512
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out_size(input_shape)

        self.head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x):
        """Forward pass through the DQN network."""
        x = self.conv(x).flatten(start_dim=1)
        return self.head(x)

    def _get_conv_out_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return x.flatten().shape[0]


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network (DQN) Architecture."""

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            action_size: int,
            hidden_dim: int = 512
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out_size(input_shape)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x):
        """Forward pass through the Dueling DQN network."""
        x = self.conv(x).flatten(start_dim=1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def _get_conv_out_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return x.flatten().shape[0]
