"""
A simple replay buffer for storing and sampling experiences.
"""
import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(self, buffer_size: int, seed: int | None = None):
        self.buffer = deque(maxlen=buffer_size)
        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a new experience to the buffer."""
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer.h"""
        self.buffer.clear()
