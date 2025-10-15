import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(self, buffer_size: int, seed: int = None):
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
        state = state.astype(np.uint8)
        next_state = next_state.astype(np.uint8)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
