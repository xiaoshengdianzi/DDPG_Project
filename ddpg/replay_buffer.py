"""
Experience Replay Buffer for DDPG.

Stores (state, action, reward, next_state, done) tuples and supports
random batch sampling.  The buffer uses a circular deque so that the
oldest transitions are discarded once capacity is reached.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store.
        device:   PyTorch device to move sampled tensors to.
    """

    def __init__(self, capacity: int = 1_000_000, device: torch.device = torch.device("cpu")):
        self.buffer: deque = deque(maxlen=capacity)
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            state:      Current observation.
            action:     Action taken.
            reward:     Scalar reward received.
            next_state: Observation after the action.
            done:       Whether the episode ended.
        """
        self.buffer.append((state, action, float(reward), next_state, float(done)))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors,
            each with a leading batch dimension.
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t     = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
