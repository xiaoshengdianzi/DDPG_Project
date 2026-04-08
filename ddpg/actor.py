"""
Actor network for DDPG.

The Actor maps states to deterministic actions.  A tanh output layer
scales actions into the range [-1, 1]; the agent rescales to the
environment's actual action bounds before stepping the environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Deterministic policy network: s -> a.

    Architecture: FC(state_dim -> 400) -> ReLU ->
                  FC(400 -> 300)        -> ReLU ->
                  FC(300 -> action_dim) -> tanh
    """

    def __init__(self, state_dim: int, action_dim: int, hidden1: int = 400, hidden2: int = 300):
        """
        Args:
            state_dim:  Dimensionality of the observation space.
            action_dim: Dimensionality of the action space.
            hidden1:    Number of units in the first hidden layer.
            hidden2:    Number of units in the second hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

        # Initialise output layer weights to a small range for stability
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: Batch of states, shape (batch, state_dim).

        Returns:
            Actions in [-1, 1], shape (batch, action_dim).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
