"""
Critic network for DDPG.

The Critic estimates the action-value function Q(s, a).  The action is
injected into the network after the first hidden layer, following the
original DDPG paper (Lillicrap et al., 2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """Action-value network: (s, a) -> Q.

    Architecture: FC(state_dim -> 400) -> ReLU ->
                  FC(400 + action_dim -> 300) -> ReLU ->
                  FC(300 -> 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden1: int = 400, hidden2: int = 300):
        """
        Args:
            state_dim:  Dimensionality of the observation space.
            action_dim: Dimensionality of the action space.
            hidden1:    Number of units in the first hidden layer (state only).
            hidden2:    Number of units in the second hidden layer (state + action).
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        # Action is concatenated *after* the first layer
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        # Initialise output layer weights to a small range for stability
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state:  Batch of states,  shape (batch, state_dim).
            action: Batch of actions, shape (batch, action_dim).

        Returns:
            Estimated Q-values, shape (batch, 1).
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)  # inject action
        x = F.relu(self.fc2(x))
        return self.fc3(x)
