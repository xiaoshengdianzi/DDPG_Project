"""
DDPG Agent.

Ties together the Actor, Critic, target networks, Replay Buffer, and
OU Noise into a single trainable agent following the algorithm of
Lillicrap et al., "Continuous control with deep reinforcement learning"
(ICLR 2016).

Key design choices:
  - Soft target-network updates: θ' ← τ*θ + (1-τ)*θ'
  - Actor loss:  -mean(Q(s, μ(s)))
  - Critic loss: MSE between target Q and current Q
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import Critic
from .noise import OUNoise
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent.

    Args:
        state_dim:      Dimensionality of the observation space.
        action_dim:     Dimensionality of the action space.
        action_high:    Upper bound of the action space (used for scaling).
        action_low:     Lower bound of the action space (used for scaling).
        gamma:          Discount factor (default 0.99).
        tau:            Soft update coefficient for target networks (default 0.005).
        actor_lr:       Learning rate for the actor optimizer (default 1e-4).
        critic_lr:      Learning rate for the critic optimizer (default 1e-3).
        buffer_size:    Replay buffer capacity (default 1 000 000).
        batch_size:     Mini-batch size for updates (default 64).
        noise_sigma:    OU-noise sigma (default 0.2).
        device:         PyTorch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 1_000_000,
        batch_size: int = 64,
        noise_sigma: float = 0.2,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Action scaling helpers
        self.action_high = torch.FloatTensor(action_high).to(self.device)
        self.action_low  = torch.FloatTensor(action_low).to(self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias  = (self.action_high + self.action_low) / 2.0

        # ---- Networks ----
        self.actor        = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self._hard_update(self.actor_target, self.actor)

        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self._hard_update(self.critic_target, self.critic)

        # ---- Optimisers ----
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---- Replay buffer ----
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, device=self.device)

        # ---- Exploration noise ----
        self.noise = OUNoise(size=action_dim, sigma=noise_sigma)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Choose an action given the current state.

        Args:
            state:     Current environment observation.
            add_noise: Whether to add OU noise for exploration.

        Returns:
            Action clipped to [action_low, action_high].
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t)
        self.actor.train()

        # Scale from [-1, 1] to environment's action range
        action = action * self.action_scale + self.action_bias
        action = action.cpu().numpy().flatten()

        if add_noise:
            action += self.noise.sample() * self.action_scale.cpu().numpy()

        return np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self) -> Optional[tuple]:
        """Sample a mini-batch and update actor and critic.

        Returns:
            (critic_loss, actor_loss) as Python floats, or None if the
            buffer does not yet contain enough transitions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            # Next actions from the target actor (scaled)
            next_actions_raw = self.actor_target(next_states)
            next_actions = next_actions_raw * self.action_scale + self.action_bias
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        actions_pred_raw = self.actor(states)
        actions_pred = actions_pred_raw * self.action_scale + self.action_bias
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft target updates ----
        self._soft_update(self.actor_target,  self.actor,  self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, directory: str, filename: str = "ddpg") -> None:
        """Save actor and critic weights to *directory*.

        Args:
            directory: Path to the output directory (created if absent).
            filename:  Base name for the saved files.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))
        print(f"[DDPGAgent] Model saved to '{directory}/'")

    def load(self, directory: str, filename: str = "ddpg") -> None:
        """Load actor and critic weights from *directory*.

        Args:
            directory: Path to the directory containing the saved files.
            filename:  Base name used when the files were saved.
        """
        actor_path  = os.path.join(directory, f"{filename}_actor.pth")
        critic_path = os.path.join(directory, f"{filename}_critic.pth")
        self.actor.load_state_dict(torch.load(actor_path,  map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        # Synchronise target networks after loading
        self._hard_update(self.actor_target,  self.actor)
        self._hard_update(self.critic_target, self.critic)
        print(f"[DDPGAgent] Model loaded from '{directory}/'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
        """Polyak-average the *source* network into the *target* network.

        θ_target ← τ * θ_source + (1 - τ) * θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
        """Copy all weights from *source* to *target* (τ = 1)."""
        target.load_state_dict(source.state_dict())
