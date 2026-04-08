"""
Ornstein-Uhlenbeck (OU) Noise for action-space exploration.

OU noise produces temporally correlated samples that are better suited
for physical control problems than independent Gaussian noise, because
consecutive actions tend to be smooth.

Reference: Uhlenbeck & Ornstein (1930).
"""

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process.

    dx_t = theta * (mu - x_t) * dt + sigma * dW_t

    Args:
        size:   Dimensionality of the action space.
        mu:     Long-run mean of the process (default 0).
        theta:  Rate of mean reversion (default 0.15).
        sigma:  Volatility / noise scale (default 0.2).
        dt:     Time-step size (default 1e-2).
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
    ):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        """Reset the internal state to the mean."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """Generate the next noise sample.

        Returns:
            Noise vector of shape (size,).
        """
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        )
        self.state = self.state + dx
        return self.state.copy()
