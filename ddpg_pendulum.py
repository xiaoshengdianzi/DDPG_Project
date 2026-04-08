import os
import random
from collections import deque

import numpy as np
from tqdm import tqdm

try:
    import gymnasium as gym
except ImportError:  # fallback for older installs
    import gym

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = float(action_bound)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
        discrete=False,
        noise_type="gaussian",
    ):
        self.device = device
        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim, activation=F.relu, out_fn=out_fn).to(
            self.device
        )
        self.target_actor = TwoLayerFC(
            state_dim, action_dim, hidden_dim, activation=F.relu, out_fn=out_fn
        ).to(self.device)
        self.critic = TwoLayerFC(state_dim + action_dim, 1, hidden_dim).to(self.device)
        self.target_critic = TwoLayerFC(state_dim + action_dim, 1, hidden_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.action_bound = float(action_bound)
        self.action_dim = action_dim
        self.noise_type = noise_type
        self.ou_state = np.zeros(self.action_dim, dtype=np.float32)

    @torch.no_grad()
    def take_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_t).cpu().numpy().squeeze(0)
        action = action + self._sample_noise()
        return np.clip(action, -self.action_bound, self.action_bound)

    def reset_noise(self):
        self.ou_state = np.zeros(self.action_dim, dtype=np.float32)

    def _sample_noise(self):
        if self.noise_type == "ou":
            # Ornstein-Uhlenbeck process for temporally correlated noise
            theta = 0.15
            dt = 1.0
            self.ou_state = self.ou_state + theta * (-self.ou_state) * dt + self.sigma * np.sqrt(dt) * np.random.randn(
                self.action_dim
            )
            return self.ou_state
        return self.sigma * np.random.randn(self.action_dim)

    def update(self, batch):
        state, action, reward, next_state, done = batch
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.target_actor(next_state_t)
            target_q = self.target_critic(torch.cat([next_state_t, next_action], dim=1))
            target = reward_t + self.gamma * (1.0 - done_t) * target_q

        current_q = self.critic(torch.cat([state_t, action_t], dim=1))
        critic_loss = F.mse_loss(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(torch.cat([state_t, self.actor(state_t)], dim=1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


def make_env(env_id, seed=0, render_mode=None):
    if render_mode is None:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode=render_mode)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    env.action_space.seed(seed)
    return env


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
        return next_state, reward, done, info
    next_state, reward, done, info = result
    return next_state, reward, done, info


def moving_average(values, window):
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def train(
    env_id="Pendulum-v1",
    num_episodes=200,
    episodes_per_iter=20,
    hidden_dim=64,
    actor_lr=5e-4,
    critic_lr=5e-3,
    gamma=0.98,
    tau=0.005,
    buffer_capacity=10000,
    minimal_size=1000,
    batch_size=64,
    sigma=0.01,
    device=None,
    noise_type="gaussian",
):
    env = make_env(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDPG(
        state_dim,
        action_dim,
        hidden_dim,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
        discrete=False,
        noise_type=noise_type,
    )
    replay = ReplayBuffer(buffer_capacity)

    returns = []
    num_iters = num_episodes // episodes_per_iter
    leftover = num_episodes % episodes_per_iter
    total_iters = num_iters + (1 if leftover else 0)

    episode_idx = 0
    for iteration in range(total_iters):
        this_count = episodes_per_iter if iteration < num_iters else leftover
        if this_count == 0:
            continue
        iter_bar = tqdm(range(this_count), desc=f"Iteration {iteration}", ncols=90)
        for _ in iter_bar:
            state = reset_env(env)
            agent.reset_noise()
            done = False
            episode_return = 0.0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = step_env(env, action)
                replay.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if len(replay) >= minimal_size:
                    batch = replay.sample(batch_size)
                    agent.update(batch)

            returns.append(episode_return)
            episode_idx += 1
            iter_bar.set_postfix(episode=episode_idx, ep_return=f"{episode_return:.3f}")

    env.close()
    return returns, agent, state_dim, action_dim, action_bound


@torch.no_grad()
def evaluate(agent, env_id, num_episodes=5, render=False):
    render_mode = "human" if render else None
    env = make_env(env_id, render_mode=render_mode)
    returns = []
    for _ in range(num_episodes):
        state = reset_env(env)
        done = False
        episode_return = 0.0
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            action = agent.actor(state_t).cpu().numpy().squeeze(0)
            action = np.clip(action, -agent.action_bound, agent.action_bound)
            next_state, reward, done, _ = step_env(env, action)
            state = next_state
            episode_return += reward
        returns.append(episode_return)
    env.close()
    return returns


def save_weights(agent, path):
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "target_actor": agent.target_actor.state_dict(),
            "target_critic": agent.target_critic.state_dict(),
        },
        path,
    )


def load_weights(agent, path):
    checkpoint = torch.load(path, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.target_actor.load_state_dict(checkpoint["target_actor"])
    agent.target_critic.load_state_dict(checkpoint["target_critic"])


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env_id = "Pendulum-v1"
    weights_path = "ddpg_pendulum_weights.pth"

    if os.path.exists(weights_path):
        # Build a matching agent and load existing weights for rendering.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DDPG(
            state_dim=3,
            action_dim=1,
            hidden_dim=64,
            action_bound=2.0,
            sigma=0.0,
            actor_lr=5e-4,
            critic_lr=5e-3,
            tau=0.005,
            gamma=0.98,
            device=device,
            discrete=False,
            noise_type="gaussian",
        )
        load_weights(agent, weights_path)
    else:
        returns, agent, state_dim, action_dim, action_bound = train(noise_type="ou")
        episodes = list(range(len(returns)))
        mv_returns = moving_average(returns, 9)
        mv_episodes = list(range(len(mv_returns)))

        plt.plot(episodes, returns, color="steelblue", label="Returns")
        plt.plot(mv_episodes, mv_returns, color="tomato", label="Moving Average (9)")
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("DDPG on Pendulum-v1")
        plt.legend()
        plt.show()

        save_weights(agent, weights_path)

    eval_returns = evaluate(agent, env_id, num_episodes=5, render=True)
    print(f"Eval average return: {float(np.mean(eval_returns)):.2f}")


if __name__ == "__main__":
    main()
