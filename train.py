"""
Training script for DDPG on Pendulum-v1 (Gymnasium).

Usage
-----
    python train.py [--episodes N] [--max-steps N] [--seed N]
                    [--save-dir DIR] [--no-plot]

The script trains a DDPGAgent on the Pendulum-v1 environment, saves the
model checkpoints, and (optionally) produces a reward-curve plot.

Typical training time on a modern CPU: ~5 minutes for 300 episodes.
"""

import argparse
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from ddpg import DDPGAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG on Pendulum-v1")
    parser.add_argument("--episodes",   type=int,   default=300,          help="Number of training episodes")
    parser.add_argument("--max-steps",  type=int,   default=200,          help="Max steps per episode")
    parser.add_argument("--seed",       type=int,   default=42,           help="Random seed")
    parser.add_argument("--save-dir",   type=str,   default="checkpoints",help="Directory to save model weights")
    parser.add_argument("--plot-dir",   type=str,   default="plots",      help="Directory to save reward plot")
    parser.add_argument("--no-plot",    action="store_true",              help="Disable reward plot saving")
    parser.add_argument("--warmup",     type=int,   default=1000,         help="Steps before learning begins")
    parser.add_argument("--device",     type=str,   default="cpu",        help="PyTorch device (cpu / cuda)")
    return parser.parse_args()


def plot_rewards(rewards: list, save_path: str) -> None:
    """Save a smoothed reward curve to *save_path*."""
    window = min(10, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.3, label="Raw reward")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"Smoothed ({window}-ep window)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DDPG – Pendulum-v1 Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[train] Reward plot saved to '{save_path}'")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # ---- Environment ----
    env = gym.make("Pendulum-v1")
    state_dim  = env.observation_space.shape[0]   # 3
    action_dim = env.action_space.shape[0]         # 1
    action_high = env.action_space.high             # [2.0]
    action_low  = env.action_space.low              # [-2.0]

    print(f"[train] Env: Pendulum-v1 | state_dim={state_dim} | "
          f"action_dim={action_dim} | device={args.device}")

    # ---- Agent ----
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        device=args.device,
    )

    # ---- Training ----
    episode_rewards: list = []
    total_steps = 0
    best_reward = float("-inf")

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        agent.noise.reset()
        episode_reward = 0.0

        for _ in range(args.max_steps):
            # Random actions during warm-up to pre-fill the buffer
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        # Console logging every 10 episodes
        if ep % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {ep:4d}/{args.episodes} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg(10): {avg_reward:8.2f} | "
                f"Buffer: {len(agent.replay_buffer):6d}"
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(args.save_dir, filename="ddpg_best")

    # Save final model
    agent.save(args.save_dir, filename="ddpg_final")
    env.close()

    # ---- Plot ----
    if not args.no_plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_rewards(episode_rewards, os.path.join(args.plot_dir, "training_rewards.png"))

    print(f"\n[train] Done. Best episode reward: {best_reward:.2f}")


if __name__ == "__main__":
    train(parse_args())
