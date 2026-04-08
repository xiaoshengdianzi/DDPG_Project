"""
Testing / evaluation script for a trained DDPG agent on Pendulum-v1.

Usage
-----
    python test.py [--checkpoint-dir DIR] [--filename BASE] [--episodes N]
                   [--render] [--seed N]

Loads the saved actor and critic weights from *checkpoint_dir*, runs the
agent in evaluation mode (no exploration noise), and reports per-episode
and mean reward.

Examples
--------
    # Evaluate the best checkpoint saved by train.py
    python test.py --checkpoint-dir checkpoints --filename ddpg_best

    # Render in a window (requires a display / GUI)
    python test.py --render
"""

import argparse

import gymnasium as gym
import numpy as np

from ddpg import DDPGAgent


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG agent on Pendulum-v1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing saved model weights")
    parser.add_argument("--filename", type=str, default="ddpg_best",
                        help="Base filename used when saving (e.g. 'ddpg_best' or 'ddpg_final')")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment in a window")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu / cuda)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    render_mode = "human" if args.render else None
    env = gym.make("Pendulum-v1", render_mode=render_mode)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low  = env.action_space.low

    # ---- Load agent ----
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        device=args.device,
    )
    agent.load(args.checkpoint_dir, filename=args.filename)

    # ---- Run episodes ----
    episode_rewards = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        episode_reward = 0.0

        for _ in range(args.max_steps):
            # No noise during evaluation
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {ep:3d} | Reward: {episode_reward:8.2f}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward  = np.std(episode_rewards)
    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Mean reward : {mean_reward:.2f}")
    print(f"  Std  reward : {std_reward:.2f}")
    print(f"  Best episode: {max(episode_rewards):.2f}")
    print(f"  Worst episode: {min(episode_rewards):.2f}")


if __name__ == "__main__":
    evaluate(parse_args())
