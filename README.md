# DDPG Project

A clean, modular implementation of **Deep Deterministic Policy Gradient (DDPG)**
for continuous control, demonstrated on the `Pendulum-v1` environment from
[Gymnasium](https://gymnasium.farama.org/).

> **Reference:** Lillicrap et al., *"Continuous control with deep reinforcement
> learning"*, ICLR 2016. ([arXiv:1509.02971](https://arxiv.org/abs/1509.02971))

---

## Project Structure

```
DDPG_Project/
├── ddpg/
│   ├── __init__.py        # Package exports
│   ├── actor.py           # Actor (policy) network
│   ├── critic.py          # Critic (Q-value) network
│   ├── replay_buffer.py   # Experience replay buffer
│   ├── noise.py           # Ornstein-Uhlenbeck exploration noise
│   └── agent.py           # DDPGAgent – ties everything together
├── train.py               # Training entry-point (Pendulum-v1)
├── test.py                # Evaluation entry-point
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/xiaoshengdianzi/DDPG_Project.git
cd DDPG_Project

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

Key command-line options:

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | `300` | Number of training episodes |
| `--max-steps` | `200` | Maximum steps per episode |
| `--seed` | `42` | Random seed |
| `--save-dir` | `checkpoints` | Directory to save model weights |
| `--plot-dir` | `plots` | Directory to save reward curve plot |
| `--no-plot` | `False` | Disable reward plot |
| `--warmup` | `1000` | Random-action warm-up steps |
| `--device` | `cpu` | PyTorch device (`cpu` or `cuda`) |

After training, two sets of weights are written to `checkpoints/`:
- `ddpg_best_actor.pth` / `ddpg_best_critic.pth` – best episode
- `ddpg_final_actor.pth` / `ddpg_final_critic.pth` – last episode

A reward-curve plot is saved to `plots/training_rewards.png`.

---

## Evaluation

```bash
python test.py
```

Key command-line options:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-dir` | `checkpoints` | Directory with saved weights |
| `--filename` | `ddpg_best` | Weight file prefix |
| `--episodes` | `10` | Number of evaluation episodes |
| `--render` | `False` | Render the environment (needs a display) |
| `--seed` | `0` | Random seed |
| `--device` | `cpu` | PyTorch device |

---

## Algorithm Overview

DDPG is an **off-policy, model-free** algorithm for continuous action spaces.

1. **Actor** μ(s; θ^μ) outputs a deterministic action for each state.
2. **Critic** Q(s, a; θ^Q) estimates the action-value function.
3. **Target networks** θ^μ′, θ^Q′ are slowly updated via Polyak averaging
   (τ = 0.005) for training stability.
4. **Replay buffer** stores past transitions; mini-batches break temporal
   correlations.
5. **OU Noise** adds temporally-correlated exploration noise to actions
   during training.

### Update equations

```
y_i = r_i + γ · Q'(s_{i+1}, μ'(s_{i+1}))   (target Q)
L   = (1/N) Σ (y_i - Q(s_i, a_i))²          (critic loss)
J   = -(1/N) Σ Q(s_i, μ(s_i))               (actor loss, maximise Q)
```

---

## Expected Results

On `Pendulum-v1` a well-trained agent should reach a mean episode reward of
around **−200 to −150** within 300 episodes. The theoretical maximum is 0
(perfect swing-up and balance).

---

## License

[MIT](LICENSE)
