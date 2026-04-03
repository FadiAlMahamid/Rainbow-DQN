# Rainbow DQN for Atari Breakout

A complete implementation of **Rainbow DQN** (Hessel et al., 2018) that combines six key improvements over standard DQN into a single integrated agent, applied to the Atari Breakout environment.

## Overview

Rainbow DQN unifies the following techniques, each addressing a different limitation of standard DQN:

| Component | Addresses | Reference |
|---|---|---|
| **Double DQN** | Q-value overestimation | van Hasselt et al. (2016) |
| **Dueling Architecture** | State/action value separation | Wang et al. (2016) |
| **NoisyLinear Layers** | Efficient exploration | Fortunato et al. (2018) |
| **C51 Distributional RL** | Richer learning signal | Bellemare et al. (2017) |
| **N-Step Returns** | Faster credit assignment | Sutton (1988) |
| **Prioritized Experience Replay** | Focused experience reuse | Schaul et al. (2016) |

## Project Structure

```
Rainbow-DQN/
├── config.yaml          # All hyperparameters and settings
├── training_script.py   # Main training loop and entry point
├── dqn_agent.py         # Rainbow agent (combines all 6 improvements)
├── q_network.py         # CNN with Dueling + Noisy + Distributional output
├── noisy_linear.py      # NoisyLinear layer (factorised Gaussian noise)
├── replay_buffer.py     # PER buffer with Sum Tree
├── sum_tree.py          # Sum Tree data structure for O(log n) sampling
├── environment.py       # Atari preprocessing and frame stacking
├── utils.py             # Config loading, plotting, deployment
└── README.md
```

## How Rainbow Combines the Six Improvements

### 1. Network Architecture: Dueling + Noisy + Distributional

The Q-network (`q_network.py`) integrates three improvements:

- **Shared CNN backbone**: Standard DeepMind 2015 convolutional layers extract spatial features from stacked 84x84 grayscale frames.
- **Dueling streams**: After the CNN, features split into a **value stream** V(s) and an **advantage stream** A(s,a), combined as: `Q_atoms(s, a) = V_atoms(s) + A_atoms(s, a) − meanₐ'(A_atoms)`. This helps the network learn state values independently of action advantages.
- **NoisyLinear layers**: All fully connected layers use learned parametric noise (`noisy_linear.py`) instead of standard `nn.Linear`. This replaces epsilon-greedy exploration with state-dependent exploration that adapts automatically.
- **Distributional output**: Instead of scalar Q-values, each stream outputs a categorical distribution over 51 atoms (return values). Q-values are recovered as expected values: `Q(s, a) = Σᵢ pᵢ · zᵢ`.

### 2. Agent: Double DQN + N-Step + PER

The agent (`dqn_agent.py`) integrates the remaining three improvements:

- **Double DQN target**: The policy network selects the best next action, but the target network evaluates it. This decoupling reduces the overestimation bias inherent in standard DQN.
- **N-step returns**: Instead of storing single-step transitions, the agent accumulates N=3 transitions and computes the discounted return `Rₙ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂` before storing in the buffer. The Bellman target uses `γ^n` for proper discounting.
- **Prioritized Experience Replay**: Transitions are sampled proportional to their cross-entropy loss magnitude (proxy for TD error) using a Sum Tree. Importance sampling weights correct for the resulting bias, with beta annealed from 0.4 to 1.0 over training.

### 3. Loss: Distributional Cross-Entropy with IS Weights

The training loss combines C51's categorical projection with PER's importance sampling:

```
1. Project target distribution onto fixed support: Tz = Rₙ + γⁿ · z
2. Compute per-sample cross-entropy: Lᵢ = −Σⱼ target_distⱼ · log(pred_distⱼ)
3. Update PER priorities using Lᵢ as proxy for TD error
4. Apply IS-weighted mean: L = mean(wᵢ · Lᵢ)
```

### Key Formulas

```
N-step return:        Rₙ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂
Dueling combination:  Q(s, a) = V(s) + A(s, a) − meanₐ'(A(s, a'))
Q-value recovery:     Q(s, a) = Σᵢ pᵢ · zᵢ  (expected value from distribution)
Bellman projection:   Tzⱼ = Rₙ + γⁿ · zⱼ   (clipped to [V_min, V_max])
Cross-entropy loss:   Lᵢ = −Σⱼ target_distⱼ · log(pred_distⱼ)
IS-weighted loss:     L = mean(wᵢ · Lᵢ)
Beta annealing:       βₜ = min(1, β₀ + (1 − β₀) · t / T)
```

Note: Rainbow uses γⁿ (not γ) in the Bellman projection because the N-step return Rₙ already covers n steps of discounted rewards. Standalone C51 uses γ (single-step).

## Key Differences from Individual Variants

| Aspect | Standard Variants | Rainbow |
|---|---|---|
| Exploration | Epsilon-greedy with linear decay | NoisyLinear (no epsilon needed) |
| Replay | Uniform sampling | Prioritized with IS correction |
| Returns | 1-step TD | N-step (n=3) |
| Q-values | Scalar | Distributional (51 atoms) |
| Architecture | Single FC stream | Dueling (value + advantage) |
| Target selection | Varies by variant | Double DQN (policy selects, target evaluates) |
| Gradient clip | 1.0 | 10.0 (distributional loss has different scale) |
| Learning rate | 0.00025 | 0.0000625 (lower due to stronger gradient signal) |

## Requirements

- Python 3.10+
- PyTorch
- Gymnasium with Atari support (`ale-py`)
- NumPy, PyYAML, Matplotlib

## Installation

```bash
pip install torch gymnasium ale-py numpy pyyaml matplotlib
```

## Running

```bash
# Train from scratch
python training_script.py

# Resume training (set mode: "resume" in config.yaml)
python training_script.py

# Deploy trained agent (set mode: "deploy" in config.yaml)
python training_script.py
```

## Configuration

All hyperparameters are in `config.yaml`. Key settings:

- **C51**: `num_atoms: 51`, support range `[-10, 10]`
- **N-step**: `n: 3` (accumulate 3 transitions)
- **PER**: `alpha: 0.6`, `beta_start: 0.4`, annealed to 1.0
- **NoisyNets**: `sigma_init: 0.5` (initial noise scale)
- **Target network**: updated every 8000 learning steps
- **Training**: 2.5M steps (10M frames)

## References

- Hessel, M., et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." *AAAI*.
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
- van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
- Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.
- Fortunato, M., et al. (2018). "Noisy Networks for Exploration." *ICLR*.
- Bellemare, M., et al. (2017). "A Distributional Perspective on Reinforcement Learning." *ICML*.
- Sutton, R. (1988). "Learning to Predict by the Methods of Temporal Differences." *Machine Learning*.
- Schaul, T., et al. (2016). "Prioritized Experience Replay." *ICLR*.
