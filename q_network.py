import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from noisy_linear import NoisyLinear  # --- [RAINBOW: NOISY] Replaces nn.Linear ---


# ===================================================================
# --- Rainbow DQN Q-Network (Dueling + Noisy + Distributional) ---
# ===================================================================
class QNetwork(nn.Module):
    """
    Rainbow CNN for distributional Q-value approximation.

    Combines three architectural improvements over the standard DeepMind
    2015 Nature CNN:

    1. **Dueling Architecture** (Wang et al., 2016): Splits the fully
       connected layers into a value stream V(s) and an advantage stream
       A(s, a), combined as Q(s, a) = V(s) + A(s, a) − meanₐ'(A). This helps
       the network learn state values independently of action advantages.

    2. **NoisyLinear Layers** (Fortunato et al., 2018): Replaces standard
       nn.Linear layers with NoisyLinear, injecting learned parametric
       noise into the weights. This replaces epsilon-greedy exploration
       with state-dependent exploration that adapts automatically.

    3. **Distributional Output** (Bellemare et al., 2017): Instead of
       outputting scalar Q-values, each stream outputs a categorical
       distribution over num_atoms possible return values. Q-values are
       recovered as expected values: Q(s, a) = Σᵢ pᵢ · zᵢ.

    Output shape: (batch, num_actions, num_atoms) — log-probabilities.
    """
    def __init__(self, input_shape, num_actions, num_atoms=51, sigma_init=0.5):  # --- [RAINBOW: C51] num_atoms parameter ---
        """
        Initializes the Rainbow network layers.

        Args:
            input_shape (tuple): The shape of the input state (e.g., (4, 84, 84)).
            num_actions (int): The number of possible actions.
            num_atoms (int): The number of atoms in the categorical distribution.
            sigma_init (float): Initial noise scale for NoisyLinear layers.
        """
        super(QNetwork, self).__init__()

        self.num_actions = num_actions
        self.num_atoms = num_atoms  # --- [RAINBOW: C51] Distributional output dimension ---

        # Assumes input is (Channels, Height, Width)
        # For Atari, input_shape is (4, 84, 84) -> (FrameStack, H, W)
        in_channels = input_shape[0]

        # Convolutional layers shared between both streams.
        # These remain standard (non-noisy) convolutions — noise is only added
        # to the fully connected layers where it directly affects action selection.
        # Each layer progressively reduces spatial dimensions while increasing feature depth:
        #   Input:  (4, 84, 84)  -- 4 stacked grayscale frames
        #   conv1:  (32, 20, 20) -- large 8x8 filters with stride 4 capture coarse features
        #   conv2:  (64, 9, 9)   -- 4x4 filters with stride 2 capture mid-level patterns
        #   conv3:  (64, 7, 7)   -- 3x3 filters with stride 1 capture fine-grained details
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened feature map after conv layers.
        # A dummy forward pass determines the output size dynamically, so the code
        # doesn't break if input_shape changes (e.g., different screen_size).
        # For (4, 84, 84) input: conv_out_size = 64 * 7 * 7 = 3136.
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)

        # --- [RAINBOW: DUELING] Value Stream ---
        # Estimates V(s): a distribution over the value of being in state s.
        # Many states have similar value regardless of which action is chosen (e.g.,
        # when the ball is far from the paddle). The value stream captures this
        # shared component so the advantage stream only learns relative differences.
        # Outputs num_atoms log-probabilities for the value distribution.
        # Vanilla DQN: single fc (3136 -> 512) + output (512 -> num_actions)
        self.value_fc = NoisyLinear(conv_out_size, 512, sigma_init=sigma_init)       # --- [RAINBOW: NOISY] NoisyLinear instead of nn.Linear ---
        self.value_out = NoisyLinear(512, num_atoms, sigma_init=sigma_init)           # --- [RAINBOW: C51] Outputs num_atoms instead of 1 ---

        # --- [RAINBOW: DUELING] Advantage Stream ---
        # Estimates A(s,a): a distribution over the relative advantage of each action.
        # Outputs (num_actions * num_atoms) log-probabilities, reshaped to
        # (batch, num_actions, num_atoms) before combining with the value stream.
        # Vanilla DQN: no separate advantage stream
        self.advantage_fc = NoisyLinear(conv_out_size, 512, sigma_init=sigma_init)    # --- [RAINBOW: NOISY] NoisyLinear instead of nn.Linear ---
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms, sigma_init=sigma_init)  # --- [RAINBOW: C51] Outputs num_actions * num_atoms ---

    def _get_conv_out(self, x):
        """Helper function to calculate the output size of the conv layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the tensor, except for the batch dimension, to get the total size
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        """
        Forward pass returning log-probabilities of the return distribution.

        The shared convolutional features are split into value and advantage
        streams (both noisy), then recombined using the dueling formula
        applied per-atom:
            Q_atoms(s, a) = V_atoms(s) + A_atoms(s, a) − meanₐ'(A_atoms(s, a'))

        Finally, log_softmax is applied across atoms for each action to produce
        valid log-probability distributions.

        Args:
            x (torch.Tensor): The input batch of states.

        Returns:
            torch.Tensor: Log-probabilities of shape (batch, num_actions, num_atoms).
        """
        batch_size = x.size(0)

        # Normalize pixel values from [0, 255] to [0.0, 1.0] for stable training
        x = x / 255.0

        # Pass through shared convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the conv layers into a 1D vector
        x = x.view(batch_size, -1)

        # --- [RAINBOW: DUELING] Value stream ---
        # Vanilla DQN: single path from conv features to Q-values
        # Output shape: (batch, num_atoms) -> (batch, 1, num_atoms) for broadcasting
        value = F.relu(self.value_fc(x))
        value = self.value_out(value).view(batch_size, 1, self.num_atoms)

        # --- [RAINBOW: DUELING] Advantage stream ---
        # Output shape: (batch, num_actions, num_atoms)
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage).view(batch_size, self.num_actions, self.num_atoms)

        # --- [RAINBOW: DUELING] Combine streams using the dueling formula per-atom ---
        # Vanilla DQN: returns Q-values directly from a single stream
        #   Q_atoms(s, a) = V_atoms(s) + A_atoms(s, a) − meanₐ'(A_atoms(s, a'))
        # Mean-centering of advantages ensures V genuinely represents state value
        # and makes the decomposition identifiable (unique).
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # --- [RAINBOW: C51] Apply log_softmax across the atoms dimension ---
        # Vanilla DQN: returns raw Q-values (no softmax needed)
        # This converts raw logits into log-probabilities, ensuring that
        # for each action, the probabilities across all atoms sum to 1.
        return F.log_softmax(q_atoms, dim=2)

    def reset_noise(self):  # --- [RAINBOW: NOISY] No equivalent in vanilla DQN ---
        """
        Resamples noise in all NoisyLinear layers.

        Called once per learning step so that the agent explores with
        fresh noise perturbations. Between resets, the same noise is
        used for consistent Q-value estimates within a single step.
        """
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
