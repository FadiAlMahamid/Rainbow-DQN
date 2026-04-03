import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================================================================
# --- NoisyLinear Layer (Fortunato et al., 2018) ---
# ===================================================================
class NoisyLinear(nn.Module):
    """
    A linear layer with factorised Gaussian noise added to its weights and biases.

    Instead of using epsilon-greedy for exploration, NoisyLinear injects
    learned parametric noise directly into the network. The noise magnitude
    (sigma) is a trainable parameter — the network learns how much to
    explore in each part of the weight space. As training progresses,
    sigma shrinks in well-learned regions and the agent naturally shifts
    from exploration to exploitation without any epsilon schedule.

    Uses factorised Gaussian noise (Section 3.2 of Fortunato et al., 2018)
    which is more parameter-efficient than independent noise:
        w = μʷ + σʷ · (εₒᵤₜ ⊗ εᵢₙ)
        b = μᵇ + σᵇ · εₒᵤₜ

    where εᵢₙ and εₒᵤₜ are factorised noise vectors transformed by f(x) = sign(x) · √|x|.
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        """
        Initializes the NoisyLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            sigma_init (float): Initial value for the noise scale parameter sigma.
                Higher values = more initial exploration. 0.5 is the default
                from the paper for factorised noise.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters: mu (mean) and sigma (noise scale) for weights and biases.
        # These are optimized by gradient descent alongside the rest of the network.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers are not model parameters — they are resampled each forward pass.
        # register_buffer ensures they move to the correct device (GPU/MPS) automatically.
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initializes mu and sigma parameters.

        μ is initialized uniformly in [-1/√p, 1/√p] where p = in_features,
        following the factorised noise initialization from the paper.
        σ is initialized to σ₀ / √p so that the initial noise
        magnitude is proportional to the layer size.
        """
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def _scale_noise(self, size):
        """
        Generates factorised noise using the f(x) = sign(x) · √|x| transform.

        This transform ensures the noise has heavier tails than standard Gaussian,
        which provides more robust exploration. Factorised noise uses O(p + q)
        random numbers instead of O(p * q) for independent noise.

        Args:
            size (int): The dimension of the noise vector.

        Returns:
            torch.Tensor: The scaled noise vector.
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        Resamples the factorised Gaussian noise.

        Called once per learning step so that different forward passes within
        the same step see the same noise (important for consistent Q-value
        estimates during action selection and target computation).
        """
        # Generate two independent noise vectors (factorised approach)
        epsilon_in = self._scale_noise(self.in_features)    # input-side noise
        epsilon_out = self._scale_noise(self.out_features)  # output-side noise

        # Outer product creates the full weight noise matrix from two vectors:
        # (out_features, 1) * (1, in_features) -> (out_features, in_features)
        # This is O(p + q) random numbers instead of O(p * q).
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        Forward pass with noisy weights: w = μ + σ · ε.

        During training, noise is added to explore; during evaluation,
        only the mean weights (mu) are used for deterministic action selection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output with noisy linear transformation applied.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            # Eval mode: use only the learned mean weights (no noise)
            return F.linear(x, self.weight_mu, self.bias_mu)
