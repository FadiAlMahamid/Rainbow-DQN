import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from q_network import QNetwork  # --- [RAINBOW: DUELING+NOISY+C51] Combined architecture ---
from replay_buffer import PrioritizedReplayBuffer  # --- [RAINBOW: PER] Vanilla DQN uses uniform ReplayBuffer ---


# ===================================================================
# --- The Rainbow DQN Agent Class ---
# ===================================================================
class DQNAgent:
    """
    The Rainbow DQN agent combining six key improvements over standard DQN:

    1. **Double DQN** (van Hasselt et al., 2016): Policy network selects the
       best next action, target network evaluates it. Reduces overestimation.
    2. **Dueling Architecture** (Wang et al., 2016): Separate value and advantage
       streams in the network, combined as Q = V + A − meanₐ'(A).
    3. **NoisyLinear Layers** (Fortunato et al., 2018): Learned parametric noise
       replaces epsilon-greedy exploration with state-dependent exploration.
    4. **C51 Distributional RL** (Bellemare et al., 2017): Learns a full return
       distribution instead of scalar Q-values. Richer learning signal.
    5. **N-Step Returns** (Sutton, 1988): Accumulates N transitions before
       bootstrapping, providing faster credit assignment.
    6. **Prioritized Experience Replay** (Schaul et al., 2016): Samples
       transitions proportional to TD error magnitude with IS correction.

    Reference: Hessel et al. (2018). "Rainbow: Combining Improvements in
    Deep Reinforcement Learning." AAAI.
    """
    def __init__(self, state_shape, action_size, learning_rate, gamma,
                 device, buffer_capacity, batch_size, learning_starts,
                 target_update_freq, optimizer="adam", grad_clip_norm=10.0,
                 clip_rewards=True, sigma_init=0.5,
                 # C51 parameters
                 num_atoms=51, v_min=-10.0, v_max=10.0,
                 # N-step parameters
                 n_steps=3,
                 # PER parameters
                 per_alpha=0.6, per_beta_start=0.4,
                 per_beta_frames=2500000, priority_epsilon=1e-5):
        """
        Initializes the Rainbow DQN Agent.

        Note: No epsilon parameters are needed — exploration is handled
        entirely by the NoisyLinear layers in the Q-network.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.clip_rewards = clip_rewards

        # Total steps taken across all training episodes
        self.total_steps = 0

        # Counter for tracking when to update the target network
        self.learn_step_counter = 0

        # --- [RAINBOW: C51] Distribution Parameters ---
        # Vanilla DQN: no distributional parameters (scalar Q-values)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Δz is the spacing between adjacent atoms in the support.
        # For 51 atoms over [-10, 10]: Δz = 20 / 50 = 0.4
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # The support is the fixed set of atom values: z = [v_min, ..., v_max].
        # Each atom represents a possible return value; the network predicts
        # the probability that the true return equals each atom.
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)

        # --- [RAINBOW: N-STEP] N-Step Parameters ---
        # Vanilla DQN: stores single-step transitions directly
        self.n_steps = n_steps
        # Local buffer for accumulating N transitions before computing the
        # discounted N-step return and storing in the main replay buffer.
        self.n_step_buffer = deque(maxlen=n_steps)

        # --- [RAINBOW: PER] Prioritized Replay Parameters ---
        # Vanilla DQN: no beta annealing (uniform sampling)
        self.per_beta = per_beta_start
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames

        # --- [RAINBOW: DUELING+NOISY+C51] Networks ---
        # Vanilla DQN: QNetwork outputs (batch, num_actions) scalar Q-values
        # Rainbow: outputs (batch, num_actions, num_atoms) log-probabilities
        # using a dueling architecture with NoisyLinear layers.
        self.policy_network = QNetwork(
            state_shape, action_size, num_atoms, sigma_init=sigma_init
        ).to(self.device)

        self.target_network = QNetwork(
            state_shape, action_size, num_atoms, sigma_init=sigma_init
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # --- Optimizer ---
        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # --- [RAINBOW: PER] Prioritized Replay Buffer ---
        # Vanilla DQN: ReplayBuffer(capacity, frame_shape, stack_size) with uniform sampling
        frame_shape = state_shape[1:]  # (H, W) from (stack_size, H, W)
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_capacity, frame_shape=frame_shape, stack_size=state_shape[0],
            alpha=per_alpha, priority_epsilon=priority_epsilon
        )

    def choose_action(self, state, training=True):  # --- [RAINBOW: NOISY] No epsilon parameter needed ---
        """
        Chooses an action using the noisy distributional network.

        No epsilon-greedy exploration is needed — the NoisyLinear layers
        inject learned noise into the forward pass, causing the network to
        output slightly different Q-value distributions each time. The agent
        selects the action with the highest expected Q-value (mean of the
        predicted distribution).

        Vanilla DQN: uses epsilon-greedy (random action with probability epsilon).

        In deployment mode (training=False), eval mode disables noise for
        deterministic action selection.

        Args:
            state (np.ndarray): The current state observation.
            training (bool): If True, increment step counter and update beta.
                If False, use eval mode for deterministic actions.

        Returns:
            int: The selected action index.
        """
        if training:
            self.total_steps += 1
            self._update_beta()  # --- [RAINBOW: PER] Anneal beta; vanilla DQN decays epsilon here ---
        else:
            self.policy_network.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # --- [RAINBOW: C51] Forward pass returns log-probabilities: (1, num_actions, num_atoms) ---
            log_probs = self.policy_network(state_tensor)

            # --- [RAINBOW: C51] Convert distribution to expected Q-values ---
            probs = log_probs.exp()
            q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)

            action = q_values.argmax().item()

        if not training:
            self.policy_network.train()

        return action

    def store_transition(self, state, action, reward, next_state, done):  # --- [RAINBOW: N-STEP] N-step accumulation ---
        """
        Accumulates transitions in the N-step buffer and stores completed
        N-step transitions in the PER replay buffer.

        Instead of storing raw single-step transitions, we accumulate N
        transitions and compute the discounted N-step return:
            Rₙ = rₜ + γ · rₜ₊₁ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁

        The stored transition becomes (sₜ, aₜ, Rₙ, sₜ₊ₙ, doneₜ₊ₙ),
        which provides faster credit assignment during learning.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode terminated.
        """
        # Reward clipping normalizes the learning signal.
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        # Append to the local N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If the episode ended, flush all remaining transitions in the buffer
        # with partial (< N) step returns so no experience is wasted.
        if done:
            self._flush_n_step_buffer()
            return

        # Wait until we have accumulated N transitions.
        if len(self.n_step_buffer) < self.n_steps:
            return

        # Compute the N-step discounted return:
        # Rₙ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁
        n_step_return = sum(
            self.n_step_buffer[i][2] * (self.gamma ** i)
            for i in range(self.n_steps)
        )

        # The transition stored in the replay buffer uses:
        # - State from the OLDEST transition in the buffer (s_t)
        # - Action from the OLDEST transition (a_t)
        # - The accumulated N-step return (R_n)
        # - Next state from the NEWEST transition (s_{t+n})
        # - Done flag from the NEWEST transition
        first_state = self.n_step_buffer[0][0]
        first_action = self.n_step_buffer[0][1]
        last_next_state = self.n_step_buffer[-1][3]
        last_done = self.n_step_buffer[-1][4]

        self.replay_buffer.push(first_state, first_action, n_step_return,
                                last_next_state, last_done)

    def _flush_n_step_buffer(self):
        """
        Flushes remaining transitions at episode end with partial returns.

        When an episode terminates, there may be fewer than N transitions
        left in the buffer. We compute partial returns (k-step for k < N)
        for each remaining transition so no experience is wasted.
        """
        while len(self.n_step_buffer) > 0:
            k = len(self.n_step_buffer)

            # Compute k-step return for the remaining transitions
            k_step_return = sum(
                self.n_step_buffer[i][2] * (self.gamma ** i)
                for i in range(k)
            )

            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]

            self.replay_buffer.push(first_state, first_action, k_step_return,
                                    last_next_state, last_done)

            # Remove the oldest transition and repeat for the rest
            self.n_step_buffer.popleft()

    def project_distribution(self, next_dist, rewards_t, dones_t, gamma_n):
        """
        Projects the Bellman-updated target distribution onto the fixed atom support.

        The Bellman equation with N-step returns updates each atom:
            Tzⱼ = Rₙ + γⁿ · zⱼ  (for non-terminal states)
        But these shifted atoms generally don't land exactly on our fixed support
        atoms. This method distributes each shifted atom's probability mass to
        its two nearest neighbors, proportional to their distance.

        Example: if atom z=3.0 gets shifted to Tz=3.7, and the nearest
        support atoms are zₗ=3.6 and zᵤ=4.0 (Δz=0.4), then:
          - z_l gets 75% of the probability (closer)
          - z_u gets 25% of the probability (farther)

        Args:
            next_dist (torch.Tensor): Target distribution for the greedy action,
                shape (batch, num_atoms). Each row sums to 1.0.
            rewards_t (torch.Tensor): N-step returns for the batch, shape (batch,).
            dones_t (torch.Tensor): Terminal flags for the batch, shape (batch,).
                1.0 for terminal states, 0.0 otherwise.
            gamma_n (float): Discount factor raised to the n-th power (γⁿ).
                --- [RAINBOW: N-STEP] Vanilla C51 uses gamma (single-step) ---

        Returns:
            torch.Tensor: The projected target distribution, shape (batch, num_atoms).
        """
        batch_size = next_dist.size(0)

        # Apply Bellman equation to each atom: Tzⱼ = Rₙ + γⁿ · zⱼ
        # rewards_t: (batch,) -> (batch, 1), support: (num_atoms,) -> (1, num_atoms)
        t_z = rewards_t.unsqueeze(1) + (1 - dones_t.unsqueeze(1)) * gamma_n * self.support.unsqueeze(0)
        # Clamp to ensure projected atoms stay within the support range [v_min, v_max]
        t_z = t_z.clamp(min=self.v_min, max=self.v_max)

        # Find the fractional index of each projected atom in the support.
        # frac_idx tells us where Tz lands relative to the support atoms.
        # e.g., frac_idx=3.7 means Tz is 70% of the way between atom 3 and atom 4.
        frac_idx = (t_z - self.v_min) / self.delta_z  # (batch, num_atoms)

        # Lower and upper neighboring atom indices
        lower = frac_idx.floor().long()  # (batch, num_atoms)
        upper = frac_idx.ceil().long()   # (batch, num_atoms)

        # Clamp indices to valid range [0, num_atoms - 1]
        lower = lower.clamp(0, self.num_atoms - 1)
        upper = upper.clamp(0, self.num_atoms - 1)

        # Distribute probability mass to lower and upper neighbors.
        #   Lower neighbor weight: (upper - frac_idx) — larger when Tz is closer to lower
        #   Upper neighbor weight: (frac_idx - lower) — larger when Tz is closer to upper
        #
        # We use index_add_ with flattened tensors for efficient vectorized
        # distribution of probability mass across all batch elements at once.
        # offset shifts indices so each batch element writes to its own row
        # in the flattened 1D array.
        offset = torch.linspace(
            0, (batch_size - 1) * self.num_atoms, batch_size
        ).long().unsqueeze(1).expand(batch_size, self.num_atoms).to(self.device)

        # Flatten to 1D for index_add_ operations
        proj_dist = torch.zeros(batch_size * self.num_atoms, device=self.device)

        # Add mass to lower neighbor: p_lower += p_target * (upper - frac_idx)
        proj_dist.index_add_(0, (lower + offset).view(-1),
                             (next_dist * (upper.float() - frac_idx)).view(-1))
        # Add mass to upper neighbor: p_upper += p_target * (frac_idx - lower)
        proj_dist.index_add_(0, (upper + offset).view(-1),
                             (next_dist * (frac_idx - lower.float())).view(-1))

        # Reshape back to (batch, num_atoms)
        return proj_dist.view(batch_size, self.num_atoms)

    def learn(self):
        """
        Performs one gradient descent step using the Rainbow loss.

        Combines:
        - C51 distributional loss (cross-entropy between projected target
          and predicted distributions)
        - Double DQN target selection (policy selects, target evaluates)
        - N-step Bellman projection (γⁿ discounting)
        - PER importance sampling weights (bias correction)

        After each step, noise is resampled in the policy network for
        fresh exploration on the next action selection.

        Returns:
            float: The cross-entropy loss value for this gradient step.
        """
        # --- [RAINBOW: PER] Sample a prioritized mini-batch with IS weights ---
        # Vanilla DQN: states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, tree_indices, is_weights = \
            self.replay_buffer.sample(self.batch_size, beta=self.per_beta)

        # 2. Convert numpy arrays to PyTorch tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # ---------------------------------------------------------------
        # --- [RAINBOW: C51] 3. Get Predicted Log-Probabilities ---
        # Vanilla DQN: q_values = policy_network(states); q = q_values.gather(1, actions)
        # ---------------------------------------------------------------
        # policy_network returns: (batch, num_actions, num_atoms)
        predicted_log_probs_all = self.policy_network(states_t)

        # Select the distribution for the action that was actually taken.
        # actions_t: (batch,) -> (batch, 1, num_atoms) for gather
        actions_expanded = actions_t.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms)
        # gather along the action dimension (dim=1), then squeeze: (batch, num_atoms)
        predicted_log_probs = predicted_log_probs_all.gather(1, actions_expanded).squeeze(1)

        # ---------------------------------------------------------------
        # 4. Compute the Projected Target Distribution
        # ---------------------------------------------------------------
        with torch.no_grad():
            # --- [RAINBOW: DOUBLE] 4a. Policy network selects the best next action ---
            # Vanilla DQN: next_q = target_network(next_states); next_action = next_q.argmax()
            # Double DQN: policy network selects, target network evaluates
            predicted_next_log_probs = self.policy_network(next_states_t)
            predicted_next_probs = predicted_next_log_probs.exp()

            # Compute expected Q-values for action selection
            # Q(s', a) = Σᵢ pᵢ · zᵢ across atoms
            next_q_values = (predicted_next_probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            # Select the action with highest expected Q-value (policy network decides)
            next_actions = next_q_values.argmax(dim=1)

            # --- [RAINBOW: DOUBLE] 4b. Target network evaluates the selected action ---
            # Vanilla DQN: uses target_network for both selection and evaluation
            target_log_probs_all = self.target_network(next_states_t)
            target_probs_all = target_log_probs_all.exp()

            # Get the distribution for the action selected by the policy network
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms)
            target_dist = target_probs_all.gather(1, next_actions_expanded).squeeze(1)

            # --- [RAINBOW: C51+N-STEP] 4c. Project the Bellman-updated distribution ---
            # Vanilla DQN: target = rₜ + γ · maxₐ Q(sₜ₊₁)
            # --- [RAINBOW: N-STEP] γⁿ because the N-step return covers n steps ---
            gamma_n = self.gamma ** self.n_steps
            projected_target_dist = self.project_distribution(target_dist, rewards_t, dones_t, gamma_n)

        # ---------------------------------------------------------------
        # --- [RAINBOW: C51] 5. Cross-Entropy Loss ---
        # Vanilla DQN: loss = MSE(Q_predicted, Q_target)
        # ---------------------------------------------------------------
        # Per-element cross-entropy: −Σⱼ projected_targetⱼ · log(predictedⱼ)
        elementwise_loss = -(projected_target_dist * predicted_log_probs).sum(dim=1)

        # --- [RAINBOW: PER] Update priorities using loss as TD error proxy ---
        # Vanilla DQN: no priority updates
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # --- [RAINBOW: PER] Apply IS weights to correct for non-uniform sampling bias ---
        # Vanilla DQN: loss = loss.mean() (uniform weighting)
        loss = (elementwise_loss * is_weights_t).mean()

        # ---------------------------------------------------------------
        # 6. Gradient Descent
        # ---------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # --- [RAINBOW: NOISY] 7. Resample noise after each learning step ---
        # Vanilla DQN: no noise resampling (uses epsilon-greedy instead)
        # Fresh noise ensures the agent explores differently on the next action
        # selection.
        self.policy_network.reset_noise()

        # 8. Periodically update the target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def _update_beta(self):  # --- [RAINBOW: PER] No equivalent in vanilla DQN ---
        """
        Anneals the PER beta parameter linearly from beta_start to 1.0.

        Beta controls how much importance sampling correction is applied.
        Low beta early on allows aggressive prioritization (acceptable bias
        when the policy is still random). By the end of training, beta = 1.0
        provides full bias correction (needed for convergence guarantees).
        """
        self.per_beta = min(
            1.0,
            self.per_beta_start + (self.total_steps * (1.0 - self.per_beta_start) / self.per_beta_frames)
        )

    def update_target_network(self):
        """
        Hard update: copies policy network weights to the target network.
        Called periodically (every target_update_freq learning steps) to keep
        the target network's distribution estimates stable between updates.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model(self, filepath):
        """Saves a full training checkpoint to file."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load_model(self, filepath):
        """Loads a training checkpoint from file into agent components."""
        # map_location ensures tensors are loaded onto the correct device (CPU/GPU/MPS).
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Support both new checkpoint format (dict with keys) and legacy format
        # (raw state_dict) for backward compatibility with older saved models.
        if isinstance(checkpoint, dict) and 'policy_network' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.learn_step_counter = checkpoint['learn_step_counter']
        else:
            # Legacy format: checkpoint is just the policy network state_dict
            self.policy_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(self.policy_network.state_dict())

        self.target_network.eval()
        self.policy_network.train()
