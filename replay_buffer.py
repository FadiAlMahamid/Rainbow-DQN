import numpy as np
from sum_tree import SumTree


# ===================================================================
# --- Prioritized Experience Replay Buffer ---
# ===================================================================
class PrioritizedReplayBuffer:
    """
    A memory-efficient Prioritized Experience Replay (PER) buffer.

    Instead of sampling transitions uniformly at random (as in standard ER),
    PER samples transitions with probability proportional to their priority.
    Priorities are based on TD error magnitude — transitions the agent
    learned poorly from are replayed more often.

    This buffer combines two key ideas:
    1. **Prioritized sampling via Sum Tree**: O(log n) sampling proportional
       to priority, with stratified segments for lower variance.
    2. **Importance Sampling (IS) weights**: Correct for the bias introduced
       by non-uniform sampling. IS weights are annealed via beta from
       partial correction (early training) to full correction (late training).

    Memory efficiency is preserved from the DQN-ER design: individual frames
    are stored as uint8 and stacked states are reconstructed on demand.

    Reference: Schaul et al. (2016), "Prioritized Experience Replay"
    """

    def __init__(self, capacity, frame_shape=(84, 84), stack_size=4,
                 alpha=0.6, priority_epsilon=1e-5):
        """
        Initializes the prioritized replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
            frame_shape (tuple): Height and width of a single preprocessed frame.
            stack_size (int): Number of frames stacked together to form a state.
            alpha (float): Prioritization exponent. 0 = uniform, 1 = full priority.
            priority_epsilon (float): Small constant added to TD errors to ensure
                all transitions have non-zero probability of being sampled.
        """
        self.capacity = capacity
        self.stack_size = stack_size
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon

        # pos: the index where the NEXT transition will be written (circular)
        self.pos = 0
        # size: how many transitions have been stored so far (capped at capacity)
        self.size = 0

        # Sum Tree for O(log n) priority-based sampling
        self.tree = SumTree(capacity)

        # Track the maximum priority seen so far. New transitions are inserted
        # with max priority so they are guaranteed to be sampled at least once.
        self.max_priority = 1.0

        # Pre-allocate fixed-size arrays for each component of a transition.
        # Using uint8 for frames saves 4x memory vs float32 (1 byte vs 4 bytes per pixel).
        # For 1M capacity with 84x84 frames: ~7 GB (uint8) vs ~28 GB (float32).
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer with maximum priority.

        New transitions get the highest priority so they are sampled at
        least once before their priority is updated based on actual TD error.

        Args:
            state (np.ndarray): The current frame-stacked state (stack_size, H, W).
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next frame-stacked state (stack_size, H, W).
            done (bool): Whether the episode terminated.
        """
        # Store only the newest frame from next_state (the last in the stack).
        # Frames from earlier in the stack were stored by previous transitions.
        self.frames[self.pos] = next_state[-1]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # Insert into the Sum Tree with max priority raised to alpha.
        # This ensures new transitions are sampled early and get their
        # priority updated based on actual TD error.
        self.tree.add(self.max_priority ** self.alpha)

        # Advance the write pointer (wraps around for circular buffer behavior)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stacked_state(self, idx):
        """
        Reconstructs a frame-stacked state ending at the given index.

        If a frame boundary crosses an episode done flag, earlier frames
        are zeroed out to avoid leaking information from a previous episode.

        Args:
            idx (int): The index of the last frame in the stack.

        Returns:
            np.ndarray: The stacked state of shape (stack_size, H, W).
        """
        # Build the list of frame indices that make up this stacked state.
        # For stack_size=4 and idx=10: indices = [7, 8, 9, 10]
        # Modular arithmetic handles wraparound at the buffer boundary.
        indices = [(idx - i) % self.capacity for i in reversed(range(self.stack_size))]
        stack = self.frames[indices]

        # Zero out frames that belong to a previous episode to prevent
        # the agent from "seeing" the end of one episode blended into the
        # start of another. If done[i] is True, frames at index i and
        # earlier came from a finished episode and must be blanked.
        # We only check up to stack_size-1 because the last frame (the
        # "current" frame) is always valid by definition.
        for i in range(self.stack_size - 1):
            if self.dones[indices[i]]:
                stack[:i + 1] = 0
                break
        return stack

    def sample(self, batch_size, beta):
        """
        Samples a mini-batch of transitions proportional to their priority.

        Uses stratified sampling: divides [0, total_priority) into batch_size
        equal segments and samples one value uniformly within each segment.
        This reduces variance compared to pure proportional sampling.

        Args:
            batch_size (int): The number of transitions to sample.
            beta (float): IS exponent for bias correction (annealed to 1.0).

        Returns:
            tuple: (states, actions, rewards, next_states, dones,
                    tree_indices, is_weights)
                - tree_indices: needed to update priorities after learning.
                - is_weights: importance sampling weights for loss correction.
        """
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Stratified sampling: divide the total priority into equal segments,
        # then sample one point uniformly within each segment.
        # This ensures coverage across the full priority spectrum.
        total_priority = self.tree.total()
        segment_size = total_priority / batch_size

        for i in range(batch_size):
            # Sample a random value within segment [i * segment_size, (i+1) * segment_size)
            low = segment_size * i
            high = segment_size * (i + 1)
            s = np.random.uniform(low, high)

            tree_idx, priority, data_idx = self.tree.get(s)
            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = priority

        # --- Compute Importance Sampling (IS) weights ---
        # P(i) = pᵢ / Σⱼ pⱼ (sampling probability)
        # wᵢ = (N · P(i))⁻ᵝ (raw IS weight)
        # Normalize by max(w) so weights are in [0, 1] — this only scales
        # gradients DOWN (never up), preventing gradient explosion.
        sampling_probs = priorities / total_priority
        is_weights = (self.size * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()
        is_weights = is_weights.astype(np.float32)

        # Reconstruct stacked states from individual frames:
        #   - "state" (s)  = stack ending one position BEFORE the stored index
        #   - "next_state" (s') = stack ending AT the stored index
        states = np.zeros((batch_size, self.stack_size, *self.frames.shape[1:]), dtype=np.float32)
        next_states = np.zeros((batch_size, self.stack_size, *self.frames.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(data_indices):
            states[i] = self._get_stacked_state((idx - 1) % self.capacity)
            next_states[i] = self._get_stacked_state(idx)

        actions = self.actions[data_indices]
        rewards = self.rewards[data_indices]
        # Convert boolean dones to float for use in the Bellman equation: (1 − doneₜ) · γ · Q
        dones = self.dones[data_indices].astype(np.float32)

        return states, actions, rewards, next_states, dones, tree_indices, is_weights

    def update_priorities(self, tree_indices, td_errors):
        """
        Updates the priorities of sampled transitions based on their TD errors.

        Transitions with larger TD errors get higher priority — the agent
        learns more from surprising transitions it predicted poorly.

        Args:
            tree_indices (np.ndarray): Tree indices returned from sample().
            td_errors (np.ndarray): Absolute TD errors from the learning step.
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            # pᵢ = (|δᵢ| + ε)^α
            # Epsilon ensures transitions with zero TD error can still be sampled.
            priority = (abs(td_error) + self.priority_epsilon) ** self.alpha
            self.tree.update(tree_idx, priority)

            # Track max priority for initializing new transitions
            self.max_priority = max(self.max_priority, abs(td_error) + self.priority_epsilon)

    def __len__(self):
        """Returns the current number of transitions stored in the buffer."""
        return self.size
