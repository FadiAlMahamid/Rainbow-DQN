import numpy as np


# ===================================================================
# --- Sum Tree Data Structure ---
# ===================================================================
class SumTree:
    """
    A binary tree where each leaf holds a priority value and each internal
    node holds the sum of its children. This structure enables two key
    operations in O(log n) time:

    1. **Priority-proportional sampling**: Given a random value s in
       [0, total_priority), traverse the tree to find the leaf whose
       cumulative priority range contains s. This is equivalent to
       sampling transitions with probability proportional to their priority.

    2. **Priority update**: When a transition's TD error changes after
       learning, update its leaf and propagate the change up to the root.

    The tree is stored as a flat array of size (2 * capacity - 1):
        - Internal nodes: indices [0, capacity - 2]
        - Leaf nodes:     indices [capacity - 1, 2 * capacity - 2]
        - Root node:      index 0 (holds the sum of ALL priorities)

    For a node at index i:
        - Left child:  2 * i + 1
        - Right child: 2 * i + 2
        - Parent:      (i - 1) // 2
    """

    def __init__(self, capacity):
        """
        Initializes the Sum Tree.

        Args:
            capacity (int): Maximum number of leaf nodes (transitions).
        """
        self.capacity = capacity

        # Circular write pointer — next leaf to be written
        self.write_pos = 0

        # Number of leaves currently populated
        self.size = 0

        # Flat array for the tree. Using float64 for precision when
        # summing millions of small priority values.
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def _propagate(self, idx, change):
        """
        Propagates a priority change from a leaf up to the root.

        After updating a leaf's priority, every ancestor's sum must be
        updated to keep the tree consistent.

        Args:
            idx (int): The tree index of the updated leaf.
            change (float): The difference (new_priority - old_priority).
        """
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, tree_idx, priority):
        """
        Updates the priority of a specific leaf node.

        Args:
            tree_idx (int): The tree index of the leaf to update.
            priority (float): The new priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority):
        """
        Adds a new priority to the tree at the current write position.

        If the tree is full, the oldest leaf is overwritten (circular buffer).

        Args:
            priority (float): The priority value for the new transition.
        """
        # Convert the data write position to the corresponding tree leaf index.
        # Leaf nodes start at index (capacity - 1) in the flat array.
        tree_idx = self.write_pos + self.capacity - 1

        self.update(tree_idx, priority)

        # Advance the circular write pointer
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx, s):
        """
        Traverses the tree downward to find the leaf whose cumulative
        priority range contains the value s.

        At each internal node, if s <= left child's value, go left;
        otherwise subtract the left child's value and go right.

        Args:
            idx (int): Current node index (start from 0 for root).
            s (float): The sampled value in [0, total_priority).

        Returns:
            int: The tree index of the selected leaf.
        """
        while True:
            left = 2 * idx + 1

            # If left child is beyond the array, we've reached a leaf
            if left >= len(self.tree):
                return idx

            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def get(self, s):
        """
        Samples a leaf by traversing the tree with value s.

        Args:
            s (float): A value in [0, total_priority) used to select a leaf.

        Returns:
            tuple: (tree_idx, priority, data_idx)
                - tree_idx: The leaf's index in the tree array.
                - priority: The leaf's priority value.
                - data_idx: The corresponding index in the data buffer.
        """
        s = np.clip(s, 0, self.total() - 1e-8)
        tree_idx = self._retrieve(0, s)
        # Convert tree index back to data index
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx

    def total(self):
        """Returns the total sum of all priorities (stored at the root)."""
        return self.tree[0]

    def __len__(self):
        """Returns the number of transitions currently stored."""
        return self.size
