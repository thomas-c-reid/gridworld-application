"""
5x5 Grid World Environment for RL agents.

Grid layout:
  1   2   3   4   5
1 [ ] [ ] [ ] [ ] [ ]
2 [S] [ ] [ ] [J] [ ]
3 [ ] [ ] [X] [X] [X]
4 [ ] [ ] [X] [*] [ ]
5 [ ] [ ] [ ] [ ] [G]

S = Start [2,1], G = Goal [5,5] (+10)
J = Jump [2,4] -> [4,4] (+5)
X = Obstacles [3,3], [4,3]
* = Jump destination [4,4]
"""

import numpy as np


class GridWorld:
    """5x5 grid world environment."""

    def __init__(self):
        self.grid_size = 5
        self.n_states = 25  # 5x5
        self.n_actions = 4  # North, South, East, West

        # Grid configuration (1-indexed in problem, 0-indexed here)
        self.start_pos = (1, 0)      # [2,1] in 1-indexed
        self.goal_pos = (4, 4)        # [5,5] in 1-indexed
        self.jump_from = (1, 3)       # [2,4] in 1-indexed
        self.jump_to = (3, 3)         # [4,4] in 1-indexed

        # Obstacles (black cells)
        self.obstacles = {(2, 2), (2, 3), (2, 4), (3, 2)}  # [3,3], [3,4], [3,5], [4,3] in 1-indexed

        self.current_state = self.pos_to_state(*self.start_pos)

    def pos_to_state(self, row, col):
        """Convert (row, col) to flat state index."""
        return row * self.grid_size + col

    def state_to_pos(self, state):
        """Convert flat state index to (row, col)."""
        return divmod(state, self.grid_size)

    def reset(self):
        """Reset environment to start position."""
        self.current_state = self.pos_to_state(*self.start_pos)
        return self.current_state

    def step(self, action):
        """
        Execute one action.

        Actions: 0=North, 1=South, 2=East, 3=West

        Returns:
            next_state (int): resulting state
            reward (float): reward for this transition
            done (bool): whether episode is finished
        """
        row, col = self.state_to_pos(self.current_state)

        # Apply action
        if action == 0:      # North
            row = max(0, row - 1)
        elif action == 1:    # South
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:    # East
            col = min(self.grid_size - 1, col + 1)
        elif action == 3:    # West
            col = max(0, col - 1)

        # Check if new position is an obstacle
        if (row, col) in self.obstacles:
            # Can't move there, stay in place
            row, col = self.state_to_pos(self.current_state)
            reward = -1.0
            self.current_state = self.pos_to_state(row, col)
            return self.current_state, reward, False

        new_state = self.pos_to_state(row, col)

        # Check for jump
        if (row, col) == self.jump_from:
            new_state = self.pos_to_state(*self.jump_to)
            reward = 5.0
            self.current_state = new_state
            return new_state, reward, False

        # Check for goal
        if (row, col) == self.goal_pos:
            reward = 10.0
            self.current_state = new_state
            return new_state, reward, True

        # Default penalty
        reward = -1.0
        self.current_state = new_state
        return new_state, reward, False

    def get_grid_visualization(self):
        """Return grid layout with state indices for visualization."""
        grid = np.arange(self.n_states).reshape(self.grid_size, self.grid_size)
        return grid
