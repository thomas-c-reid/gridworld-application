"""Tile-driven grid world environment for RL agents.

The grid layout, rewards, and special tiles are all defined by a
WorldConfig loaded from YAML — no tile logic is hardcoded here.
"""

import numpy as np
from src.world_config import WorldConfig


class GridWorld:
    """Configurable grid world environment."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.grid = config.grid
        self.current_state = self.pos_to_state(*config.start)

    # ── coordinate helpers ──────────────────────────────────────

    def pos_to_state(self, row, col):
        """Convert (row, col) to flat state index."""
        return row * self.grid_size + col

    def state_to_pos(self, state):
        """Convert flat state index to (row, col)."""
        return divmod(state, self.grid_size)

    # ── environment interface ───────────────────────────────────

    def reset(self):
        """Reset environment to start position."""
        self.current_state = self.pos_to_state(*self.config.start)
        return self.current_state

    def step(self, action):
        """Execute one action.

        Actions: 0=North, 1=South, 2=East, 3=West

        Returns:
            (next_state, reward, done)
        """
        row, col = self.state_to_pos(self.current_state)

        # Apply movement with boundary clamping
        if action == 0:      # North
            new_row, new_col = max(0, row - 1), col
        elif action == 1:    # South
            new_row, new_col = min(self.grid_size - 1, row + 1), col
        elif action == 2:    # East
            new_row, new_col = row, min(self.grid_size - 1, col + 1)
        elif action == 3:    # West
            new_row, new_col = row, max(0, col - 1)

        tile = self.grid[new_row][new_col]

        # Impassable tile (wall/obstacle) — stay in place
        if not tile.passable:
            return self.current_state, tile.reward, False

        # Redirect tile (jump/teleport)
        if tile.redirect is not None:
            self.current_state = self.pos_to_state(*tile.redirect)
            return self.current_state, tile.reward, tile.terminal

        # Normal tile (possibly terminal like a goal)
        self.current_state = self.pos_to_state(new_row, new_col)
        return self.current_state, tile.reward, tile.terminal

    def get_grid_visualization(self):
        """Return grid of state indices for visualization."""
        return np.arange(self.n_states).reshape(self.grid_size, self.grid_size)

    # ── backward-compat properties (used by visualizer) ─────────

    @property
    def start_pos(self):
        return self.config.start

    @property
    def goal_pos(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c].terminal:
                    return (r, c)
        return None

    @property
    def jump_from(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c].redirect is not None:
                    return (r, c)
        return None

    @property
    def jump_to(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c].redirect is not None:
                    return self.grid[r][c].redirect
        return None

    @property
    def obstacles(self):
        return {
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if not self.grid[r][c].passable
        }
