"""Tile definition for grid world cells."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Tile:
    """A single cell in the grid world.

    Attributes:
        reward: Reward received when entering this tile.
        passable: Whether the agent can move onto this tile.
        terminal: Whether entering this tile ends the episode.
        redirect: Optional destination (row, col) to teleport the agent.
    """
    reward: float = -1.0
    passable: bool = True
    terminal: bool = False
    redirect: Optional[Tuple[int, int]] = None
