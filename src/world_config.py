"""World configuration loader for grid world environments."""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from src.tile import Tile


@dataclass
class WorldConfig:
    """Parsed grid world configuration.

    Built from a YAML config file. The grid is a 2D list of Tile objects
    where grid[row][col] describes the cell at that position.
    """

    grid_size: int
    start: Tuple[int, int]
    grid: List[List[Tile]]
    tile_map: Dict[Tuple[int, int], str] = field(default_factory=dict)

    @property
    def n_states(self):
        return self.grid_size * self.grid_size

    @property
    def n_actions(self):
        return 4  # N, S, E, W

    @classmethod
    def from_yaml(cls, path: str) -> 'WorldConfig':
        """Load a world configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A WorldConfig instance with the grid fully built.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        grid_size = data['grid_size']
        start = tuple(data['start'])

        # Parse tile type definitions
        tile_types = {}
        for name, props in data.get('tile_types', {}).items():
            redirect = tuple(props['redirect']) if 'redirect' in props else None
            tile_types[name] = Tile(
                reward=props.get('reward', -1.0),
                passable=props.get('passable', True),
                terminal=props.get('terminal', False),
                redirect=redirect,
            )

        # Build grid — every cell defaults to the 'empty' tile type
        empty = tile_types.get('empty', Tile())
        grid = [
            [Tile(reward=empty.reward, passable=empty.passable,
                  terminal=empty.terminal, redirect=empty.redirect)
             for _ in range(grid_size)]
            for _ in range(grid_size)
        ]

        # Place configured tiles onto the grid
        tile_map = {}
        for pos_str, type_name in data.get('tiles', {}).items():
            row, col = (int(x.strip()) for x in pos_str.split(','))
            assert type_name in tile_types, \
                f"Unknown tile type '{type_name}' at position ({row}, {col})"
            template = tile_types[type_name]
            grid[row][col] = Tile(
                reward=template.reward,
                passable=template.passable,
                terminal=template.terminal,
                redirect=template.redirect,
            )
            tile_map[(row, col)] = type_name

        # Validation
        assert 0 <= start[0] < grid_size and 0 <= start[1] < grid_size, \
            f"Start position {start} out of bounds for {grid_size}x{grid_size} grid"
        assert grid[start[0]][start[1]].passable, \
            "Start position must be on a passable tile"
        has_terminal = any(
            grid[r][c].terminal
            for r in range(grid_size) for c in range(grid_size)
        )
        assert has_terminal, "Grid must have at least one terminal tile (e.g. a goal)"

        return cls(grid_size=grid_size, start=start, grid=grid, tile_map=tile_map)
