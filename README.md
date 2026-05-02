# Q-Learning & SARSA Grid World

Reinforcement learning agents solving a configurable grid world environment using tabular Q-Learning (off-policy) and SARSA (on-policy).

## Grid Layout (CW2 default)

```
  1   2   3   4   5
1 [ ] [ ] [ ] [ ] [ ]
2 [S] [ ] [ ] [J] [ ]
3 [ ] [ ] [X] [X] [X]
4 [ ] [ ] [X] [*] [ ]
5 [ ] [ ] [ ] [ ] [G]

S = Start    G = Goal (+10)
J = Jump (+5, teleports to *)
X = Obstacle (-1, blocks movement)
```

## Project Structure

```
gridworld-application/
├── configs/
│   └── cw2_5x5.yaml              # world layout as YAML config
├── models/
│   ├── base_agent.py              # abstract base class for RL agents
│   ├── q_learning_agent.py        # off-policy Q-learning
│   └── sarsa_agent.py             # on-policy SARSA
├── src/
│   ├── tile.py                    # Tile dataclass (reward, passable, terminal, redirect)
│   ├── world_config.py            # loads YAML → WorldConfig (grid of Tiles)
│   ├── gridworld.py               # tile-driven environment
│   ├── experiment.py              # ExperimentRunner (sweep, compare, visualize)
│   ├── trainer.py                 # single-agent training loop
│   └── visualizer.py              # matplotlib plots
├── main.py                        # thin CLI — argparse → ExperimentRunner
└── visualizer_app.py              # Streamlit interactive dashboard
```

## Setup

```bash
pip install numpy matplotlib pyyaml
```

## Usage

### Run the full experiment (defaults)

```bash
python main.py
```

This runs the alpha sweep, algorithm comparison, and visualization with the default CW2 config.

### Specify a world config

```bash
python main.py --config configs/cw2_5x5.yaml
```

### Override hyperparameters

```bash
python main.py --gamma 0.95 --episodes 200 --alphas 0.5 0.1 0.05
```

### All CLI options

```
--config          World config YAML          (default: configs/cw2_5x5.yaml)
--output-dir      Plot output directory       (default: plots)
--alphas          Learning rates to sweep     (default: 1.0 0.5 0.1 0.01)
--gamma           Discount factor             (default: 0.9)
--epsilon-start   Initial exploration rate    (default: 1.0)
--epsilon-min     Minimum exploration rate    (default: 0.01)
--epsilon-decay   Epsilon decay per episode   (default: 0.95)
--show            Show plots interactively    (for PyCharm screenshots)
--episodes        Max training episodes       (default: 100)
--conv-window     Convergence window size     (default: 30)
--conv-threshold  Convergence reward threshold (default: 10.0)
```

## World Config Format

Worlds are defined in YAML. Each cell is a tile type placed on a grid:

```yaml
grid_size: 5

tile_types:
  empty:
    reward: -1.0
    passable: true
  wall:
    passable: false
    reward: -1.0
  goal:
    reward: 10.0
    passable: true
    terminal: true
  jump:
    reward: 5.0
    passable: true
    redirect: [3, 3]    # teleport destination (0-indexed row, col)

start: [1, 0]           # agent start position (0-indexed)

# Only list non-empty cells — everything else defaults to 'empty'
tiles:
  "1,3": jump
  "4,4": goal
  "2,2": wall
  "2,3": wall
  "2,4": wall
  "3,2": wall
```

### Tile properties

| Property   | Type        | Default | Description                            |
|------------|-------------|---------|----------------------------------------|
| `reward`   | float       | -1.0    | Reward when agent enters this tile     |
| `passable` | bool        | true    | Whether the agent can move onto it     |
| `terminal` | bool        | false   | Ends the episode when entered          |
| `redirect` | [row, col]  | null    | Teleports agent to this position       |

## Output

Plots are saved to `plots/` (or `--output-dir`):

- `task_c_alpha_comparison.png` — learning rate comparison
- `task_d_algorithm_comparison.png` — Q-learning vs SARSA
- `task_e_learning_curves.png` — training progress
- `task_e_q_progression.png` — Q-learning trajectory over training
- `task_e_sarsa_progression.png` — SARSA trajectory over training
- `task_f_gridworld_qlearning.png` — state values and policy grid
- `task_f_gridworld_sarsa.png` — state values and policy grid
- `bonus_qheatmap_action_*.png` — per-action Q-value heatmaps
