# Q-Learning & SARSA Grid World

Reinforcement learning agents solving a 5x5 grid world environment using tabular Q-Learning (off-policy) and SARSA (on-policy).

## Grid Layout

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
Deep Learning CW2/
  main.py                  # Entry point - alpha sweep, training, visualization
  models/
    base_agent.py          # Abstract base class for RL agents
    q_learning_agent.py    # Off-policy Q-learning (Q-table)
    sarsa_agent.py         # On-policy SARSA (Q-table)
  src/
    gridworld.py           # 5x5 grid environment
    trainer.py             # Training loop with convergence checking
    visualizer.py          # Matplotlib plots for all tasks
```

## Setup

```bash
pip install numpy matplotlib
```

## Usage

```bash
python main.py
```

This runs:
1. **Alpha sweep** - trains Q-learning with learning rates [1.0, 0.5, 0.1, 0.01]
2. **Algorithm comparison** - trains both Q-learning and SARSA with the best alpha
3. **Visualization** - saves all plots to `plots/`

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rates (alpha) | 1.0, 0.5, 0.1, 0.01 |
| Discount factor (gamma) | 0.9 |
| Epsilon start | 1.0 |
| Epsilon min | 0.01 |
| Epsilon decay | 0.995 per episode |
| Max episodes | 100 |
| Convergence window | 30 episodes |
| Convergence threshold | 10.0 avg reward |

## Output

Plots are saved to `plots/`:

- `task_c_alpha_comparison.png` - learning rate comparison
- `task_d_algorithm_comparison.png` - Q-learning vs SARSA
- `task_e_learning_curves.png` - training progress
- `task_e_q_progression.png` - Q-learning trajectory over training
- `task_e_sarsa_progression.png` - SARSA trajectory over training
- `task_f_gridworld_qlearning.png` - state values and policy grid
- `task_f_gridworld_sarsa.png` - state values and policy grid
- `bonus_qheatmap_action_*.png` - per-action Q-value heatmaps
