"""
Visualization utilities for RL training results.

All plots use matplotlib for PyCharm screenshot compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_gridworld(env, agent, title="Grid World - State Values and Policy"):
    """
    Visualize the grid world with state values and policy arrows.

    Task (f): Visualize state values in grid cells.

    Args:
        env: GridWorld environment
        agent: trained agent (has get_state_values() and get_policy())
        title (str): plot title
    """
    state_values = agent.get_state_values()
    policy = agent.get_policy()

    # Action names and directions
    action_names = ['N', 'S', 'E', 'W']  # North, South, East, West
    action_dirs = {0: (-0.5, 0), 1: (0.5, 0), 2: (0, 0.5), 3: (0, -0.5)}

    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize state values for coloring
    norm = Normalize(vmin=state_values.min(), vmax=state_values.max())
    cmap = plt.cm.RdYlGn

    # Draw grid cells
    for state in range(env.n_states):
        row, col = env.state_to_pos(state)

        # Cell background color based on state value
        value = state_values[state]
        color = cmap(norm(value))
        rect = patches.Rectangle((col, row), 1, 1, linewidth=2,
                                 edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)

        # Draw state value
        ax.text(col + 0.5, row + 0.7, f"{value:.1f}",
               ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw policy arrow
        action = policy[state]
        arrow_text = action_names[action]
        dy, dx = action_dirs[action]
        ax.arrow(col + 0.5, row + 0.5, dx * 0.15, dy * 0.15,
                head_width=0.08, head_length=0.08, fc='black', ec='black', alpha=0.6)

    # Highlight special cells
    start_row, start_col = env.start_pos
    goal_row, goal_col = env.goal_pos
    jump_from_row, jump_from_col = env.jump_from
    jump_to_row, jump_to_col = env.jump_to

    # Start (red marker)
    ax.add_patch(patches.Circle((start_col + 0.5, start_row + 0.5), 0.15,
                               color='red', zorder=10))

    # Goal (blue marker)
    ax.add_patch(patches.Circle((goal_col + 0.5, goal_row + 0.5), 0.15,
                               color='cyan', zorder=10))

    # Jump from (yellow star outline)
    ax.plot(jump_from_col + 0.5, jump_from_row + 0.5, 'y*', markersize=15, zorder=10)

    # Jump to (yellow star)
    ax.plot(jump_to_col + 0.5, jump_to_row + 0.5, 'y*', markersize=12, zorder=10)

    # Obstacles (black)
    for obs_row, obs_col in env.obstacles:
        rect = patches.Rectangle((obs_col, obs_row), 1, 1, linewidth=2,
                                 edgecolor='black', facecolor='black', alpha=0.9)
        ax.add_patch(rect)

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(1, env.grid_size + 1))
    ax.set_yticks(range(1, env.grid_size + 1))
    ax.set_xticklabels(range(1, env.grid_size + 1))
    ax.set_yticklabels(range(1, env.grid_size + 1))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar for state values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='State Value')

    plt.tight_layout()
    return fig, ax


def plot_learning_curve(metrics_list, labels, title="Learning Curves"):
    """
    Plot reward over episodes with rolling average.

    Task (c)/(e): Show training results.

    Args:
        metrics_list: list of metrics dicts (each from trainer.train())
        labels: list of labels (e.g., ['α=1.0', 'α=0.5', ...])
        title (str): plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for metrics, label in zip(metrics_list, labels):
        rewards = np.array(metrics['episode_rewards'])

        # Raw rewards
        ax.plot(rewards, alpha=0.3, linewidth=0.5)

        # Rolling average
        window = 5
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        episodes = np.arange(len(rolling_avg)) + window - 1
        ax.plot(episodes, rolling_avg, label=label, linewidth=2, marker='o', markersize=4)

    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Convergence threshold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_alpha_comparison(results_dict, title="Learning Rate Comparison"):
    """
    Compare convergence for different learning rates.

    Task (c): Alpha sweep results.

    Args:
        results_dict: dict {alpha: metrics}
        title (str): plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (alpha, metrics) in enumerate(sorted(results_dict.items())):
        ax = axes[idx]
        rewards = np.array(metrics['episode_rewards'])

        ax.plot(rewards, alpha=0.4, linewidth=0.5, label='Raw')

        # Rolling average
        window = 5
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        episodes = np.arange(len(rolling_avg)) + window - 1
        ax.plot(episodes, rolling_avg, linewidth=2, color='orange', label='Rolling avg (window=5)')

        ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.7)

        converged_at = metrics.get('converged_at')
        if converged_at:
            ax.axvline(x=converged_at, color='green', linestyle=':', linewidth=2, alpha=0.7,
                       label=f'Converged at ep. {converged_at}')

        ax.set_title(f'α = {alpha}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Cumulative Reward', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig, axes


def plot_algorithm_comparison(q_metrics, sarsa_metrics,
                             title="Q-Learning vs SARSA Comparison"):
    """
    Compare Q-learning and SARSA convergence.

    Task (d): Algorithm comparison.

    Args:
        q_metrics: metrics dict from Q-learning agent
        sarsa_metrics: metrics dict from SARSA agent
        title (str): plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Learning curves
    ax = axes[0]
    q_rewards = np.array(q_metrics['episode_rewards'])
    sarsa_rewards = np.array(sarsa_metrics['episode_rewards'])

    window = 5
    q_rolling = np.convolve(q_rewards, np.ones(window) / window, mode='valid')
    sarsa_rolling = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')
    q_episodes = np.arange(len(q_rolling)) + window - 1
    sarsa_episodes = np.arange(len(sarsa_rolling)) + window - 1

    ax.plot(q_episodes, q_rolling, label='Q-Learning', linewidth=2.5, marker='o', markersize=5)
    ax.plot(sarsa_episodes, sarsa_rolling, label='SARSA', linewidth=2.5, marker='s', markersize=5)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward (Rolling Avg)', fontsize=12)
    ax.set_title('Learning Progress', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Convergence comparison
    ax = axes[1]
    algorithms = ['Q-Learning', 'SARSA']
    q_conv = q_metrics.get('converged_at')
    sarsa_conv = sarsa_metrics.get('converged_at')
    convergence_eps = [
        q_conv if q_conv is not None else 100,
        sarsa_conv if sarsa_conv is not None else 100
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(algorithms, convergence_eps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val, converged in zip(bars, convergence_eps, [q_conv is not None, sarsa_conv is not None]):
        height = bar.get_height()
        label = f'{int(val)}' if converged else 'Not conv.'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label,
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Episodes to Convergence', fontsize=12)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


def plot_q_heatmap(agent, env, action_idx, title="Q-Value Heatmap"):
    """
    Visualize Q-values for a specific action as a heatmap.

    Bonus visualization.

    Args:
        agent: trained agent
        env: GridWorld environment
        action_idx (int): action to visualize (0=N, 1=S, 2=E, 3=W)
        title (str): plot title
    """
    action_names = ['North', 'South', 'East', 'West']

    # Extract Q-values for this action
    q_values = agent.q_table[:, action_idx].reshape(env.grid_size, env.grid_size)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(q_values, cmap='coolwarm', aspect='auto', origin='upper')

    # Add text annotations
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.set_xticklabels(range(1, env.grid_size + 1))
    ax.set_yticklabels(range(1, env.grid_size + 1))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(f'{title} - Action: {action_names[action_idx]}', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Q-Value')
    plt.tight_layout()
    return fig, ax


def get_episode_trajectory(env, agent, q_table=None):
    """
    Run one deterministic (greedy) episode and return the trajectory.

    Args:
        env: GridWorld environment
        agent: agent with choose_action method
        q_table: optional Q-table to temporarily use (for snapshots)

    Returns:
        (path, rewards, final_reward) where path is list of (row, col) tuples
    """
    # Temporarily use snapshot Q-table if provided
    original_q = None
    if q_table is not None:
        original_q = agent.q_table
        agent.q_table = q_table

    state = env.reset()
    path = [env.state_to_pos(state)]
    rewards = []
    total_reward = 0.0
    done = False
    max_steps = 50  # Prevent infinite loops

    for _ in range(max_steps):
        if done:
            break
        # Greedy action (no exploration)
        q_values = agent.get_q_values(state)
        action = np.argmax(q_values)
        next_state, reward, done = env.step(action)
        path.append(env.state_to_pos(next_state))
        rewards.append(reward)
        total_reward += reward
        state = next_state

    # Restore original Q-table
    if original_q is not None:
        agent.q_table = original_q

    return path, rewards, total_reward


def plot_episode_trajectory(env, agent, episode_num=0, q_table=None,
                           title="Episode Trajectory"):
    """
    Visualize one episode: agent's path through the grid.

    Shows the trajectory the agent takes, including rewards collected.

    Args:
        env: GridWorld environment
        agent: trained agent
        episode_num (int): label for the episode
        q_table: optional Q-table snapshot to use
        title (str): plot title
    """
    path, rewards, total_reward = get_episode_trajectory(env, agent, q_table)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid background
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1,
                                     edgecolor='gray', facecolor='white', alpha=0.5)
            ax.add_patch(rect)

    # Draw obstacles
    for obs_row, obs_col in env.obstacles:
        rect = patches.Rectangle((obs_col, obs_row), 1, 1, linewidth=2,
                                 edgecolor='black', facecolor='black', alpha=0.9)
        ax.add_patch(rect)

    # Draw special cells
    start_row, start_col = env.start_pos
    goal_row, goal_col = env.goal_pos
    jump_from_row, jump_from_col = env.jump_from
    jump_to_row, jump_to_col = env.jump_to

    # Start
    ax.add_patch(patches.Circle((start_col + 0.5, start_row + 0.5), 0.15,
                               color='red', zorder=5, label='Start'))
    # Goal
    ax.add_patch(patches.Circle((goal_col + 0.5, goal_row + 0.5), 0.15,
                               color='cyan', zorder=5, label='Goal'))
    # Jump from
    ax.plot(jump_from_col + 0.5, jump_from_row + 0.5, 'y*', markersize=15, zorder=5, label='Jump')
    # Jump to
    ax.plot(jump_to_col + 0.5, jump_to_row + 0.5, 'y*', markersize=12, zorder=5)

    # Draw path
    for step, (row, col) in enumerate(path):
        # Draw step number
        ax.text(col + 0.5, row + 0.3, f"{step}", ha='center', va='center',
               fontsize=8, color='blue', fontweight='bold')

        # Draw circle at position
        if step < len(path) - 1:
            ax.add_patch(patches.Circle((col + 0.5, row + 0.5), 0.08,
                                       color='green', alpha=0.6, zorder=3))

    # Draw arrows connecting path
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        ax.arrow(c1 + 0.5, r1 + 0.5, (c2 - c1) * 0.3, (r2 - r1) * 0.3,
                head_width=0.08, head_length=0.06, fc='blue', ec='blue',
                alpha=0.6, zorder=2)

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(1, env.grid_size + 1))
    ax.set_yticks(range(1, env.grid_size + 1))
    ax.set_xticklabels(range(1, env.grid_size + 1))
    ax.set_yticklabels(range(1, env.grid_size + 1))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)

    # Title with info
    info_str = f"Ep. {episode_num} | Steps: {len(path)-1} | Total Reward: {total_reward:.1f}"
    ax.set_title(f"{title}\n{info_str}", fontsize=13, fontweight='bold')

    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    return fig, ax


def plot_training_progression(env, agent, agent_snapshots, snapshot_episodes,
                             title="Training Progression"):
    """
    Show how agent behavior changes over training.

    Displays trajectories from multiple epochs side-by-side.

    Task (e): Visualization of convergence.

    Args:
        env: GridWorld environment
        agent: agent (used for non-snapshot episodes)
        agent_snapshots: dict {episode: q_table}
        snapshot_episodes: list of episode numbers to display
        title (str): plot title
    """
    n_episodes = len(snapshot_episodes)
    cols = min(3, n_episodes)
    rows = (n_episodes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_episodes > 1 else [axes]

    for idx, episode in enumerate(snapshot_episodes):
        ax = axes[idx]

        # Get Q-table for this episode
        q_table = agent_snapshots.get(episode, agent.q_table)

        # Get trajectory
        path, rewards, total_reward = get_episode_trajectory(env, agent, q_table)

        # Draw grid
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0.5,
                                         edgecolor='lightgray', facecolor='white', alpha=0.3)
                ax.add_patch(rect)

        # Draw obstacles
        for obs_row, obs_col in env.obstacles:
            rect = patches.Rectangle((obs_col, obs_row), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor='black', alpha=0.9)
            ax.add_patch(rect)

        # Draw special cells (smaller for space)
        start_row, start_col = env.start_pos
        goal_row, goal_col = env.goal_pos

        ax.add_patch(patches.Circle((start_col + 0.5, start_row + 0.5), 0.12,
                                   color='red', zorder=5))
        ax.add_patch(patches.Circle((goal_col + 0.5, goal_row + 0.5), 0.12,
                                   color='cyan', zorder=5))

        # Draw path
        for step, (row, col) in enumerate(path):
            ax.add_patch(patches.Circle((col + 0.5, row + 0.5), 0.06,
                                       color='green', alpha=0.7, zorder=3))

        # Draw arrows
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            ax.arrow(c1 + 0.5, r1 + 0.5, (c2 - c1) * 0.25, (r2 - r1) * 0.25,
                    head_width=0.07, head_length=0.05, fc='blue', ec='blue',
                    alpha=0.6, zorder=2)

        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        info = f"Episode {episode}\nReward: {total_reward:.1f}\nSteps: {len(path)-1}"
        ax.set_title(info, fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_episodes, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes
