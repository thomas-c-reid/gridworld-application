"""
Main entry point for grid world RL training.

Performs:
1. Alpha sweep: train Q-learning with multiple learning rates
2. Best alpha: train both Q-learning and SARSA
3. Visualizations for all tasks (a)-(f)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.gridworld import GridWorld
from models.q_learning_agent import QLearningAgent
from models.sarsa_agent import SARSAAgent
from src.trainer import Trainer
from src.visualizer import (
    plot_gridworld,
    plot_learning_curve,
    plot_alpha_comparison,
    plot_algorithm_comparison,
    plot_q_heatmap,
    plot_training_progression
)


# Hyperparameters
ALPHA_VALUES = [1.0, 0.5, 0.1, 0.01]
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
N_EPISODES = 100
CONV_WINDOW = 30
CONV_THRESHOLD = 10.0


def alpha_sweep():
    """
    Task (c): Test Q-learning with different learning rates.

    Returns:
        results_dict: {alpha: metrics}
    """
    print("=" * 60)
    print("TASK (C): ALPHA SWEEP - LEARNING RATE COMPARISON")
    print("=" * 60)

    env = GridWorld()
    results = {}

    for alpha in ALPHA_VALUES:
        print(f"\nTraining Q-Learning with alpha = {alpha}")
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=alpha,
            gamma=GAMMA,
            epsilon=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY
        )

        trainer = Trainer(env, agent, n_episodes=N_EPISODES,
                         conv_window=CONV_WINDOW, conv_threshold=CONV_THRESHOLD,
                         snapshot_intervals=[0, 5, 10, 20, 40, 80])
        metrics = trainer.train()
        results[alpha] = metrics

        converged = metrics['converged_at']
        if converged is not None:
            print(f"  > Converged at episode {converged}")
        else:
            print(f"  > Did not converge within {N_EPISODES} episodes")

    return results


def best_alpha_comparison(results_dict):
    """
    Task (d)/(e): Train both Q-learning and SARSA with best alpha.

    Args:
        results_dict: from alpha_sweep()

    Returns:
        (q_agent, sarsa_agent, q_metrics, sarsa_metrics)
    """
    print("\n" + "=" * 60)
    print("TASK (D)/(E): Q-LEARNING vs SARSA WITH BEST ALPHA")
    print("=" * 60)

    # Find best alpha (fastest convergence, tie-break by highest total reward)
    best_alpha = min(results_dict.items(),
                     key=lambda x: (
                         float('inf') if x[1]['converged_at'] is None else x[1]['converged_at'],
                         -sum(x[1]['episode_rewards'])
                     ))[0]
    print(f"\nBest alpha: {best_alpha}")

    env = GridWorld()

    # Q-Learning
    print(f"\nTraining Q-Learning with alpha = {best_alpha}")
    q_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=best_alpha,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )
    q_trainer = Trainer(env, q_agent, n_episodes=N_EPISODES,
                       conv_window=CONV_WINDOW, conv_threshold=CONV_THRESHOLD,
                       snapshot_intervals=[0, 5, 10, 20, 40, 80])
    q_metrics = q_trainer.train()
    if q_metrics['converged_at']:
        print(f"  > Q-Learning converged at episode {q_metrics['converged_at']}")

    # SARSA
    print(f"\nTraining SARSA with alpha = {best_alpha}")
    sarsa_agent = SARSAAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=best_alpha,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )
    sarsa_trainer = Trainer(env, sarsa_agent, n_episodes=N_EPISODES,
                           conv_window=CONV_WINDOW, conv_threshold=CONV_THRESHOLD,
                           snapshot_intervals=[0, 5, 10, 20, 40, 80])
    sarsa_metrics = sarsa_trainer.train()
    if sarsa_metrics['converged_at']:
        print(f"  > SARSA converged at episode {sarsa_metrics['converged_at']}")

    return q_agent, sarsa_agent, q_metrics, sarsa_metrics, best_alpha


def visualize_results(alpha_results, q_agent, sarsa_agent, q_metrics, sarsa_metrics, env, best_alpha):
    """
    Create all visualizations.

    Task (f): Grid visualization with state values.

    Args:
        alpha_results: from alpha_sweep()
        q_agent, sarsa_agent: trained agents
        q_metrics, sarsa_metrics: training results
        env: GridWorld environment
    """
    print("\n" + "=" * 60)
    print("TASK (F): VISUALIZATIONS")
    print("=" * 60)

    # Task (c): Alpha comparison
    fig, axes = plot_alpha_comparison(alpha_results,
                                     title="Task (C): Learning Rate Comparison")
    plt.savefig('plots/task_c_alpha_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[*] Saved: plots/task_c_alpha_comparison.png")
    plt.close()

    # Task (c)/(e): Learning curves
    q_label = f"Q-Learning (alpha={best_alpha})"
    sarsa_label = f"SARSA (alpha={best_alpha})"
    fig, ax = plot_learning_curve([q_metrics, sarsa_metrics],
                                 [q_label, sarsa_label],
                                 title="Task (E): Training Progress")
    plt.savefig('plots/task_e_learning_curves.png', dpi=150, bbox_inches='tight')
    print("[*] Saved: plots/task_e_learning_curves.png")
    plt.close()

    # Task (d): Algorithm comparison
    fig, axes = plot_algorithm_comparison(q_metrics, sarsa_metrics,
                                         title="Task (D): Algorithm Comparison")
    plt.savefig('plots/task_d_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("[*] Saved: plots/task_d_algorithm_comparison.png")
    plt.close()

    # Task (e) bonus: Training progression - Q-Learning
    if q_metrics.get('agent_snapshots'):
        snapshot_eps = sorted(q_metrics['agent_snapshots'].keys())
        fig, axes = plot_training_progression(env, q_agent, q_metrics['agent_snapshots'],
                                             snapshot_eps,
                                             title="Task (E): Q-Learning Training Progression")
        plt.savefig('plots/task_e_q_progression.png', dpi=150, bbox_inches='tight')
        print("[*] Saved: plots/task_e_q_progression.png")
        plt.close()

    # Task (e) bonus: Training progression - SARSA
    if sarsa_metrics.get('agent_snapshots'):
        snapshot_eps = sorted(sarsa_metrics['agent_snapshots'].keys())
        fig, axes = plot_training_progression(env, sarsa_agent, sarsa_metrics['agent_snapshots'],
                                             snapshot_eps,
                                             title="Task (E): SARSA Training Progression")
        plt.savefig('plots/task_e_sarsa_progression.png', dpi=150, bbox_inches='tight')
        print("[*] Saved: plots/task_e_sarsa_progression.png")
        plt.close()

    # Task (f): Grid world with state values - Q-Learning
    fig, ax = plot_gridworld(env, q_agent,
                            title="Task (F): Q-Learning - State Values & Policy")
    plt.savefig('plots/task_f_gridworld_qlearning.png', dpi=150, bbox_inches='tight')
    print("[*] Saved: plots/task_f_gridworld_qlearning.png")
    plt.close()

    # Task (f): Grid world with state values - SARSA
    fig, ax = plot_gridworld(env, sarsa_agent,
                            title="Task (F): SARSA - State Values & Policy")
    plt.savefig('plots/task_f_gridworld_sarsa.png', dpi=150, bbox_inches='tight')
    print("[*] Saved: plots/task_f_gridworld_sarsa.png")
    plt.close()

    # Bonus: Q-value heatmaps
    print("\nBonus visualizations (Q-value heatmaps):")
    for action in range(4):
        fig, ax = plot_q_heatmap(q_agent, env, action,
                                title="Q-Learning Q-Value Heatmap")
        plt.savefig(f'plots/bonus_qheatmap_action_{action}.png', dpi=150, bbox_inches='tight')
        print(f"[*] Saved: plots/bonus_qheatmap_action_{action}.png")
        plt.close()


def main():
    """Main entry point."""
    import os

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    print("\n" + "=" * 60)
    print("COM762 CW2: Q-LEARNING & SARSA GRID WORLD")
    print("=" * 60)

    # Task (a)/(b): Described in comments and report
    print("\nTASK (A)/(B): See report for exploration/exploitation and RL formulation")

    # Task (c): Alpha sweep
    alpha_results = alpha_sweep()

    # Task (d)/(e): Best alpha comparison
    q_agent, sarsa_agent, q_metrics, sarsa_metrics, best_alpha = best_alpha_comparison(alpha_results)

    # Task (f): Visualizations
    env = GridWorld()
    visualize_results(alpha_results, q_agent, sarsa_agent, q_metrics, sarsa_metrics, env, best_alpha)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - plots/task_c_alpha_comparison.png")
    print("  - plots/task_e_learning_curves.png")
    print("  - plots/task_e_q_progression.png (training progression)")
    print("  - plots/task_e_sarsa_progression.png (training progression)")
    print("  - plots/task_d_algorithm_comparison.png")
    print("  - plots/task_f_gridworld_qlearning.png")
    print("  - plots/task_f_gridworld_sarsa.png")
    print("  - plots/bonus_qheatmap_action_*.png")
    print("\nNote: Use plots from task_f_gridworld_*.png for report screenshot")
    print("      Use progression plots to show convergence visually")


if __name__ == '__main__':
    main()
