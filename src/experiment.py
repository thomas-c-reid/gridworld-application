"""Experiment runner for gridworld RL experiments."""

import os
import matplotlib.pyplot as plt

from src.world_config import WorldConfig
from src.gridworld import GridWorld
from src.trainer import Trainer
from src.visualizer import (
    plot_gridworld,
    plot_learning_curve,
    plot_alpha_comparison,
    plot_algorithm_comparison,
    plot_q_heatmap,
    plot_training_progression,
)
from models.q_learning_agent import QLearningAgent
from models.sarsa_agent import SARSAAgent


AGENT_REGISTRY = {
    'q-learning': QLearningAgent,
    'sarsa': SARSAAgent,
}


class ExperimentRunner:
    """Orchestrates gridworld RL experiments."""

    def __init__(self, config_path, output_dir='plots',
                 alpha_values=None, gamma=0.9,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.95,
                 n_episodes=100, conv_window=30, conv_threshold=10.0,
                 snapshot_intervals=None):
        self.config = WorldConfig.from_yaml(config_path)
        self.output_dir = output_dir
        self.alpha_values = alpha_values or [1.0, 0.5, 0.1, 0.01]
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.conv_window = conv_window
        self.conv_threshold = conv_threshold
        self.snapshot_intervals = snapshot_intervals or [0, 5, 10, 20, 40, 80]

    # ── helpers ─────────────────────────────────────────────────

    def _make_env(self):
        return GridWorld(self.config)

    def _make_agent(self, agent_type, alpha):
        cls = AGENT_REGISTRY[agent_type]
        return cls(
            n_states=self.config.n_states,
            n_actions=self.config.n_actions,
            alpha=alpha,
            gamma=self.gamma,
            epsilon=self.epsilon_start,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
        )

    def _make_trainer(self, env, agent):
        return Trainer(
            env, agent,
            n_episodes=self.n_episodes,
            conv_window=self.conv_window,
            conv_threshold=self.conv_threshold,
            snapshot_intervals=self.snapshot_intervals,
        )

    # ── experiment stages ───────────────────────────────────────

    def alpha_sweep(self, agent_type='q-learning'):
        """Train an agent with each learning rate and return results."""
        print("=" * 60)
        print("ALPHA SWEEP - LEARNING RATE COMPARISON")
        print("=" * 60)

        results = {}
        for alpha in self.alpha_values:
            print(f"\nTraining {agent_type} with alpha = {alpha}")
            env = self._make_env()
            agent = self._make_agent(agent_type, alpha)
            trainer = self._make_trainer(env, agent)
            metrics = trainer.train()
            results[alpha] = metrics

            converged = metrics['converged_at']
            if converged is not None:
                print(f"  > Converged at episode {converged}")
            else:
                print(f"  > Did not converge within {self.n_episodes} episodes")

        return results

    def compare_algorithms(self, alpha_results):
        """Train Q-learning and SARSA with the best alpha from a sweep.

        Returns:
            (results_dict, best_alpha) where results_dict maps
            agent type to {'agent': ..., 'metrics': ...}.
        """
        print("\n" + "=" * 60)
        print("Q-LEARNING vs SARSA COMPARISON")
        print("=" * 60)

        # Best alpha: fastest convergence, tiebreak by highest total reward
        best_alpha = min(
            alpha_results.items(),
            key=lambda x: (
                float('inf') if x[1]['converged_at'] is None else x[1]['converged_at'],
                -sum(x[1]['episode_rewards']),
            ),
        )[0]
        print(f"\nBest alpha: {best_alpha}")

        results = {}
        for agent_type in ['q-learning', 'sarsa']:
            print(f"\nTraining {agent_type} with alpha = {best_alpha}")
            env = self._make_env()
            agent = self._make_agent(agent_type, best_alpha)
            trainer = self._make_trainer(env, agent)
            metrics = trainer.train()
            results[agent_type] = {'agent': agent, 'metrics': metrics}

            if metrics['converged_at']:
                print(f"  > {agent_type} converged at episode {metrics['converged_at']}")

        return results, best_alpha

    def visualize(self, alpha_results, comparison_results, best_alpha):
        """Generate all plots and save to output_dir."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)
        env = self._make_env()

        q_data = comparison_results['q-learning']
        sarsa_data = comparison_results['sarsa']
        q_agent, q_metrics = q_data['agent'], q_data['metrics']
        sarsa_agent, sarsa_metrics = sarsa_data['agent'], sarsa_data['metrics']

        # Alpha comparison
        fig, _ = plot_alpha_comparison(
            alpha_results, title="Learning Rate Comparison")
        plt.savefig(f'{self.output_dir}/task_c_alpha_comparison.png',
                    dpi=150, bbox_inches='tight')
        print(f"\n[*] Saved: {self.output_dir}/task_c_alpha_comparison.png")
        plt.close()

        # Learning curves
        fig, _ = plot_learning_curve(
            [q_metrics, sarsa_metrics],
            [f"Q-Learning (alpha={best_alpha})",
             f"SARSA (alpha={best_alpha})"],
            title="Training Progress")
        plt.savefig(f'{self.output_dir}/task_e_learning_curves.png',
                    dpi=150, bbox_inches='tight')
        print(f"[*] Saved: {self.output_dir}/task_e_learning_curves.png")
        plt.close()

        # Algorithm comparison
        fig, _ = plot_algorithm_comparison(
            q_metrics, sarsa_metrics, title="Algorithm Comparison")
        plt.savefig(f'{self.output_dir}/task_d_algorithm_comparison.png',
                    dpi=150, bbox_inches='tight')
        print(f"[*] Saved: {self.output_dir}/task_d_algorithm_comparison.png")
        plt.close()

        # Training progression snapshots
        for label, data in [('q', q_data), ('sarsa', sarsa_data)]:
            snapshots = data['metrics'].get('agent_snapshots', {})
            if snapshots:
                snapshot_eps = sorted(snapshots.keys())
                fig, _ = plot_training_progression(
                    env, data['agent'], snapshots, snapshot_eps,
                    title=f"{label.upper()} Training Progression")
                plt.savefig(
                    f'{self.output_dir}/task_e_{label}_progression.png',
                    dpi=150, bbox_inches='tight')
                print(f"[*] Saved: {self.output_dir}/task_e_{label}_progression.png")
                plt.close()

        # Final policy grids (kept open for --show / PyCharm screenshots)
        interactive = plt.get_backend().lower() != 'agg'
        for label, agent in [('qlearning', q_agent), ('sarsa', sarsa_agent)]:
            fig, _ = plot_gridworld(
                env, agent,
                title=f"{label.replace('qlearning', 'Q-Learning').replace('sarsa', 'SARSA')}"
                      f" - State Values & Policy")
            plt.savefig(f'{self.output_dir}/task_f_gridworld_{label}.png',
                        dpi=150, bbox_inches='tight')
            print(f"[*] Saved: {self.output_dir}/task_f_gridworld_{label}.png")
            if not interactive:
                plt.close()

        # Q-value heatmaps
        print("\nBonus visualizations (Q-value heatmaps):")
        for action in range(4):
            fig, _ = plot_q_heatmap(
                q_agent, env, action, title="Q-Learning Q-Value Heatmap")
            plt.savefig(
                f'{self.output_dir}/bonus_qheatmap_action_{action}.png',
                dpi=150, bbox_inches='tight')
            print(f"[*] Saved: {self.output_dir}/bonus_qheatmap_action_{action}.png")
            plt.close()

    # ── main entry point ────────────────────────────────────────

    def run(self):
        """Run the full experiment pipeline: sweep → compare → visualize."""
        print("\n" + "=" * 60)
        print("GRIDWORLD RL EXPERIMENT")
        print("=" * 60)

        alpha_results = self.alpha_sweep()
        comparison_results, best_alpha = self.compare_algorithms(alpha_results)
        self.visualize(alpha_results, comparison_results, best_alpha)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nPlots saved to: {self.output_dir}/")

        # If using an interactive backend, show all open figures
        if plt.get_backend().lower() != 'agg':
            plt.show()
