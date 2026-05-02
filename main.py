"""CLI entry point for gridworld RL experiments."""

import argparse
import matplotlib
from src.experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description='Run gridworld RL experiments (Q-Learning & SARSA)')

    parser.add_argument(
        '--config', default='configs/cw2_5x5.yaml',
        help='Path to world config YAML (default: configs/cw2_5x5.yaml)')
    parser.add_argument(
        '--output-dir', default='plots',
        help='Output directory for plots (default: plots)')
    parser.add_argument(
        '--alphas', nargs='+', type=float, default=[1.0, 0.5, 0.1, 0.01],
        help='Learning rates to sweep (default: 1.0 0.5 0.1 0.01)')
    parser.add_argument(
        '--gamma', type=float, default=0.9,
        help='Discount factor (default: 0.9)')
    parser.add_argument(
        '--epsilon-start', type=float, default=1.0,
        help='Initial exploration rate (default: 1.0)')
    parser.add_argument(
        '--epsilon-min', type=float, default=0.01,
        help='Minimum exploration rate (default: 0.01)')
    parser.add_argument(
        '--epsilon-decay', type=float, default=0.95,
        help='Epsilon decay per episode (default: 0.95)')
    parser.add_argument(
        '--episodes', type=int, default=100,
        help='Max training episodes (default: 100)')
    parser.add_argument(
        '--conv-window', type=int, default=30,
        help='Convergence window size (default: 30)')
    parser.add_argument(
        '--conv-threshold', type=float, default=10.0,
        help='Convergence reward threshold (default: 10.0)')
    parser.add_argument(
        '--show', action='store_true',
        help='Show plots interactively (for PyCharm screenshots)')

    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    runner = ExperimentRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        alpha_values=args.alphas,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        n_episodes=args.episodes,
        conv_window=args.conv_window,
        conv_threshold=args.conv_threshold,
    )
    runner.run()


if __name__ == '__main__':
    main()
