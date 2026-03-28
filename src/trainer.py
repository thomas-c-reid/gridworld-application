"""
Training loop for RL agents.

Handles both Q-learning and SARSA via the agent's on_policy flag.
"""

import numpy as np


class Trainer:
    """Trainer for RL agents."""

    MAX_STEPS_PER_EPISODE = 200

    def __init__(self, env, agent, n_episodes=100, conv_window=30, conv_threshold=10.0,
                 snapshot_intervals=None):
        """
        Initialize trainer.

        Args:
            env: GridWorld environment
            agent: BaseAgent subclass (Q-learning or SARSA)
            n_episodes (int): max episodes to train
            conv_window (int): window size for convergence check
            conv_threshold (float): avg reward threshold for early stopping
            snapshot_intervals (list): episodes at which to snapshot agent (e.g., [0, 10, 30, 50, 90])
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.conv_window = conv_window
        self.conv_threshold = conv_threshold
        self.snapshot_intervals = snapshot_intervals or []

        # Metrics
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.converged_at = None
        self.agent_snapshots = {}  # {episode: copy of q_table}

    def train(self):
        """
        Run training loop.

        Returns:
            dict: metrics including episode_rewards, episode_steps, converged_at
        """
        for episode in range(self.n_episodes):
            if episode % 10 == 0:
                print(f"  Episode {episode}/{self.n_episodes}...", end='\r')

            state = self.env.reset()
            action = self.agent.choose_action(state)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done and episode_steps < self.MAX_STEPS_PER_EPISODE:
                # Take step
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Update Q-values
                if self.agent.on_policy:
                    # SARSA: next_action is chosen now and used on the next step
                    next_action = self.agent.choose_action(next_state)
                    self.agent.update(state, action, reward, next_state, done,
                                     next_action=next_action)
                    action = next_action
                else:
                    # Q-learning (off-policy): action choice is independent of update
                    self.agent.update(state, action, reward, next_state, done)
                    action = self.agent.choose_action(next_state)

                state = next_state

            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)
            self.epsilon_history.append(self.agent.epsilon)

            # Decay exploration
            self.agent.decay_epsilon()

            # Save snapshot if at interval
            if episode in self.snapshot_intervals:
                self._save_snapshot(episode)

            # Check convergence
            if self._check_convergence():
                self.converged_at = episode
                print(f"Converged at episode {episode}")
                break

        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'epsilon_history': self.epsilon_history,
            'converged_at': self.converged_at,
            'agent_snapshots': self.agent_snapshots,
        }

    def _save_snapshot(self, episode):
        """Save a copy of the agent's Q-table at this episode."""
        self.agent_snapshots[episode] = self.agent.q_table.copy()

    def _check_convergence(self):
        """Check if agent has converged."""
        if len(self.episode_rewards) < self.conv_window:
            return False

        recent_rewards = self.episode_rewards[-self.conv_window:]
        avg_reward = np.mean(recent_rewards)
        return avg_reward > self.conv_threshold
