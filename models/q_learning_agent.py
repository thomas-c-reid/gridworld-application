"""
Q-Learning Agent (off-policy TD learning).

Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
"""

import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent with Q-table."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize Q-learning agent.

        Args:
            n_states (int): number of states
            n_actions (int): number of actions
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): initial exploration rate
            epsilon_min (float): minimum exploration rate
            epsilon_decay (float): epsilon decay per episode
        """
        super().__init__(n_states, n_actions, alpha, gamma,
                         epsilon, epsilon_min, epsilon_decay)
        self.q_table = np.zeros((n_states, n_actions))

    def get_q_values(self, state):
        """Return Q-values for a state."""
        return self.q_table[state]

    def update(self, state, action, reward, next_state, done, **kwargs):
        """
        Q-learning update (off-policy).

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        Args:
            state (int): current state
            action (int): action taken
            reward (float): reward received
            next_state (int): next state
            done (bool): whether episode is finished
        """
        current_q = self.q_table[state, action]

        if done:
            # Terminal state has no value
            target = reward
        else:
            # Bootstrap from next state's max Q-value
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        # TD update
        self.q_table[state, action] += self.alpha * (target - current_q)
