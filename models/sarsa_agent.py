"""
SARSA Agent (on-policy TD learning).

Update rule: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

Requires knowledge of the next action (a') taken.
"""

import numpy as np
from .base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    """SARSA agent with Q-table."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize SARSA agent.

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
        self.on_policy = True
        self.q_table = np.zeros((n_states, n_actions))

    def get_q_values(self, state):
        """Return Q-values for a state."""
        return self.q_table[state]

    def update(self, state, action, reward, next_state, done, next_action=None, **kwargs):
        """
        SARSA update (on-policy).

        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Args:
            state (int): current state
            action (int): action taken
            reward (float): reward received
            next_state (int): next state
            done (bool): whether episode is finished
            next_action (int): action to be taken in next state (required for SARSA)
        """
        if next_action is None:
            raise ValueError("SARSA requires next_action parameter")

        current_q = self.q_table[state, action]

        if done:
            # Terminal state has no value
            target = reward
        else:
            # Bootstrap from next state's Q-value for the next action (on-policy)
            next_q = self.q_table[next_state, next_action]
            target = reward + self.gamma * next_q

        # TD update
        self.q_table[state, action] += self.alpha * (target - current_q)
