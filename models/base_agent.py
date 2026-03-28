"""
Abstract base class for RL agents.

Subclasses implement:
  - get_q_values(state): return Q-values for a state
  - update(s, a, r, s_next, ...): update the value function
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base agent with common RL logic."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize agent.

        Args:
            n_states (int): number of states
            n_actions (int): number of actions
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): initial exploration rate
            epsilon_min (float): minimum exploration rate
            epsilon_decay (float): epsilon decay per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.on_policy = False  # Override in on-policy agents (e.g., SARSA)

    @abstractmethod
    def get_q_values(self, state):
        """
        Get Q-values for a state.

        Returns:
            array of shape (n_actions,): Q-values for each action
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done, **kwargs):
        """
        Update the value function based on a transition.

        Args:
            state (int): current state
            action (int): action taken
            reward (float): reward received
            next_state (int): next state
            done (bool): whether episode is finished
            **kwargs: algorithm-specific args (e.g., next_action for SARSA)
        """
        pass

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state (int): current state

        Returns:
            action (int): selected action
        """
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def get_state_values(self):
        """
        Get maximum Q-value for each state (V(s) = max Q(s, a)).

        Returns:
            array of shape (n_states,): state values
        """
        state_values = np.zeros(self.n_states)
        for state in range(self.n_states):
            q_values = self.get_q_values(state)
            state_values[state] = np.max(q_values)
        return state_values

    def get_policy(self):
        """
        Get greedy policy: action with highest Q-value per state.

        Returns:
            array of shape (n_states,): policy action indices
        """
        policy = np.zeros(self.n_states, dtype=int)
        for state in range(self.n_states):
            q_values = self.get_q_values(state)
            policy[state] = np.argmax(q_values)
        return policy

    def decay_epsilon(self):
        """Apply epsilon decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
