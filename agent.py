"""Tabular Q-learning agent for FrozenLake-v1."""
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.8, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.0005, seed=42):
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, terminated):
        target = reward if terminated else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * (1.0 - self.epsilon_decay))
