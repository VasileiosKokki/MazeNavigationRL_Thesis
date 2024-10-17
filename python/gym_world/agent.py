import sys
from collections import defaultdict
import gymnasium as gym
import numpy as np


def debug_print(*args):
    """Prints debug information to stderr."""
    print(" | ".join(map(str, args)), file=sys.stderr)

def convert_obs_to_tuple(obs):
    return (
        int(obs["agent"][0]),  # agent_x
        int(obs["agent"][1]),  # agent_y
        int(obs["target"][0]),  # target_x
        int(obs["target"][1])   # target_y
    )

class GridWorldAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        #self.q_values = defaultdict(lambda: np.zeros((env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.action_space.n)))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs, is_training) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if is_training and np.random.default_rng().random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs, :]))

    def update(
            self,
            obs,
            action: int,
            reward: float,
            terminated: bool,
            next_obs,
    ):
        """Updates the Q-value of an action."""

        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * np.max(self.q_values[next_obs, :]) - self.q_values[obs, action]
        )

        # self.q_values[obs][action] = (
        #         self.q_values[obs][action] + self.lr * temporal_difference
        # )
        self.training_error.append(temporal_difference)

        self.q_values[obs, action] = self.q_values[obs, action] + self.lr * temporal_difference

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        if self.epsilon == 0:
            self.lr = 0.0001
