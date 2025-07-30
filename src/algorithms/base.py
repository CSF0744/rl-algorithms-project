from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import gymnasium as gym
import numpy as np

class BaseAgent(ABC):
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        config: Dict[str, Any]
    ):
        """
        Initialize the agent with configuration parameters.

        Args:
            config (dict): Configuration dict containing
                - input_dim: int, input dimension of model
                - hidden_dim: int, hidden layer dimension
                - output_dim: int, output dimension (action space size)
                - lr: float, learning rate
                - other hyperparameters as needed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.config = config

    @abstractmethod
    def action_selection(self, state: np.ndarray, **kwargs) -> Tuple[Any, Optional[Dict]]:
        """
        Select action given the state.

        Args:
            state (np.ndarray): Current environment state
            **kwargs: Optional keyword args for derived classes

        Returns:
            action: action to take (int or np.ndarray)
            info: optional additional info dict (can be None)
        """
        pass
    
    @abstractmethod
    def collect_performance(self, episode: int, **kwargs):
        """
        Collect performance information during training phase.
        Args:
            episode (int): current episode of performance
            **kwargs: Optional keywor args to store
        """
        pass

    @abstractmethod
    def update(self, epoches: int, batch_size: int):
        """
        Update the model parameters.

        Args:
            epochs (int): Number of epochs to update
            batch_size (int): Mini-batch size per update
        """
        pass

    @abstractmethod
    def train(self, env: gym.Env):
        """
        Train the agent in the given environment.

        Args:
            env (gym.Env): Gym environment to train on
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save model parameters to file.

        Args:
            filepath (str): Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load model parameters from file.

        Args:
            filepath (str): Path to load the model from
        """
        pass

    def evaluate(self, env: gym.Env, num_episodes: int = 10, render: bool = False) -> float:
        """
        Evaluate the agent's performance in the environment.

        Args:
            env (gym.Env): Gym environment for evaluation
            num_episodes (int): Number of episodes to average over
            render (bool): Whether to render during evaluation

        Returns:
            float: Average cumulative reward over episodes
        """
        total_reward = 0.0
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = self.action_selection(state, eval=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if render:
                    env.render()
            total_reward += episode_reward

        avg_reward = total_reward / num_episodes
        print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    