import gymnasium as gym
import numpy as np

# This module initializes custom environments for reinforcement learning algorithms.
class CustomEnv1(gym.Env):
    def __init__(self):
        super(CustomEnv1, self).__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)  # Example: two actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        return np.array([0.5], dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        state = np.array([0.5], dtype=np.float32)  # Example state
        reward = self.compute_reward(state, action)  # Example reward
        terminated = 0  # Example termination condition
        truncated = 0
        info = {}
        return state, reward, terminated, truncated, info

    def compute_reward(self, state, action):
        return 0
    
    
class CustomEnv2(gym.Env):
    def __init__(self):
        super(CustomEnv2, self).__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Example: three actions
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        return np.array([5.0], dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        state = np.array([5.0], dtype=np.float32)  # Example state
        reward = 2.0  # Example reward
        done = False  # Example termination condition
        return state, reward, done, {}

__all__ = ['CustomEnv1', 'CustomEnv2']