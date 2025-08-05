"""
    Unittest file for testing RL algorithms agent function properly.
    This file contains tests for DDPG agents in continuous environment.
"""

import unittest
import gymnasium as gym
import torch

from src.algorithms.DQN import DQN_Agent
from src.algorithms.PPO import PPO_Agent
from src.algorithms.A3C import A2C_Agent
from src.algorithms.DDPG import DDPG_Agent

class TestAlgorithms_Discrete(unittest.TestCase):
    
    def set_up_hyperparameters(self):
        # Set up hyperparameters for each agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.ddpg_hyper_dict = {
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'gamma': 0.99,
            'tau': 0.99,
            'batch_size': 4,
            'num_episodes': 4,
            'buffer_size':1000
        }    

    def setUp(self):
        # Set up test environment 
        self.env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)  # Example environment
        self.env.reset()
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.set_up_hyperparameters()
        
        self.ddpg = DDPG_Agent(
            state_dim,
            action_dim,
            max_action=1,
            hidden_dim=64,
            hyper_parameters=self.ddpg_hyper_dict,
            device=self.device
        )
    
    def test_ddpg_training(self):
        print('\n Testing DDPG!')
        result = self.ddpg.train(self.env, noise_std=0.01)
        self.assertTrue(result['success'])
        print("\n DDPG Test Pass!")

    
if __name__ == '__main__':
    unittest.main()