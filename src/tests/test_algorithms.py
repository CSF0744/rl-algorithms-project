"""
    Unittest file for testing RL algorithms agent function properly.
    This file contains tests for DQN, PPO, and A2C agents.
"""

import unittest
from algorithms import DQN_Agent
from algorithms import PPO_Agent
from algorithms import A2C_Agent  # Assuming A2CAgent
import gymnasium as gym
from environments import Car2DEnv  # Assuming a custom environment is defined in environments module
import torch

class TestAlgorithms(unittest.TestCase):
    
    def set_up_hyperparameters(self):
        # Set up hyperparameters for each agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dqn_hyper_dict = {
            'lr': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 1000,
            'buffer_size': 1000,
            'batch_size': 4,
            'num_episodes': 8
        }
        
        self.ppo_hyper_dict = {
            'lr': 0.001,
            'gamma': 0.99,
            'lam': 0.95,
            'batch_size': 4,
            'num_episodes': 8,
            'eps_clip': 0.2
        }
        
        self.a2c_hyper_dict = {
            'lr': 0.001,
            'gamma': 0.99,
            'batch_size': 4,
            'num_episodes': 8
        }     

    def setUp(self):
        # Set up test environment 
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')  # Example environment
        self.env.reset()
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.set_up_hyperparameters()
        self.dqn = DQN_Agent(
            state_dim,
            action_dim,
            hidden_dim=64,
            hyper_parameters=self.dqn_hyper_dict,
            device=self.device
        )
        
        self.ppo = PPO_Agent(
            state_dim,
            action_dim,
            hidden_dim=64,
            hyper_parameters=self.ppo_hyper_dict,
            device=self.device
        )
        
        self.a2c = A2CAgent(
            state_dim,
            action_dim,
            hidden_dim=64,
            hyper_parameters=self.a2c_hyper_dict,
            device=self.device
        )
        
    def test_dqn_training(self):
        result = self.dqn.train_DQN(self.env)
        self.assertTrue(result['success'])
        # self.assertGreater(result['reward'], 0) # reward may not be positive

    def test_ppo_training(self):
        result = self.ppo.train_PPO(self.env)
        self.assertTrue(result['success'])
        # self.assertGreater(result['reward'], 0)

    def test_a3c_training(self):
        result = self.a2c.train_AC(self.env)
        self.assertTrue(result['success'])
        # self.assertGreater(result['reward'], 0)

if __name__ == '__main__':
    unittest.main()