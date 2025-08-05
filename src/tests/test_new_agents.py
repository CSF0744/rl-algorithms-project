import unittest
import gymnasium as gym
import torch

from src.algorithms.DQN import DQNAgent

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.env.reset()
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.dqn_config = {'num_episodes':5}
        self.dqn_agent = DQNAgent(state_dim, action_dim, 64, self.dqn_config)
    
    def test_dqn_initialization(self):
        self.assertIsInstance(self.dqn_agent, DQNAgent)
    
    def test_dqn_action_selection(self):
        pass
    
    def test_dqn_train(self):
        print('\nTesting New DQN Train!')
        self.dqn_agent.train(self.env)
        print('\nNew DQN Train Pass!')
    
    def test_dqn_evaluate(self):
        print('\nEvaluating Agent!')
        env = gym.make('CartPole-v1',render_mode='human')
        self.dqn_agent.evaluate(env, num_episodes=5, render=True)
        print('\nEvaluating Complete!')
    
if __name__ == '__main__':
    unittest.main()