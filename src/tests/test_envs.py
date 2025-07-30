import unittest
import gymnasium as gym
import pygame
from src.environments.custom_env import Car2DEnv  # Assuming a custom environment is defined in environments module


class TestEnvironments(unittest.TestCase):
    
    def setUp(self):
        # Set up test environment
        self.env = gym.make('Car2DEnv-v0')
        self.env.reset()
        
    def test_environment_initialization(self):
        # Test if the environment initializes correctly
        self.assertIsInstance(self.env.unwrapped, Car2DEnv)
        self.assertEqual(self.env.action_space.n, 4)  # Assuming 4 discrete actions
        self.assertEqual(self.env.observation_space.shape, (2,))  # Assuming 2D state space

    def test_step_function(self):
        # Test the step function of the environment
        state, reward, done, _, info = self.env.step(0)  # Taking an action
        self.assertEqual(len(state), 2)  # Check state dimension
        self.assertIsInstance(reward, float)  # Check reward type
        self.assertIsInstance(done, bool)  # Check done type
    
    def test_render_function(self):
        # Test the render function of the environment
        self.env.render()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
    def test_all_functions(self):
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, reward, done, _, info = self.env.step(action)
            self.env.render()    
        
        print("\nTest completed with termination status:", done)
        self.env.close()
        
if __name__ == '__main__':
    unittest.main()