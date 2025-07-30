import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

def fast_norm(v):
    return np.sqrt(np.sum(np.square(v)))

# This module initializes custom environments for reinforcement learning algorithms.
class CustomEnvTemplate(gym.Env):
    def __init__(self):
        super(CustomEnvTemplate, self).__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)  # Example: two actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.state = None  # Initialize state

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

    def close(self):
        pass
    
class Car2DEnv(gym.Env):
    """A simple 2D car environment for reinforcement learning."""
    metadata = {
        'render_modes': ['human'],
        'render_fps': 30,
        'name': 'Car2DEnv',
        'description': 'A simple 2D car environment for reinforcement learning.',
        'author': 'Your Name',
        'version': '0.1.0'
    }
    def __init__(self,render_mode=None, max_steps=500):
        """Possible Improvement: 
        1. continuous dynamic
        2. parameterize env
        3. add info dict for step
        4. show velocity direction
        
        """
        super(Car2DEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Four directionary steps
        # self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32) # steering angle and acceleration
        
        # start with simple 2D position observation
        self.observation_space = spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32) # position
        # self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi, 0]), high=np.array([100, 100, np.pi, 10]), dtype=np.float32) # x, y, theta, velocity
        self.max_steps = max_steps
        self.goal_threshold = 2.0
        self.current_step = 0
        
        self.state = None  # Initialize state
        self.goal = np.array([90.0, 90.0])
        # List of circular obstacles with [x,y,radius]
        self.obstacles = [np.array([40.0, 50.0, 5.0]), np.array([60.0, 70.0, 4.0])]  # Example circular obstacles
        
        self.dt = 5 # time step/ step size
        self.trajectory = [] # Store trajectory for rendering
        
        # Redering settings for pygame
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_size = 250
    
    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # 1. add random initial point and random seed properly in future
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.trajectory = [self.state]
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        x, y= self.state
        
        # Update state based on action and kinematics
        directions = {
            0: np.array([0, 1]),   # up
            1: np.array([0, -1]),  # down
            2: np.array([1, 0]),   # right
            3: np.array([-1, 0]),  # left
        }
        self.state += directions[action] * self.dt
            
        self.state = np.array([x, y])
        # add noise for robustness
        # self.state += np.random.normal(0, 0.1, size=self.state.shape)
        # clip state to avoid out off bound
        self.state = np.clip(self.state, -100, 100)
        self.current_step += 1
        # Collision check
        collision = self.collision_check()
            
        # Check for goal reach
        goal_distance = fast_norm(self.state - self.goal)
        done = goal_distance < self.goal_threshold or collision or self.current_step >= self.max_steps
        
        # Reward calculation
        reward = self.reward_function(collision)
        self.trajectory.append(self.state)
        
        return self.state, reward, done, False, {} #Return state, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human':
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()
        
        self.screen.fill((255, 255, 255))  # White background
        pygame.draw.circle(self.screen, (0, 255, 0), self._world_to_screen(self.goal), 10) # green goal
        
        for obs in self.obstacles: # red obstacles
            pygame.draw.circle(self.screen, (255, 0, 0), self._world_to_screen(obs[:2]), int(obs[2]))
            
        if len(self.trajectory) > 1:
            points = [self._world_to_screen(pos) for pos in self.trajectory]
            pygame.draw.lines(self.screen, (0, 0, 0), False, points, 2)  # black trajectory line
            
        car_pos = self.state[:2]
        pygame.draw.circle(self.screen, (0, 0, 255), self._world_to_screen(car_pos), 5)  # blue position
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])  # Limit FPS
    
    def _world_to_screen(self, pos):
        """Convert world coordinates to screen pixel coordinates."""
        scale = self.screen_size / 200.0  # Assuming the world is 200x200 units, screen is 250x250 pixels
        # center the world to middle of screen
        return int(pos[0] * scale + self.screen_size / 2), int(pos[1] * scale + self.screen_size / 2)
    
    def collision_check(self):
        # collision check
        collision = False
        for obs in self.obstacles:
            # collison check with circular obs
            if fast_norm(self.state - obs[:2]) < obs[2]:
                collision = True
                break
            
        return collision
    
    def reward_function(self, collision):
        goal_distance = fast_norm(self.state - self.goal)
        reward = -goal_distance * 0.1
        if collision: # collision to obstacle
            reward = - 100
        elif goal_distance < self.goal_threshold: # reach goal
            reward = 100
        else: # add reward for approaching to goal
            previous_distance = np.linalg.norm(self.trajectory[-1] - self.goal) if len(self.trajectory)>1 else goal_distance
            reward += (previous_distance-goal_distance) * 10
            
        return reward
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
    