# basic package
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from tqdm import tqdm

# NN package
import torch
import torch.nn.functional as F
from torch import nn

class DQN(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64
    ):
        super().__init__()
        # Define the neural network architecture
        # 3 Layer MLP with ReLU activations
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        # Forward pass through the network
        # input state x, return Q-values for each action
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity: int = 10000):
        # Initialize the replay buffer with a fixed capacity
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        # Add a new transition to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        # Sample a batch of transitions from the buffer
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class DQN_Agent():
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        hyper_parameters: dict = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Initialize the DQN agent with the given parameters
        # list of hyper_parameters:
        # - lr: learning rate
        # - gamma: discount factor for future rewards
        # - epsilon_start: initial exploration rate
        # - epsilon_end: final exploration rate
        # - epsilon_decay: number of steps to decay epsilon
        # - epsilon: current exploration rate
        # - buffer_size: size of the replay buffer 
        self.lr = hyper_parameters['lr'] if hyper_parameters else 0.001
        self.gamma = hyper_parameters['gamma'] if hyper_parameters else 0.99
        self.epsilon_start = hyper_parameters['epsilon_start'] if hyper_parameters else 0.5
        self.epsilon_end = hyper_parameters['epsilon_end'] if hyper_parameters else 0.01
        self.epsilon_decay = hyper_parameters['epsilon_decay'] if hyper_parameters else 1000
        self.epsilon = self.epsilon_start
        self.buffer = ReplayBuffer(hyper_parameters['buffer_size'] if hyper_parameters else 10000)
        self.batch_size = hyper_parameters['batch_size'] if hyper_parameters else 64
        self.num_episodes = hyper_parameters['num_episodes'] if hyper_parameters else 500
        self.device = device
        
        # network, optimizer and loss function
        self.policy_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.logger = [] # collect avearge reward in training steps
        
    def action_selection(
        self, 
        state: int, 
        epsilon: float = 0.5
    ):
        # Select an action based on epsilon-greedy policy
        # input state numpy array, epsilon for exploration
        # return action index
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self, batch):
        # Update the model using a batch of transitions
        # input batch: list of transitions (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states), device=self.device)
        actions = torch.LongTensor(actions, device=self.device)
        rewards = torch.FloatTensor(rewards, device=self.device)
        next_states = torch.FloatTensor(np.array(next_states), device=self.device)
        dones = torch.FloatTensor(dones, device=self.device)

        # Compute Q-values for current states using policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and update the model
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self, step: int):
        # Decay epsilon based on the current step
        # input step: current training step
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)
    
    def collect_performance(self, episode: int, avg_reward: float):
        self.logger.append(avg_reward)
            
    def train(self,
        env: gym.Env
    ):
        # for each episode, reset environment, get initial state and transition
        avg_reward = 0
        for episode in tqdm(range(self.num_episodes)):
            state, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Select action using epsilon-greedy policy
                action = self.action_selection(state, self.epsilon)
                
                # Take action in the environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Store transition in replay buffer
                self.buffer.push((state, action, reward, next_state, done))
                
                # Sample a batch from the replay buffer and update the model
                if len(self.buffer) >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    self.update(batch)
                
                # Update state
                state = next_state
            avg_reward += total_reward
            # Decay epsilon after each episode
            self.decay_epsilon(episode)
            if episode % 10 == 0:
                # Update target network every 10 episodes
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if episode % 100 == 0:
                # Print progress every 100 episodes
                avg_reward /= 100
                self.collect_performance(episode, avg_reward)
                
                print(f"\nDQN Episode {episode + 1}/{self.num_episodes}, Average Reward: {avg_reward}, Epsilon: {self.epsilon:.4f}")
                avg_reward = 0
        
        # Return success train status and final total reward        
        return {'success': True, 'reward': total_reward}
    
    def save_model(self, path: str):
        # Save the model to a file
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path: str):
        # Load the model from a file
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(self.device)
        
    def evaluate(self, env: gym.Env, num_episodes: int = 10):
        # Evaluate the agent's performance in the environment
        total_rewards = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Select action using greedy policy (epsilon = 0)
                action = self.action_selection(state, epsilon=0.0)
                
                # Take action in the environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Update state
                state = next_state
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
# End of DQN.py