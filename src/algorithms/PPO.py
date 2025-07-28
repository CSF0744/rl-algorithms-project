# basic package
import numpy as np
import math
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

# NN package
import torch
import torch.nn.functional as F
from torch import nn

class PPO(nn.Module):
    # PPO algorithm that gives a policy net and a value net
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64
    ):
        super().__init__()
        # Define the neural network architecture
        # policy network as actor
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # value network as critic
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor):
        # Forward pass through the network
        # input state x, return action logit and state value
        action_logit = self.policy_net(x)
        state_value = self.value_net(x)
        return action_logit, state_value
    
class PPO_buffer():
    # buffer for PPO algorithm to store transitions
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()
    
    def reset(self):
        # reset the buffer to empty
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        
    def push(self, state, action, reward, done, value, log_prob):
        # Add a new transition to the buffer
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_trajectory(self, end_value):
        # compute advantages and returns for one trajectory using GAE
        # append advantages and returns to buffer
        advantages = []
        returns = []
        gae = 0
        values = self.values + [end_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1-self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1-self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        self.advantages.extend(advantages)
        self.returns.extend(returns)
    
    def sample(self, batch_size: int, device: str = 'cpu'):
        # Sampel a batch of transition from buffer
        if len(self.states) < batch_size:
            return None
        
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        
        # normalize advantages to lower variance in batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                'states' : states[batch_idx],
                'actions' : actions[batch_idx],
                'returns' : returns[batch_idx],
                'advantages' : advantages[batch_idx],
                'log_probs' : log_probs[batch_idx]
            }
        
        self.reset()  # Reset the buffer after sampling
                
    def __len__(self):
        return len(self.states)
    
class PPO_Agent():
    # PPO agent that implement PPO algorithm
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        hyper_parameters: dict = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.lr = hyper_parameters['lr'] if hyper_parameters else 0.001
        self.gamma = hyper_parameters['gamma'] if hyper_parameters else 0.99
        self.lam = hyper_parameters['lam'] if hyper_parameters else 0.95
        
        self.batch_size = hyper_parameters['batch_size'] if hyper_parameters else 64
        self.num_episodes = hyper_parameters['num_episodes'] if hyper_parameters else 500
        self.eps_clip = hyper_parameters['eps_clip'] if hyper_parameters else 0.2
        self.device = device
        
        # Initialize the PPO buffer for storing transitions
        self.buffer = PPO_buffer(gamma=self.gamma, lam = self.lam)
        # Initialize the PPO networks
        self.model = PPO(state_dim, action_dim, hidden_dim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.logger = [] #  collect average reward in training steps

    def action_selection(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits = logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.item()

    def update(self, epoches: int = 10, batch_size: int = 64):
        # check buffer enough for training
        if len(self.buffer) < batch_size:
            return
        # Update the PPO model using the collected transitions
        for _ in range(epoches):
            for batch in self.buffer.sample(self.batch_size, self.device):
                states = batch['states']
                actions = batch['actions']
                returns = batch['returns']
                advantages = batch['advantages']
                old_log_probs = batch['log_probs']
                
                logits, values = self.model(states)
                dist = torch.distributions.Categorical(logits = logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions)
                
                # surrogate loss for actor
                ratios = torch.exp(new_log_probs - old_log_probs)
                sur1 = ratios * advantages
                sur2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(sur1, sur2).mean()
                # loss for critic
                critic_loss = self.criterion(values.squeeze(-1), returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    def collect_performance(self, episode: int, avg_reward: float):
        self.logger.append(avg_reward)
        
            
    def train(self, env: gym.Env):
        avg_reward = 0
        for episode in tqdm(range(self.num_episodes)):
            state, _ = env.reset()
            done = False
            total_reward = 0
            # run for whole trajectory
            while not done:
                action, log_prob, value = self.action_selection(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # store to buffer
                self.buffer.push(state, action, reward, done, value, log_prob)
                
                state = next_state
            # compute advantages and returns
            avg_reward += total_reward
            with torch.no_grad():
                _, end_value = self.model(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.buffer.compute_trajectory(end_value.item())
            
            self.update(epoches=10, batch_size=self.batch_size)
            
            if episode % 100 == 0:
                avg_reward /= 100
                self.collect_performance(episode, avg_reward)
                print(f"\nPPO Episode {episode+1}/{self.num_episodes}, Average Reward: {avg_reward}")
                avg_reward = 0
        
        return {
            'success': True,
            'reward': total_reward
        }  
                             
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def evaluate(self, env: gym.Env, num_episodes: int = 10):
        total_rewards = []
        for _ in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _, _ = self.action_selection(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                state = next_state
                
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f'\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}')
        return avg_reward
    
# End of PPO.py