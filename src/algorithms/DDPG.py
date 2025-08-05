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

# custom file
from src.memory.buffer import ReplayBuffer
from src.networks.actor_critic_continuous import Actor, Critic

class DDPG_Agent():
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action,
        hidden_dim: int = 64,
        hyper_parameters: dict = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):  
        self.actor_lr = hyper_parameters['actor_lr'] if hyper_parameters else 0.001
        self.critic_lr = hyper_parameters['critic_lr'] if hyper_parameters else 0.001
        self.gamma = hyper_parameters['gamma'] if hyper_parameters else 0.99
        self.tau = hyper_parameters['tau'] if hyper_parameters else 0.99
        self.batch_size = hyper_parameters['batch_size'] if hyper_parameters else 64
        self.num_episodes = hyper_parameters['num_episodes'] if hyper_parameters else 500
        self.buffer = ReplayBuffer(hyper_parameters['buffer_size'] if hyper_parameters else 10000)
        
        self.max_action = max_action
        self.device = device
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.logger = [] # collect avearge reward in training steps
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    def action_selection(self, state, noise_std=0.1, eval: bool =False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_std != 0:
            action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def collect_performance(self, episode: int, avg_reward: float):
        self.logger.append(avg_reward)
    
    def update(self, epoches: int, batch_size: int):
        if len(self.buffer)<batch_size:
            return
        
        for _ in range(epoches):
            batch = self.buffer.sample(batch_size, self.device, discrete=False)
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            dones = batch['dones']
            
            # critic update
            q_values = self.critic(states, actions)
            
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q_values = self.critic_target(next_states, next_actions)
                target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
            
            critic_loss = nn.MSELoss()(q_values, target_q_values)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # actor update
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # softupdate parameters in target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
    
    def train(self, env: gym.Env, noise_std=0.1):
        avg_reward = 0
        for episode in tqdm(range(self.num_episodes)):
            state, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # select action
                action = self.action_selection(state, noise_std)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                self.buffer.push(state, action, reward, next_state, done)
                if len(self.buffer)>= self.batch_size:
                    self.update(1, self.batch_size)
                
                state = next_state
                
            avg_reward += total_reward
            if episode % 100 == 0:
                # Print progress every 100 episodes
                avg_reward /= 100
                self.collect_performance(episode, avg_reward)
                
                tqdm.write(f"\nDDPG Episode {episode + 1}/{self.num_episodes}, Average Reward: {avg_reward}")
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
        