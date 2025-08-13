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
from .base import BaseAgent

class TD3_Agent(BaseAgent):
    def __init__(
        self, 
        state_dim, 
        action_dim,
        max_action,
        hidden_dim, 
        config
    ):
        super().__init__(state_dim, action_dim, hidden_dim, config)
        self.actor_lr = config.get('actor_lr',0.001)
        self.critic_lr = config.get('critic_lr',0.001)
        self.gamma = config.get('gamma',0.99)
        self.tau = config.get('tau',0.99)
        self.max_action = max_action
        self.policy_noise = config.get('policy_noise',0)
        self.noise_clip = config.get('noise_clip', 0)
        
        self.buffer = ReplayBuffer(config.get('buffer_size',10_000))
        self.batch_size = config.get('batch_size', 64)
        self.num_episodes = config.get('num_episodes', 500)
        self.device = config.get('device','cpu')
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
        self.logger = []
        
    def action_selection(self, state, **kwargs):
        noise_std = kwargs.get('noise_std',0.1)
        eval = kwargs.get('eval',False)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_std != 0:
            action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action), None    
    
    def collect_performance(self, episode, avg_reward: float):
        self.logger.append(avg_reward)
        
    def update(self, epoches, batch_size, policy_freq: int = 2):
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
            with torch.no_grad():
                # target policy smoothing?
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
                
                target_q1_values = self.critic1_target(next_states, next_actions)
                target_q2_values = self.critic2_target(next_states, next_actions)
                target_q_values = torch.min(target_q1_values, target_q2_values)
                target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
            
            q1_values = self.critic1(states, actions)
            q2_values = self.critic2(states, actions)
            critic1_loss = nn.MSELoss()(q1_values, target_q_values)
            critic2_loss = nn.MSELoss()(q2_values, target_q_values)
            # critic 1 update
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            # critic 2 update
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            if self.total_it % policy_freq == 0:
                # actor update less frequently
                actor_loss = -self.critic1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            
            # softupdate parameters in target
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
                
            self.total_it += 1
                
    def train(self, env, noise_std = 0.1, **kwargs):
        self.total_it = 0
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
                
                tqdm.write(f"\nTD3 Episode {episode + 1}/{self.num_episodes}, Average Reward: {avg_reward}")
                avg_reward = 0
        
        # Return success train status and final total reward        
        return {'success': True, 'reward': total_reward}            
    
    def save_model(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(self.device)
        
    def evaluate(self, env, num_episodes = 10, render = False):
        return super().evaluate(env, num_episodes, render)

# End of TD3.py