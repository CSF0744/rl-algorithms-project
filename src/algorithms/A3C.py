# A2C and A3C Implementation in PyTorch (for discrete action spaces)
# code from Chatgpt.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Shared Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_dim = 64
        ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


# === A2C Trainer (Single Process Synchronous) ===
class A2C_Agent:
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        hyper_parameters: dict = None, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
        self.lr = hyper_parameters['lr'] if hyper_parameters else 0.001
        self.gamma = hyper_parameters['gamma'] if hyper_parameters else 0.99
        self.batch_size = hyper_parameters['batch_size'] if hyper_parameters else 64
        self.num_episodes = hyper_parameters['num_episodes'] if hyper_parameters else 500
        self.device = device
        
        # Initialize model, optimizer and loss function
        self.model = ActorCritic(state_dim, action_dim, hidden_dim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.logger = []

    def action_selection(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs, value = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def collect_performance(self, episode: int, avg_reward: float):
        self.logger.append(avg_reward)

    def train(self, env: gym.Env):
        avg_reward = 0
        for episode in tqdm(range(self.num_episodes)):
            state = env.reset(seed=episode)[0]
            log_probs = []
            values = []
            rewards = []
            done = False
            total_reward = 0

            while not done:
                action, log_prob, value = self.action_selection(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state
                total_reward += reward

            # Compute returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(values).squeeze()
            log_probs = torch.stack(log_probs)
            advantage = returns - values.detach()

            # Compute loss
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            # entropy loss can be added for exploration
            # entropy_loss = -torch.sum(log_probs * torch.exp(log_probs)).mean()
            loss = actor_loss + 0.5 * critic_loss # + 0.01 * entropy_loss
            
            avg_reward += total_reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 100 == 0:
                avg_reward /= 100
                self.collect_performance(episode, avg_reward)
                tqdm.write(f"\nA2C Episode {episode+1}/{self.num_episodes}, Average Reward: {total_reward:.2f}")
                # print(f"\nA2C Episode {episode+1}/{self.num_episodes}, Average Reward: {total_reward:.2f}")
                avg_reward = 0
        
        return {
            'success': True,
            'reward': total_reward
        }
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
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
    
# === A3C Worker ===
class A3CWorker(mp.Process):
    def __init__(self, global_model, optimizer, env_name, global_ep, gamma=0.99):
        super().__init__()
        self.local_model = ActorCritic(global_model.fc1.in_features, global_model.actor.out_features, 64)
        self.global_model = global_model
        self.optimizer = optimizer
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.global_ep = global_ep

    def run(self):
        while self.global_ep.value < 1000:
            state = self.env.reset(seed=int(time.time()))[0]
            log_probs, values, rewards = [], [], []
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                probs, value = self.local_model(state_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                next_state, reward, done, _, _ = self.env.step(action.item())

                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(reward)
                state = next_state
                total_reward += reward

            # Compute returns
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(values).squeeze()
            log_probs = torch.stack(log_probs)
            advantage = returns - values.detach()

            # Loss and update
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()
            self.local_model.load_state_dict(self.global_model.state_dict())

            with self.global_ep.get_lock():
                self.global_ep.value += 1

            if self.global_ep.value % 10 == 0:
                print(f"A3C Episode {self.global_ep.value}, Reward: {total_reward:.2f}")


# === Run A3C ===
def train_a3c(env_name="CartPole-v1"):
    global_model = ActorCritic(4, 2, 64)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    global_ep = mp.Value('i', 0)
    workers = [A3CWorker(global_model, optimizer, env_name, global_ep) for _ in range(4)]
    [w.start() for w in workers]
    [w.join() for w in workers]
