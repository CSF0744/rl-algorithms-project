import random
import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, capacity: int = 10000):
        # Initialize the replay buffer with a fixed capacity
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Add a new transition to the buffer
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = 'cpu', discrete: bool = True):
        # Sample a batch of transitions from the buffer
        if len(self.buffer) < batch_size:
            return None
        
        # indices = np.arange(len(self.buffer))
        # np.random.shuffle(indices)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Unpack entire buffer into separate arrays
        # states, actions, rewards, next_states, dones = zip(*self.buffer)
        
        actions_dtype = torch.int64 if discrete else torch.float32
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=actions_dtype, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)
        
        return {
            'states' : states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        # # Yield mini-batches
        # for start in range(0, len(self.buffer), batch_size):
        #     end = start + batch_size
        #     batch_idx = indices[start:end]

        #     yield {
        #         'states': states[batch_idx],
        #         'actions': actions[batch_idx],
        #         'rewards': rewards[batch_idx],
        #         'next_states': next_states[batch_idx],
        #         'dones': dones[batch_idx]
        #     }

    def __len__(self):
        return len(self.buffer)
    
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
    
class A2C_buffer():
    # buffer for A2C algorithm to store transitions
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
        # compute advantages and returns for one trajectory using simple advantage
        # append advantages and returns to buffer
        returns = []
        G = end_value
        for t in reversed(range(len(self.rewards))):
            G = self.rewards[t] + self.gamma * G * (1-self.dones[t])
            returns.insert(0, G)
            
        self.returns.extend(returns)
        advantages = [r - v for r, v in zip(returns, self.values)]
        self.advantages.extend(advantages)
    
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