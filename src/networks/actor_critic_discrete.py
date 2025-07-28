import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """Discrete actor critic with shared first layer
    F(s) --> p(a|s), v(s)
    """
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
    
class Actor(nn.Module):
    """Discrete independent actor
    pi(a|s)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64
    ):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self,state: torch.Tensor):
        # return action logit
        return self.policy_net(state)
    
class Critic(nn.Module):
    """Value newtork for critic
    V(s)
    """
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int = 64
    ):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self,state: torch.Tensor):
        # return state value
        return self.value_net(state)