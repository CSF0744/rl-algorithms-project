import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    policy that gives continuous action in space [-max_action, action]^action_dim
    pi(s) --> a
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        hidden_dim, 
        max_action
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


class Critic(nn.Module):
    """
    value that gives value
    Q(s,a) -- > v
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        hidden_dim : int = 64
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
## for more complecated action space, use tanh to ensure in range [-1, 1]. then scale action into correct
## range via the following
## low + 0.5 * (action + 1) * (high - low)