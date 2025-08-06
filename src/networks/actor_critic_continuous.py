import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        max_action
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x_a = F.relu(self.actor(x))
        x_c = F.relu(self.critic(x))
        return F.tanh(x_a) * self.max_action, 

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


class GaussianActor(nn.Module):
    """
    network that gives action in mean and log_std and sample with Gaussian
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
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # limit the std in range e^-20 to e^2.
        return mean, log_std
        
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1-action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
## for more complecated action space, use tanh to ensure in range [-1, 1]. then scale action into correct
## range via the following
## low + 0.5 * (action + 1) * (high - low)