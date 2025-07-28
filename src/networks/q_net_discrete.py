import torch
import torch.nn.functional as F
from torch import nn

class QNetDiscrete(nn.Module):
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