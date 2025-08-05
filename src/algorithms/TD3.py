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
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        super().__init__(state_dim, action_dim, hidden_dim, config)