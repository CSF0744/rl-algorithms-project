from src.environments.custom_env import Car2DEnv
from gymnasium.envs.registration import register

register(
    id='Car2DEnv-v0',
    entry_point='src.environments.custom_env:Car2DEnv'
)
    
__all__ = ['Car2DEnv']