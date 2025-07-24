import numpy as np
import gymnasium as gym
from src.algorithms.DQN import DQN_Agent
from src.algorithms.PPO import PPO_Agent
from src.algorithms.A3C import A2CAgent # Assuming these classes are defined in algorithms module
from src.environments import CustomEnv1  # Assuming a custom environment is defined in environments module
import torch

def main(agent_type: str = 'DQN'):
    """Main function to train different RL agents
       Using keyword to select agent type
        Internally define hyperparameters and environment, change if necessary
    Args:
        agent_type (str, optional): _description_. Defaults to 'DQN'.
    """
    # Initialize the environment
    env = gym.make('CartPole-v1', render_mode='rgb_array') 
    env.reset()
    
    # set hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 64  # Example hidden dimension for the neural network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if agent_type == 'DQN':
        print("\nUsing DQN Agent!")
        hyper_dict = dict(
            lr=0.001,  # Learning rate
            gamma=0.99,  # Discount factor
            epsilon_start=1.0,  # Start value for epsilon-greedy policy
            epsilon_end=0.1,  # End value for epsilon-greedy policy
            epsilon_decay=1000,  # Decay steps for epsilon
            buffer_size=10000,  # Size of the replay buffer
            batch_size=64,  # Batch size for training
            num_episodes=1000  # Number of episodes to train
        )
    
        # Choose the reinforcement learning algorithm
        agent = DQN_Agent(state_dim, action_dim, hidden_dim, hyper_dict, device)  # You can switch to PPO or A2C as needed

        # train the agent
        agent.train_DQN(env)
        
        # save the model
        agent.save_model('model/dqn_agent_1000.pth')
        # load the model
        #agent.load_model('model/dqn_agent_1000.pth')
    elif agent_type == 'PPO':
        print("\nUsing PPO Agent!")
        hyper_dict = dict(
            lr=0.001,  # Learning rate
            gamma=0.99,  # Discount factor
            lam=0.95,  # PPO clipping parameter
            batch_size=64,  # Batch size for training
            num_episodes=1000,  # Number of episodes to train
            eps_clip=0.2 # PPO clip parameter
        )
        
        agent = PPO_Agent(state_dim, action_dim, hidden_dim, hyper_dict, device)
        agent.train_PPO(env)
        
        # save the model
        agent.save_model('model/ppo_agent_1000.pth')
        # load the model
        #agent.load_model('model/dqn_agent_1000.pth')
    # evaluate the agent
    agent.print_performance()
    avg_reward = agent.evaluate(env, num_episodes=10)
    # print(f'Average reward over 10 episodes: {avg_reward}')    

if __name__ == "__main__":
    main('PPO')