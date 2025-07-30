import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import gymnasium as gym
from algorithms import DQN_Agent, PPO_Agent, A2C_Agent

import environments
from utils.config import config
from utils.plotter import plot_training_results
from utils.render_agent import render_trained_agent
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='dqn', choices=['dqn', 'ppo', 'a2c'])
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_episodes', type=int, default=1000)
    args = parser.parse_args()
    return args

def make_output_dir(algo_name):
    model_dir = os.path.join('model',algo_name.lower())
    figure_dir = os.path.join('figure',algo_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir,exist_ok=True)
    return model_dir, figure_dir

def main():
    """Main function to train different RL agents
       Using keyword to select agent type
        Internally define hyperparameters and environment, change if necessary
    Args:
        agent_type (str, optional): _description_. Defaults to 'DQN'.
    """   
    # get arguments
    args = parse_args()
    algo = args.algo
    
    algo_config = config.get(algo, {})
    if args.num_episodes is not None:
        algo_config['num_episode'] = args.num_episodes
        
    print(f'\n Running {algo} on {args.env} with config:')
    for key, value in algo_config.items():
        print(f'\n {key} : {value}')
    
    # generate output path
    model_dir, figure_dir = make_output_dir(args.algo)
        
    # initialize environment and agent
    env_name = args.env
    env = gym.make(env_name) 
    env.reset()
    
    # set hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 64  # Example hidden dimension for the neural network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # initialize agent
    if args.algo == 'dqn':
        agent = DQN_Agent(state_dim, action_dim, hidden_dim, algo_config, device)
    elif args.algo == 'ppo':
        print('\nNot available now, fixed the action_selection to use')
        return -1
        agent = PPO_Agent(state_dim, action_dim, hidden_dim, algo_config, device)
    elif args.algo == 'a2c':
        print('\nNot available now, fixed the action_selection to use')
        return -1
        agent = A2C_Agent(state_dim, action_dim, hidden_dim, algo_config, device)
    else:
        print('\nAgent not exist!')
        return -1
    # train agent 
    agent.train(env)
    # save the model and plot training performance
    agent.save_model(os.path.join(model_dir,f'{env_name.lower()}_model.pth'))
    plot_training_results(agent.logger, save_path=os.path.join(figure_dir,f'{env_name.lower()}_train_reward.png'), title='Training Rewards Curve')
    
    # evaluate the agent
    env = gym.make(env_name, render_mode='human')
    agent.evaluate(env, num_episodes=10)  
    # evalute agent with rendering
    render_trained_agent(agent, env, True)

if __name__ == "__main__":
    main()