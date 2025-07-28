config = {
    "dqn" : {
        'lr' : 0.001, # Learning rate
        'gamma': 0.99, # Discount factor
        'epsilon_start' : 1.0, # Start value for epsilon-greedy policy
        'epsilon_end' : 0.1, # End value for epsilon-greedy policy
        'epsilon_decay' : 1000, # Decay steps for epsilon
        'buffer_size': 10000, # Size of the replay buffer
        'batch_size': 64, # Batch size for training
        'num_episodes' : 1000, # Number of episodes to train
    },
    "ppo" : {
        'lr':0.001,  # Learning rate
        'gamma':0.99,  # Discount factor
        'lam':0.95,  # PPO clipping parameter
        'batch_size':64,  # Batch size for training
        'num_episodes':1000,  # Number of episodes to train
        'eps_clip':0.2 # PPO clip parameter
    },
    "a2c" : {
        
    }
}