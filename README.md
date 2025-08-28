# Reinforcement Learning Algorithms Project

This project is designed to implement and test various reinforcement learning algorithms. It provides a structured framework for developing, training, and evaluating different algorithms in custom environments.

## Project Structure

```
rl-algorithms-project
├── figure                     # Store figure output
├── model                      # Save model parameters
├── src
│   ├── main.py                # Entry point for the application
│   ├── algorithms             # Contains implementations of RL algorithm Agents
│   │   └── __init__.py
|   |   └── A3C.py
|   |   └── DQN.py
|   |   └── PPO.py
│   ├── environments           # Custom environments or wrappers
│   │   └── __init__.py
│   │   └── custom_env.py
│   ├── memory                 # Buffer for replay
│   │   └── __init__.py
│   │   └── buffer.py
│   ├── networks               # Custom network model folder
│   │   └── actor_critic_continuous.py
│   │   └── actor_critic_discrete.py
│   │   └── q_net_discrete.py
│   ├── utils                  # Utility functions and classes
│   │   └── __init__.py
│   │   └── config.py
│   │   └── plotter.py
│   └── tests                  # Unit tests for algorithms and environments
│       └── __init__.py
│       └── test_algorithms.py
│       └── test_envs.py
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files to ignore in version control
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rl-algorithms-project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Example Usage

To run the project, execute the `main.py` file:
```
python -m src.main --algo dqn --env CartPole-v1 --num_episodes 1000
```

## Algorithms

This project includes implementations of various reinforcement learning algorithms, such as:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Actor-Critic) __WIP__
- DDPG __WIP__

## Performance

DQN, PPO and A2C performance in CartPole-v1 environment

[DQN Reward Curve](/figure/dqn/train_reward.png)

[PPO Reward Curve](/figure/ppo/train_reward.png)

[A2C Reward Curve](/figure/a2c/train_reward.png)

In Car2DEnv custom environment

[DQN Reward Curve](/figure/dqn/car2denv-v0_train_reward.png)

## Environments

Custom environments can be defined or existing Gym environments can be wrapped for use with the algorithms.

## Testing

Unit tests for the algorithms are located in the `tests` directory. To run the tests, use:
```
python -m src.tests.test_algorithms
```

## Contribution

## Reference
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.  [https://doi.org/10.1038/nature14236](https://doi.org/10.1038/nature14236)

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.  [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *International Conference on Machine Learning (ICML)*.  [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous Advantage Actor-Critic Agents. *arXiv preprint arXiv:1602.01783*.  [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.  [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)