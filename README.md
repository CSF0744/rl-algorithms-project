# Reinforcement Learning Algorithms Project

This project is designed to implement and test various reinforcement learning algorithms. It provides a structured framework for developing, training, and evaluating different algorithms in custom environments.

## Project Structure

```
rl-algorithms-project
├── figure
├── model
├── src
│   ├── main.py                # Entry point for the application
│   ├── algorithms             # Contains implementations of RL algorithms
│   │   └── __init__.py
|   |   └── A3C.py
|   |   └── DQN.py
|   |   └── PPO.py
│   ├── environments           # Custom environments or wrappers
│   │   └── __init__.py
│   │   └── custom_env.py
│   ├── utils                  # Utility functions and classes
│   │   └── __init__.py
│   │   └── config.py
│   │   └── plotter.py
│   └── tests                  # Unit tests for algorithms
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
python src/main.py --algo dqn --env CartPole-v1 --num_episodes 1000
```

## Algorithms

This project includes implementations of various reinforcement learning algorithms, such as:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Actor-Critic) __WIP__

## Environments

Custom environments can be defined or existing Gym environments can be wrapped for use with the algorithms.

## Testing

Unit tests for the algorithms are located in the `tests` directory. To run the tests, use:
```
pytest src/tests/test_algorithms.py
```

## Contribution

