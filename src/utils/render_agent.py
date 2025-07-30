import gymnasium as gym

def render_trained_agent(agent, env: gym.Env, render: bool=False):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # only works for DQN currently
        action = agent.action_selection(state, eval = True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += reward
        state = obs
        if render:
            env.render()
    
    print(f'\nEvaluation result: Reward is {total_reward}')
    env.close()
    return None