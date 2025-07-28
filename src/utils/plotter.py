import matplotlib.pyplot as plt
import os

def plot_training_results(reward_list, save_path="figure/rewards.png", title='Training Rewards Curve'):
    """Plot the average rewards over training episodes.
    Args:
        reward_list (list): list of average rewards from agent training.
        save_path (str, optional): path to save figure. 
        title (str, optional): figure title. 
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not reward_list:
        print("No rewards to plot.")
        return
    else:
        plt.plot(reward_list)
        plt.xlabel('Episodes')
        plt.xticks(labels=range(0, 100*len(reward_list), 100), ticks=range(0, len(reward_list)))
        plt.ylabel('Average Reward')
        plt.title(title)
        plt.savefig(save_path)
        plt.show()
        # plt.close()