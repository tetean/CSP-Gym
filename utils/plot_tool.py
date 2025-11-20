from datetime import datetime

from utils.csv_read import load_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__INT_MAX = (1 << 31) - 1
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1]) + 1

    return (fig_width_in, fig_height_in)


def draw_single(data):

    x = len(data)

    fig = plt.figure(figsize=set_size(432, 0.99, (1, 1), height_add=0))
    ax = fig.subplots(ncols=1)
    x = [i for i in range(x)]
    ax.plot(x, data, label="PPO")
    plt.show()
    plt.savefig('res_fig.pdf', format='pdf', bbox_inches='tight', transparent=True)

def draw_multi(data):
    rewards = {}
    stk_top = __INT_MAX

    fig = plt.figure(figsize=set_size('thesis', 0.99, (1, 1), height_add=0))

    ax = fig.subplots(ncols=1)
    for algorithm, locations in data.items():
        rewards[algorithm] = [load_csv(locations[i])[1] for i in range(len(locations))]
        for i in range(len(rewards[algorithm])):
            rewards[algorithm][i] = moving_average(rewards[algorithm][i], 100)

            if len(rewards[algorithm][i]) < stk_top:
                stk_top = len(rewards[algorithm][i])

        for i in range(len(rewards[algorithm])):
            rewards[algorithm][i] = rewards[algorithm][i][:stk_top]

        reward_mean = np.mean(np.array(rewards[algorithm]), axis=0)
        reward_std = np.std(np.array(rewards[algorithm]), axis=0)

        x = [_ for _ in range(stk_top)]

        ci = (reward_mean - 2 * reward_std, reward_mean + 2 * reward_std)
        ax.plot(x, reward_mean, label=algorithm)
        ax.fill_between(x, ci[0], ci[1], alpha=0.2)

    plt.legend(loc='lower right')
    plt.title("")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward")
    # plt.show()
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'res_fig_{current_time}.pdf', format='pdf')


def plot_box(data_loc):

    df = pd.read_csv(data_loc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    median_props = {'color': 'black', 'linewidth': 2}

    sns.boxplot(x='Algorithm', y='SS', data=df, ax=ax1, palette="Set2", showfliers=False,
                medianprops=median_props)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('SS (Å)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    sns.boxplot(x='Algorithm', y='EE', data=df, ax=ax2, palette="Set2", showfliers=False,
                medianprops=median_props)
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('EE (eV/atom)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.show()

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'res_fig_{current_time}.pdf', format='pdf')

if __name__ == '__main__':
    file_loc = ['/home/wangxiean/PycharmProjects/CSP-gym/results/SAC/20250316_022329/monitor.csv']
    timePPOArr = []
    rewardPPOArr = []
    for name in file_loc:
        times, rewards = load_csv(name)
        timePPOArr.append(times)
        rewardPPOArr.append(rewards)

    file_loc = {
                'PPO': [
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/PPO/20250327_145040/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/PPO/20250327_082234/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/PPO/20250327_002135/monitor.csv'
                        ],
                'SAC': [
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/SAC/20250327_153701/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/SAC/20250327_090743/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/SAC/20250327_012630/monitor.csv'
                        ],
                'TD3': [
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/TD3/20250327_182005/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/TD3/20250327_114832/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/TD3/20250327_050216/monitor.csv'
                        ],
                'A2C': [
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/A2C/20250327_193156/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/A2C/20250327_125554/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/A2C/20250327_062904/monitor.csv'
                        ],
                'DDPG': [
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/DDPG/20250327_201520/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/DDPG/20250327_133555/monitor.csv',
                        '/home/wangxiean/PycharmProjects/CSP-gym/results/DDPG/20250327_071053/monitor.csv'
                        ],
                }

    draw_multi(file_loc)
    plot_box('/home/wangxiean/PycharmProjects/CSP-gym/evaluation_results/episode_rec_SiC20250328_223235.csv')