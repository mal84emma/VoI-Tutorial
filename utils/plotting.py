import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cost_dist(costs_list):

    fig, ax = plt.subplots()

    sns.kdeplot(np.array(costs_list)/1e3, color='k', lw=2, cut=0.01)

    ymax = ax.get_ylim()[1]
    ax.vlines(np.mean(costs_list)/1e3,0,ymax,color='k', alpha=0.7, ls='--', lw=2)

    ax.set_ylim(0,ymax)
    ax.get_yaxis().set_ticks([])
    ax.yaxis.set_label_text('Density')
    ax.xaxis.set_label_text('Av. annual cost (£k/yr)')

    plt.tight_layout()
    plt.show()