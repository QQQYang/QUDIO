import matplotlib.pyplot as plt
import numpy as np
import os

from ground_energy import GroundEnergy as GE

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd']
color = [(248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189), (23, 190, 207), (0, 129, 204)]
marker = ["o", "s", "^", '+']
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">", "o", "s", "d"]

# plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'bold', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

font = {
    'family':'Times New Roman',
    # 'serif':'Times New Roman',
    'style':'normal',
    'weight':'bold', #or 'bold'
}

iter_gaps = [1, 2, 4, 8, 16, 32]
atom_dist = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 6)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

aggre_strategy = ['random_seed_average', 'random_seed_median', 'random_seed_random', 'random_seed_weight']
aggre_strategy_label = ['Average', 'Median', 'Random', 'Weighted']

energy_error = []
energy_stds = []
energy_gap = []
workers = [1, 2, 4, 8, 16]
data = {}
for j, aggre in enumerate(aggre_strategy):
    data[aggre] = []
    iter_gap = 32
    energy_mean, energy_std, error, error_std = [], [], [], []
    energy_dist = []
    for i, distance in enumerate([atom_dist[1]]):
        for worker in workers:
            for iter_gap in iter_gaps:
                for seed in range(0, 5):
                    if not os.path.exists(os.path.join('logs/vqe/ideal/LiH/{}'.format(aggre), str(seed), 'energy_{}_{}_0.0_0_0_'.format(worker, iter_gap)+str(distance)+'.npy')):
                        print(os.path.join('logs/vqe/ideal/LiH/{}'.format(aggre), str(seed), 'energy_{}_{}_0.0_0_0_'.format(worker, iter_gap)+str(distance)+'.npy'))
                        continue
                    data[aggre].append(np.load(os.path.join('logs/vqe/ideal/LiH/{}'.format(aggre), str(seed), 'energy_{}_{}_0.0_0_0_'.format(worker, iter_gap)+str(distance)+'.npy'))-GE['LiH'][1])
    ax.hist(data[aggre], bins=100, density=True, color=np.array(color[j])/255, cumulative=True, histtype='step', label=aggre_strategy_label[j], linewidth=2.5)
    ax.grid(True)
    ax.tick_params(labelsize=20)

ax.set_ylabel('Probability', font, fontsize=20)
ax.set_xlabel('Approximation error', font, fontsize=20)
leg = ax.legend(fontsize=20, loc='center')
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.tight_layout()

# fig.legend(loc='best')
# plt.savefig('figure/aggre_lih_cdf.png')
# plt.savefig('figure/aggre_lih_cdf.pdf')
plt.show()

plt.clf()
plt.close()