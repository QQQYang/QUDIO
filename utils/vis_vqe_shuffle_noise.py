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

iter_gaps = [32]
atom_dist = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 6)) # (12.8, 5.5)
ax.grid(True, axis='y')
title = ['(a)', '(b)', '(c)']

aggre_strategy = ['baseline_average', 'shuffle_average']
aggre_strategy_label = ['QUDIO', 'Shuffle-QUDIO']

energy_error = []
energy_stds = []
energy_gap = []
workers = [1, 2, 4, 8, 16]
data = {}
aggre = 'baseline_average'
seed = 0
iter_gap = 32
worker = 4
shots = [1e2]
noises = [0.0, 0.1, 0.2, 0.3, 1.0]
noise_label = ['ideal', 'p=0.1', 'p=0.2', 'p=0.3', 'NISQ']
# noises = [1.0]
# noise_label = ['NISQ']
# ax.set_xscale('log', base=10)
for k, aggre in enumerate(aggre_strategy):
    energy_aggre = []
    for w, noise in enumerate(noises):
        energys = []
        energy_std = []
        for j, shot in enumerate(shots):
            for iter_gap in iter_gaps:
                for i, distance in enumerate([atom_dist[1]]):
                    energy_seed = []
                    for seed in range(5):
                        if not os.path.exists(os.path.join('logs/vqe/ideal/h2/{}'.format(aggre), str(seed), 'energy_{}_{}_{}_{}_0_'.format(worker, iter_gap, noise, int(shot))+str(distance)+'.npy')):
                            print(os.path.join('logs/vqe/ideal/h2/{}'.format(aggre), str(seed), 'energy_{}_{}_{}_{}_0_'.format(worker, iter_gap, noise, int(shot))+str(distance)+'.npy'))
                            print('aggre={}, noise={}, seed={}'.format(aggre, noise, seed))
                            continue
                        energy_seed.append(np.load(os.path.join('logs/vqe/ideal/h2/{}'.format(aggre), str(seed), 'energy_{}_{}_{}_{}_0_'.format(worker, iter_gap, noise, int(shot))+str(distance)+'.npy')))
                    energys.append(np.min(energy_seed))
                    energy_std.append(np.std(energy_seed))
            # ax.grid(True)
            ax.tick_params(labelsize=20)
            energy_aggre.append(np.abs(np.mean(energys)-GE['H2'][1]))

        # ax.plot(noises, energys, color=np.array(color[k])/255, label=aggre_strategy_label[k])
    ax.bar(np.arange(len(energy_aggre))*2.5+k, energy_aggre, width=1.0, fc=np.array(color[k])/255, label=aggre_strategy_label[k], edgecolor='k')#, tick_label=noise_label)
    for x, y in zip(np.arange(len(energy_aggre))*2.5+k, energy_aggre):
        ax.text(x-0.5, y+0.01, str(round(y, 3)), fontsize=15)
# ax.plot(noises, [GE['H2'][1]]*len(noises), color='k')
ax.set_ylabel(r'$Err$', font, fontsize=20)
ax.set_xlabel('Noise', font, fontsize=20)
ax.set_xticks(np.arange(len(energy_aggre))*2.5+0.5)
ax.set_xticklabels(noise_label)
leg = ax.legend(fontsize=20, loc='best')
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.tight_layout()

# fig.legend(loc='best')
plt.savefig('figure/h2_noise.png')
plt.savefig('figure/h2_noise.pdf')
plt.show()

plt.clf()
plt.close()