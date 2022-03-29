import matplotlib.pyplot as plt
import numpy as np
import os

from ground_energy import GroundEnergy as GE

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189), (23, 190, 207)]
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

fig, ax = plt.subplots(2, 6, sharey=True, figsize=(12.0, 7.0)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

# median sync
energy_error = []
energy_stds = []
energy_gap = []
for j, iter_gap in enumerate(iter_gaps):
    energy_mean, energy_std, error, error_std = [], [], [], []
    energy_dist = []
    for i, distance in enumerate([atom_dist[1]]):
        energy_seed = []
        for seed in range(0, 5):
            energy = []
            for subworker in range(32):
                name = 'loss_32_{}_1.0_100_{}_'.format(iter_gap, subworker)+str(distance)+'.npy'            
                energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/baseline', str(seed), name)))
            energy_all = np.sum(energy, axis=0)[33:200:32]
            energy_seed.append(energy_all)
            if seed == 0:
                ax[0][j].plot(range(32, 200, 32), energy_all, label='W={}'.format(iter_gap), color=np.array(color[j%len(color)])/255, alpha=1) # marker=marker[j//len(marker)]
            else:
                ax[0][j].plot(range(32, 200, 32), energy_all, color=np.array(color[j%len(color)])/255, alpha=0.5) # marker=marker[j//len(marker)]
        energy_dist.append(np.mean(energy_seed, axis=0))
    energy_gap.append(np.mean(energy_dist, axis=0))
    x = range(32, 200, 32)
    ax[0][j].plot(x, [GE['LiH'][1]]*len(x), '--', color='k', label='GSE', linewidth=2.0)
    ax[0][j].set_xticks(range(32, 200, 64))
    ax[0][j].tick_params(labelsize=20)
    # ax[0][j].set_xlabel('Iteration', fontsize=20)
    ax[0][j].legend(loc='upper center', fontsize=15, frameon=True, ncol=1)
    ax[0][j].grid(True)
# ax[0].plot(atom_dist, GE['LiH'][:len(atom_dist)], label='ExactEigensolver', color='k', linewidth=2.0)
ax[0][0].set_ylabel(r'$Tr(\rho H)$ (QUDIO)', font, fontsize=20)
# ax[0].set_title('Baseline', fontsize=20)

energy_error = []
energy_stds = []
energy_gap = []
for j, iter_gap in enumerate(iter_gaps):
    energy_mean, energy_std, error, error_std = [], [], [], []
    energy_dist = []
    for i, distance in enumerate([atom_dist[1]]):
        energy_seed = []
        for seed in range(0, 5):
            energy = []
            for subworker in range(32):
                name = 'loss_32_{}_1.0_100_{}_'.format(iter_gap, subworker)+str(distance)+'.npy'            
                energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/shuffle', str(seed), name)))
            energy_all = np.sum(energy, axis=0)[33:200:32]
            energy_seed.append(energy_all)
            if seed == 0:
                ax[1][j].plot(range(32, 200, 32), energy_all, label='W={}'.format(iter_gap), color=np.array(color[j%len(color)])/255, alpha=1) # marker=marker[j//len(marker)]
            else:
                ax[1][j].plot(range(32, 200, 32), energy_all, color=np.array(color[j%len(color)])/255, alpha=0.5) # marker=marker[j//len(marker)]
        energy_dist.append(np.mean(energy_seed, axis=0))
    energy_gap.append(np.mean(energy_dist, axis=0))
    x = range(32, 200, 32)
    ax[1][j].plot(x, [GE['LiH'][1]]*len(x), '--', color='k', label='GSE', linewidth=2.0)
    ax[1][j].set_xticks(range(32, 200, 64))
    ax[1][j].tick_params(labelsize=20)
    ax[1][j].set_xlabel('Iteration', font, fontsize=20)
    ax[1][j].legend(loc='upper center', fontsize=15, frameon=True, ncol=1)
    ax[1][j].grid(True)
# ax[0].plot(atom_dist, GE['LiH'][:len(atom_dist)], label='ExactEigensolver', color='k', linewidth=2.0)
ax[1][0].set_ylabel(r'$Tr(\rho H)$ (Ours)', font, fontsize=20)

plt.tight_layout()

# fig.legend(loc='best')
# plt.savefig('figure/loss_LiH.png')
# plt.savefig('figure/loss_LiH.pdf')
plt.show()

plt.clf()
plt.close()