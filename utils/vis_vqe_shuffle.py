import matplotlib.pyplot as plt
import numpy as np
import os

from ground_energy import GroundEnergy as GE

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd']
color = ['C5E1A5', 'FFB74D', 'FFF176', '9E86C9', 'E57373', 'F48FB1', 'E6E6E6', '90CAF9']
color = [(int(s[:2],16), int(s[2:4],16), int(s[4:],16)) for s in color]
# color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189), (23, 190, 207)]
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

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(14.0, 5.5)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

# median sync
energy_error = []
energy_stds = []
for j, iter_gap in enumerate(iter_gaps):
    energy_mean, energy_std, error, error_std = [], [], [], []
    for i, distance in enumerate(atom_dist):
        energy = []
        name = 'energy_32_{}_1.0_100_0_'.format(iter_gap)+str(distance)+'.npy'
        for seed in range(0, 5):
            energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/baseline', str(seed), name)))
            error_std.append(abs(energy[-1] - GE['LiH'][i]))
        energy_std.append(np.std(energy))
        energy_mean.append(np.mean(energy))
        error.append(abs(energy_mean[-1] - GE['LiH'][i]))
    energy_error.append(np.mean(error))
    energy_stds.append(np.std(error_std))
    ax[0].plot(atom_dist, energy_mean, color=np.array(color[j%len(color)])/255, marker=marker[j//len(marker)], linewidth=2.5)
ax[0].plot(atom_dist, GE['LiH'][:len(atom_dist)], color='k', linewidth=2.5)
ax[0].set_xticks(atom_dist)
ax[0].tick_params(labelsize=20)
ax[0].set_ylabel('Ground state energy (Ha)', font, fontsize=20)
ax[0].set_xlabel('Inter-atomic distance (Å)', font, fontsize=20)
# ax[0].legend(loc='best', fontsize=15, frameon=True, ncol=2)
ax[0].set_title('QUDIO', font, fontsize=20)
ax[0].grid(True)

for j, iter_gap in enumerate(iter_gaps):
    energy_mean, energy_std, error, error_std = [], [], [], []
    for i, distance in enumerate(atom_dist):
        energy = []
        name = 'energy_32_{}_1.0_100_0_'.format(iter_gap)+str(distance)+'.npy'
        for seed in range(0, 5):
            energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/shuffle', str(seed), name)))
            error_std.append(abs(energy[-1] - GE['LiH'][i]))
        energy_std.append(np.std(energy))
        energy_mean.append(np.mean(energy))
        error.append(abs(energy_mean[-1] - GE['LiH'][i]))
    energy_error.append(np.mean(error))
    energy_stds.append(np.std(error_std))
    ax[1].plot(atom_dist, energy_mean, label='W={}'.format(iter_gap), color=np.array(color[j%len(color)])/255, marker=marker[j//len(marker)], linewidth=2.5)
ax[1].plot(atom_dist, GE['LiH'][:len(atom_dist)], label='ExactEigensolver', color='k', linewidth=2.5)
ax[1].set_xticks(atom_dist)
ax[1].tick_params(labelsize=20)
# ax[1].set_ylabel('Ground state energy (Ha)', fontsize=20)
ax[1].set_xlabel('Inter-atomic distance (Å)', font, fontsize=20)
# ax[1].legend(loc='best', fontsize=20, frameon=True, ncol=2)

ax[1].set_title('Shuffle-QUDIO', font, fontsize=20)
ax[1].grid(True)

handles, labels = ax[1].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc=(0.3, 0.65), fontsize=20, frameon=True, ncol=3)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

plt.tight_layout()

# fig.legend(loc='best')
# plt.savefig('figure/app_vqe_training_q.png')
# plt.savefig('figure/app_vqe_training_loss.pdf')
# plt.savefig('figure/energy_surface_LiH.png')
# plt.savefig('figure/energy_surface_LiH.pdf')
plt.show()

plt.clf()
plt.close()

# mean of approximation error
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for i in range(2):
    if i == 0:
        ax.bar((np.arange(len(iter_gaps))+1)*2.5+i*1, energy_error[i*len(iter_gaps):(i+1)*len(iter_gaps)], width=1.0, fc=np.array(color[i])/255, label='QUDIO', edgecolor='k')
    else:
        ax.bar((np.arange(len(iter_gaps))+1)*2.5+i*1, energy_error[i*len(iter_gaps):(i+1)*len(iter_gaps)], width=1.0, fc=np.array(color[i])/255, label='Shuffle-QUDIO', edgecolor='k', tick_label=iter_gaps)
    plt.plot((np.arange(len(iter_gaps))+1)*2.5+i*1, energy_error[i*len(iter_gaps):(i+1)*len(iter_gaps)], color=np.array(color[i])/255, linestyle='--', linewidth=2.5)
    for x, y in zip((np.arange(len(iter_gaps))+1)*2.5+i*1, energy_error[i*len(iter_gaps):(i+1)*len(iter_gaps)]):
        ax.text(x-0.5, y+0.01, str(round(y, 3)), fontsize=15)
ax.tick_params(labelsize=20)
leg = ax.legend(loc='best', fontsize=20)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')
ax.set_xlabel('Number of local iterations (W)', font, fontsize=20)
ax.set_ylabel(r'$\overline{Err}$', font, fontsize=20)
ax.grid(True, axis='y')
plt.tight_layout()
# plt.savefig('figure/app_error_LiH.png')
# plt.savefig('figure/app_error_LiH.pdf')
plt.show()

plt.clf()
plt.close()

# STD of approximation error
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for i in range(2):
    if i == 0:
        ax.bar((np.arange(len(iter_gaps))+1)*2.5+i*1.0, energy_stds[i*len(iter_gaps):(i+1)*len(iter_gaps)], width=1.0, fc=np.array(color[i])/255, label='QUDIO', edgecolor='k')
    else:
        ax.bar((np.arange(len(iter_gaps))+1)*2.5+i*1.0, energy_stds[i*len(iter_gaps):(i+1)*len(iter_gaps)], width=1.0, fc=np.array(color[i])/255, label='Shuffle-QUDIO', edgecolor='k', tick_label=iter_gaps)
    plt.plot((np.arange(len(iter_gaps))+1)*2.5+i*1.0, energy_stds[i*len(iter_gaps):(i+1)*len(iter_gaps)], color=np.array(color[i])/255, linestyle='--', linewidth=2.5)
    for x, y in zip((np.arange(len(iter_gaps))+1)*2.5+i*1.0, energy_stds[i*len(iter_gaps):(i+1)*len(iter_gaps)]):
        ax.text(x-0.5, y+0.01, str(round(y, 3)), fontsize=15)
ax.tick_params(labelsize=20)
leg = ax.legend(loc='best', fontsize=20)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')
ax.set_xlabel('Number of local iterations (W)', font, fontsize=20)
ax.set_ylabel(r'$\delta(Err)$', font, fontsize=20)
ax.grid(True, axis='y')
plt.tight_layout()
# plt.savefig('figure/std_app_error_LiH.png')
# plt.savefig('figure/std_app_error_LiH.pdf')
plt.show()
plt.clf()
plt.close()

# stochastic VQE
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7.0, 5.5)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

# median sync
batch_size = [2, 4, 8, 16, 32, 64, 128]
energy_error = []
energy_stds = []
for j, batch in enumerate(batch_size):
    energy_mean, energy_std, error, error_std = [], [], [], []
    for i, distance in enumerate(atom_dist):
        energy = []
        name = 'energy_1_200_1.0_{}_0_'.format(batch)+str(distance)+'.npy'
        for seed in range(0, 1):
            energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/shuffle', str(seed), name)))
            error_std.append(abs(energy[-1] - GE['LiH'][i]))
        energy_std.append(np.std(energy))
        energy_mean.append(np.mean(energy))
        error.append(abs(energy_mean[-1] - GE['LiH'][i]))
    energy_error.append(np.mean(error))
    energy_stds.append(np.std(error_std))
    ax.plot(atom_dist, energy_mean, label='Batch={}'.format(batch), color=np.array(color[j%len(color)])/255, marker=marker[j//len(marker)])
ax.plot(atom_dist, GE['LiH'][:len(atom_dist)], label='ExactEigensolver', color='k', linewidth=2.0)
ax.set_xticks(atom_dist)
ax.tick_params(labelsize=20)
ax.set_ylabel('Ground state energy (Ha)', fontsize=20)
ax.set_xlabel('Inter-atomic distance (Å)', fontsize=20)
ax.legend(loc='best', fontsize=15, frameon=True, ncol=2)
ax.set_title('Stochastic VQE', fontsize=20)
plt.tight_layout()
# plt.savefig('figure/energy_surface_LiH_SVQE.png')
# plt.savefig('figure/energy_surface_LiH_SVQE.pdf')
plt.show()
plt.clf()
plt.close()

# non-communicating VQE
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(7.0, 5.5)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

# median sync
batch_size = [2, 4, 8, 16, 32]
energy_error = []
energy_stds = []
for j, batch in enumerate(batch_size):
    energy_mean, energy_std, error, error_std = [], [], [], []
    for i, distance in enumerate(atom_dist):
        energy = []
        name = 'energy_{}_200_1.0_100_0_'.format(batch)+str(distance)+'.npy'
        for seed in range(0, 1):
            energy.append(np.load(os.path.join('logs/vqe/ideal/LiH/shuffle', str(seed), name)))
            error_std.append(abs(energy[-1] - GE['LiH'][i]))
        energy_std.append(np.std(energy))
        energy_mean.append(np.mean(energy))
        error.append(abs(energy_mean[-1] - GE['LiH'][i]))
    energy_error.append(np.mean(error))
    energy_stds.append(np.std(error_std))
    ax.plot(atom_dist, energy_mean, label='K={}'.format(batch), color=np.array(color[j%len(color)])/255, marker=marker[j//len(marker)])
ax.plot(atom_dist, GE['LiH'][:len(atom_dist)], label='ExactEigensolver', color='k', linewidth=2.0)
ax.set_xticks(atom_dist)
ax.tick_params(labelsize=20)
ax.set_ylabel('Ground state energy (Ha)', font, fontsize=20)
ax.set_xlabel('Inter-atomic distance (Å)', font, fontsize=20)
leg = ax.legend(loc='best', fontsize=15, frameon=True, ncol=2)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')
ax.set_title('W=T', fontsize=20)
ax.grid(True)
plt.tight_layout()
# plt.savefig('figure/app_error_LiH_wt.png')
# plt.savefig('figure/app_error_LiH_wt.pdf')
plt.show()
plt.clf()
plt.close()