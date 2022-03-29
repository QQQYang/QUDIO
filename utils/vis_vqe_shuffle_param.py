import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189)]
marker = ["o", "s", '+', "v", "^", "<", ">"]
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]

# plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'bold', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

fig, ax = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4,4,0.4]), figsize=(13, 6.4))

data = []
for m, method in enumerate(['baseline', 'shuffle']):
    log_dir = 'logs\\vqe\ideal\loss_landscape\\{}_param_average\\0'.format(method)
    dists = []
    losses = []
    for i in range(0, 200, 32):
        models = []
        for worker in range(16):
            name = 'model_16_32_0.0_0_{}_0.5_{}.pth'.format(worker, i)
            model = torch.load(os.path.join(log_dir, name))['param'].numpy()
            models.append(model)
        model_average = np.mean(np.stack(models), axis=0, keepdims=True)
        dist = np.linalg.norm(np.reshape(np.stack(models) - model_average, (len(models), -1)), axis=-1)

        dists.append(dist)
        # losses.append(np.load(os.path.join(log_dir, 'loss_32_1_0.0001_100_{}_0.npy'.format(worker))))

    # plt.imshow(np.transpose(np.array(dists)), cmap='RdYlGn_r')
    # plt.colorbar()
    # plt.show()
    data.append(dists)
    ax[m].tick_params(labelsize=20)

vmin = np.min(np.array(data))
vmax = np.max(np.array(data))
sns.heatmap(np.transpose(np.array(data[0])), cmap='RdYlGn_r', cbar=False, ax=ax[0], vmin=vmin)
sns.heatmap(np.transpose(np.array(data[1])), cmap='RdYlGn_r', yticklabels=False, cbar=False, ax=ax[1], vmax=vmax)
fig.colorbar(ax[1].collections[0], cax=ax[2])

font = {
        'family':'Times New Roman',
        # 'serif':'Times New Roman',
        'style':'normal',
        'weight':'bold', #or 'bold'
    }

ax[0].set_xticklabels(np.arange(0, 200, 32))
ax[1].set_xticklabels(np.arange(0, 200, 32))
ax[0].set_title('QUDIO', font, fontsize=20)
ax[1].set_title('Shuffle-QUDIO', font, fontsize=20)
ax[0].set_xlabel('Iteration', font, fontsize=20)
ax[1].set_xlabel('Iteration', font, fontsize=20)
ax[0].set_ylabel('Worker', font, fontsize=20)
ax[2].tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('figure/diff_worker2average.png')
plt.savefig('figure/diff_worker2average.pdf')
plt.show()

# sns.heatmap(np.array(losses))
# plt.show()