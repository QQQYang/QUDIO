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

# fig, axs = plt.subplots(ncols=2, nrows=2, sharey=False, figsize=(10, 9))

# log_dir = 'logs/qnn_fix_batch_weighted/'

# workers = [1, 2, 4, 8, 16, 32]
# interval = [1, 2, 4, 8, 16, 32]

# for layer in range(4):
#     params = []
#     for worker in workers:
#         param_worker = []
#         for i in interval:
#             name = 'model_{}_{}_0.0001_100_0.pth'.format(worker, i)
#             model = torch.load(os.path.join(log_dir, name))['param'].numpy()
#             param_worker.append(model[layer, -1, -1])
#         params.append(param_worker)

#     ax = sns.heatmap(np.array(params), ax=axs[layer//2][layer%2])
#     ax.figure.axes[-1].yaxis.label.set_size(20)

#     ax.tick_params(labelsize=20)
#     ax.set_xticklabels(interval)
#     ax.set_yticklabels(workers)

#     if layer//2 == 1:
#         ax.set_xlabel('Number of local iterations (W)', fontsize=20)
#     ax.set_ylabel('Number of local nodes (Q)', fontsize=20)
#     if layer == 0:
#         ax.set_title(r'$\theta_{21}$', fontsize=20)
#     elif layer == 1:
#         ax.set_title(r'$\theta_{22}$', fontsize=20)
#     elif layer == 2:
#         ax.set_title(r'$\theta_{23}$', fontsize=20)
#     else:
#         ax.set_title(r'$\theta_{24}$', fontsize=20)

fig, axs = plt.subplots(ncols=1, nrows=1, sharey=False, figsize=(20, 10))

log_dir = 'logs/qnn_fix_batch_weighted/'

workers = [1, 2, 4, 8, 16, 32]
interval = [1, 2, 4, 8, 16]

params = []
yticklables = []
for worker in workers:
    for i in interval:
        name = 'model_{}_{}_0.0001_100_0.pth'.format(worker, i)
        model = torch.load(os.path.join(log_dir, name))['param'].numpy()
        params.append(model.flatten())
        yticklables.append('Q={},W={}'.format(worker, i))

ax = sns.heatmap(np.array(params), ax=axs, cbar=False, xticklabels=8, yticklabels=6, cmap='RdYlGn_r')#, cmap="YlGnBu") # yticklabels=yticklables
cb=ax.figure.colorbar(ax.collections[0])
cb.ax.tick_params(labelsize=25)

ax.tick_params(labelsize=30)
# ax.set_xticklabels(interval)
# ax.set_yticklabels(workers)

font = {
    'family':'Times New Roman',
    # 'serif':'Times New Roman',
    'style':'normal',
    'weight':'bold', #or 'bold'
}
ax.set_xlabel('Parameter index', font, fontsize=30)
ax.set_ylabel('Setting index', font, fontsize=30)
plt.tight_layout()
plt.savefig('figure/qnn_param.png')
plt.show()