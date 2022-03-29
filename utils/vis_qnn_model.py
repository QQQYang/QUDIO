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
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

log_dir = 'logs/qnn_fix_batch_weighted/'

optimal_model = torch.load(os.path.join(log_dir, 'model_32_1_0.0001_100_0.pth'))['param'].numpy()
dists = []
losses = []
for i in range(400):
    models = []
    for worker in range(32):
        name = 'model_32_1_0.0001_100_{}_{}_0.pth'.format(worker, i+1)
        model = torch.load(os.path.join(log_dir, name))['param'].numpy()
        models.append(model)
    model_average = np.mean(np.stack(models), axis=0, keepdims=True)
    dist = np.linalg.norm(np.reshape(np.stack(models) - model_average, (len(models), -1)), axis=-1)

    dists.append(dist)
    losses.append(np.load(os.path.join(log_dir, 'loss_32_1_0.0001_100_{}_0.npy'.format(worker))))

# plt.imshow(np.array(dists), cmap='hot')
# plt.colorbar()
# plt.show()

sns.heatmap(np.transpose(np.array(dists)))
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Worker', fontsize=20)
plt.tight_layout()
plt.savefig('figure/diff_worker2average.png')
plt.show()

sns.heatmap(np.array(losses))
plt.show()