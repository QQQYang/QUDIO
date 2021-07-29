import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52)]
marker = ["o", "s", "^", '+']
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]

plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

xticks = [1, 2, 4, 8]

fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.5)) # (12.8, 5.5)
title = ['(a)', '(b)', '(c)']

# energy curve
gt = [-0.60180371, -1.05515979, -1.13618945, -1.12056028, -1.07919294, -1.03518627, -0.99814935, -0.97142669, -0.95433885, -0.94437468]
x = range(200)
for j, seed in enumerate([0, 1, 2]):
    worker = 8
    for i, h in enumerate([8]):
        # worker = 2
        energy = []
        for distance in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
            acc_train = []
            for subworker in range(worker):
                name = 'loss_'+str(worker)+'_'+str(h)+'_1.0_100_'+str(subworker)+'_'+str(distance)+'.npy'
                if not os.path.exists(os.path.join('logs/vqe/ideal/dist', str(seed), name)):
                    print(os.path.join('logs/vqe/ideal/dist', str(seed), name))
                    continue
                acc_train.append(np.load(os.path.join('logs/vqe/ideal/dist', str(seed), name)))
            energy.append(np.sum(acc_train, axis=0)[193])
        if seed == 2:
            label = 'seed=0, shuffle'
        else:
            label = 'seed={}'.format(seed)
        ax.plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], energy, label=label, color=np.array(color[j%len(color)])/255, marker=marker[j//len(marker)])
        ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
        ax.tick_params(labelsize=20)
    # ax[j].set_title(title[j+1]+' W={}'.format(worker), fontsize=20)
    ax.set_ylabel('Ground state energy (Ha)', fontsize=20)
    ax.set_xlabel('Inter-atomic distance (Å)', fontsize=20)
# median sync
energy = []
for distance in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
    name = 'energy_8_8_1.0_100_0_'+str(distance)+'.npy'
    energy.append(np.load(os.path.join('logs/vqe/ideal/dist/9', name)))
ax.plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], energy, label='seed=0, random', color=np.array(color[-1])/255, marker=marker[-1])

energy = []
for distance in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
    name = 'energy_8_8_1.0_100_0_'+str(distance)+'.npy'
    energy.append(np.load(os.path.join('logs/vqe/ideal/dist/8', name)))
ax.plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], energy, label='seed=0, median', color=np.array(color[-2])/255, marker=marker[-2])
ax.plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], gt, label='ExactEigensolver', color='k', linewidth=2.0)
ax.set_title('W=8, K=8', fontsize=20)
ax.legend(loc='best', fontsize=20, frameon=True)
plt.tight_layout()

# fig.legend(loc='best')
# plt.savefig('figure/app_vqe_training_q.png')
# plt.savefig('figure/app_vqe_training_loss.pdf')
# plt.savefig('figure/vqe.png')
plt.savefig('figure/DistVQE.pdf')
plt.show()

# p_new = []
# var = []
# var_new = []
# for p in range(100):
#     mean = 0.6561*p/100 + (1-0.6561)/2
#     p_new.append(mean)
#     var_new.append(mean*(1-mean))
#     var.append(p/100 * (1 - p/100))

# plt.plot(p_new, color='g')
# plt.plot(var, color='r')
# plt.plot(var_new, color='b')
# plt.show()