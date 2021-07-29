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

fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4)) # (12.8, 5.5)


# running time
ax.set_yscale('log', base=2)
ax.set_yticks(xticks)
ax.set_xscale('log', base=2)
ax.set_xticks(xticks)
ax.plot(xticks, xticks, color='k', label='linear speedup')
for i, h in enumerate([1, 2, 4, 8]):
    data = []
    base = np.mean(np.load(os.path.join('logs/vqe/noisy', 'time'+str(1)+'_'+str(h)+'_0.0001_100.npy')))
    for worker in [1, 2, 4, 8]:
        data.append(base/np.mean(np.load(os.path.join('logs/vqe/noisy', 'time'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))))
        if worker > 1 and h == 1:
            ax.text(worker-0.15*worker, data[-1]+0.1, str(round(data[-1], 3)), fontsize=20)
    ax.plot(xticks, data, label='W={}'.format(h), color=np.array(color[i])/255, marker=marker[i], linewidth=2.0)
    ax.tick_params(labelsize=20)
# ax.set_title('(a)', fontsize=20)
ax.set_xlabel('Number of local nodes (Q)', fontsize=20)
ax.set_ylabel('Speedup for time', fontsize=20)
ax.legend(loc='best', fontsize=20, frameon=True)

# training loss
# x = range(200)
# for i, h in enumerate([1, 2, 4, 8]):
#     worker = 8
#     acc_train = []
#     for subworker in range(worker):
#         name = 'loss_'+str(worker)+'_'+str(h)+'_0.0001_100_'+str(subworker)+'.npy'
#         acc_train.append(np.load(os.path.join('logs/vqe/noisy', name)))
#     data = np.sum(acc_train, axis=0)[::10]
#     ax[1].plot(x[::10], data, label='K={}'.format(h), color=color[i], marker=marker[i])
# ax[1].set_ylabel('Energy')
# # ax[1].set_ylim(0.8, 1.0)
# ax[1].set_xlabel('Epoch')
# ax[1].legend(loc='best')

# energy curve
# gt = [-0.60180371, -1.05515979, -1.13618945, -1.12056028, -1.07919294, -1.03518627, -0.99814935, -0.97142669, -0.95433885, -0.94437468]
# ax.plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], gt, '--', label='ExactEigensolver', color='k', linewidth=2.0)
# x = range(200)
# for i, h in enumerate([1, 2, 4, 8]):
#     worker = 8
#     energy = []
#     energy_std = []
#     for distance in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]:
#         data = []
#         for seed in range(0, 5):
#             if seed == 2 or seed == 3:
#                 continue
#             acc_train = []
#             for subworker in range(worker):
#                 name = 'loss_'+str(worker)+'_'+str(h)+'_1.0_100_'+str(subworker)+'_'+str(distance)+'.npy'
#                 if not os.path.exists(os.path.join('logs/vqe/', str(seed), name)):
#                     print(os.path.join('logs/vqe/', str(seed), name))
#                     continue
#                 acc_train.append(np.load(os.path.join('logs/vqe/', str(seed), name)))
#             data.append(np.sum(acc_train, axis=0)[193])
#         energy.append(np.mean(data))
#         energy_std.append(np.std(data))
#     # ax[1].plot([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], energy, label='VQE: W={}'.format(h), color=color[i], marker=marker[i])
#     ax.errorbar([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1], energy, yerr=energy_std, label='VQE: W={}'.format(h), color=np.array(color[i])/255, marker=marker[i], capsize=4, linewidth=2.0)
#     ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
#     ax.tick_params(labelsize=20)
# # ax.set_title('(b)', fontsize=20)
# ax.set_ylabel('Ground state energy (Ha)', fontsize=20)
# # ax[1].set_ylim(0.8, 1.0)
# ax.set_xlabel('Inter-atomic distance (Å)', fontsize=20)
# ax.legend(loc='best', fontsize=20, frameon=True)
plt.tight_layout()

# fig.legend(loc='best')
# plt.savefig('figure/speedup_vqe.pdf')
plt.savefig('figure/speedup_vqe.png')
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