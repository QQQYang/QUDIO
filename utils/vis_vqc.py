import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker = ["o", "s", "^", '+']

xticks = [1, 2, 4, 8]

fig, ax = plt.subplots(1, 3, figsize=(12.8, 5.5))

# running time
ax[0].set_xscale('log', base=2)
for i, h in enumerate([1, 2, 4, 8]):
    data = []
    for worker in [1, 2, 4, 8]:
        data.append(np.load(os.path.join('logs/basis00/', 'time'+str(worker)+'_'+str(h)+'_0.05_5.npy')))
    # ax[0].plot(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
    ax[0].scatter(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
ax[0].set_xlabel('Number of workers')
ax[0].set_ylabel('Time /s')
ax[0].legend(loc='best')

# train accuracy
ax[1].set_xscale('log', base=2)
for i, h in enumerate([1, 2, 4, 8]):
    data = []
    for worker in [1, 2, 4, 8]:
        acc_train = []
        for subworker in range(worker):
            name = 'acc_train_'+str(worker)+'_'+str(h)+'_0.05_5_'+str(subworker)+'.txt'
            acc_train.append(np.loadtxt(os.path.join('logs/basis00/', name)))
        data.append(np.mean(acc_train))
    # ax[1].plot(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
    ax[1].scatter(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
ax[1].set_ylabel('Train accuracy')
# ax[1].set_ylim(0.8, 1.0)
ax[1].set_xlabel('Number of workers')
ax[1].legend(loc='best')

# training loss
x = range(100)
for i, h in enumerate([1, 2, 4, 8]):
    worker = 8
    acc_train = []
    for subworker in range(worker):
        name = 'loss_'+str(worker)+'_'+str(h)+'_0.05_5_'+str(subworker)+'.npy'
        acc_train.append(np.load(os.path.join('logs/basis00/', name)))
    data = np.mean(acc_train, axis=0)[::10]
    ax[2].plot(x[::10], data, label='K={}'.format(h), color=color[i], marker=marker[i])
ax[2].set_ylabel('Train loss')
# ax[1].set_ylim(0.8, 1.0)
ax[2].set_xlabel('Epoch')
ax[2].legend(loc='best')
plt.tight_layout()

# fig.legend(loc='best')
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