import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd']
marker = ["o", "s", "^", '+']
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

xticks = [1, 2, 4, 8]

fig, ax = plt.subplots(1, 3, figsize=(20.4, 5.5))

# # time
ax[0].set_xscale('log', base=2)
for i, h in enumerate([1, 2, 4, 8]):
    data = []
    for worker in [1, 2, 4, 8]:
        data.append(np.load(os.path.join('logs/basis00/', 'time'+str(worker)+'_'+str(h)+'_0.0001_100.npy')))
    # ax[0].plot(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
    ax[0].plot(xticks, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax[0].tick_params(labelsize=20)
ax[0].set_title('(a)', fontsize=20)
ax[0].set_xlabel('Number of workers', fontsize=20)
ax[0].set_ylabel('Time /s', fontsize=20)
ax[0].legend(loc='best', fontsize=15)
ax[0].grid(True)
# plt.tight_layout()
# plt.savefig('figure/running_time.pdf')

# # training loss
# x = range(100)
# for i, h in enumerate([1, 2, 4, 8]):
#     worker = 8
#     acc_train = []
#     for subworker in range(worker):
#         name = 'loss_'+str(worker)+'_'+str(h)+'_0.0001_100_'+str(subworker)+'.npy'
#         acc_train.append(np.load(os.path.join('logs/basis00/', name)))
#     data = np.mean(acc_train, axis=0)
#     data = data[::5].tolist() + [data[-1]]
#     ax[1].plot(list(x[::5])+[x[-1]], data, label='K={}'.format(h), color=color[i], marker=marker[i])
#     ax[1].tick_params(labelsize=20)
# ax[1].set_title('(b)', fontsize=20)
# ax[1].set_ylabel('Train loss', fontsize=20)
# ax[1].set_xlabel('Epoch', fontsize=20)
# ax[1].legend(loc='best', fontsize=15)
# plt.tight_layout()
# plt.savefig('figure/loss2.pdf')

# # test accuracy
ax1_twin = ax[1].twinx()
x = range(100)
for i, h in enumerate([1, 2, 4, 8]):
    worker = 8
    name = 'acc_test_'+str(worker)+'_'+str(h)+'_0.0001_100.npy'
    data = np.load(os.path.join('logs/basis00/e200', name))
    x = range(50)
    x = list(x[::2])+[x[-1]]
    data = data[::16].tolist() + [data[-1]]
    ax[1].plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax[1].tick_params(labelsize=20)

    acc_train = []
    for subworker in range(worker):
        name = 'loss_'+str(worker)+'_'+str(h)+'_0.0001_100_'+str(subworker)+'.npy'
        acc_train.append(np.load(os.path.join('logs/basis00/e200', name)))
    data = np.mean(acc_train, axis=0)
    data = data[::4].tolist() + [data[-1]]
    x = range(50)
    x = list(x[::2])+[x[-1]]
    ax1_twin.plot(x, data, '--', label='W={}'.format(h), color=color[i], marker=marker[i])
    ax1_twin.tick_params(labelsize=20)
    ax1_twin.set_ylabel('Train loss (----)', fontsize=20)

ax[1].set_title('(b)', fontsize=20)
ax[1].set_ylabel('Test accuracy (——)', fontsize=20)
ax[1].set_xlabel('Global step', fontsize=20)
ax[1].legend(loc=(0.7, 0.5), fontsize=15)
ax[1].grid(True)

# loss
# cnt = 0
# for i, p in enumerate([0.0001, 0.1]):
#     for m in [5, 100]:
#         worker = 8
#         data = []
#         for subworker in range(worker):
#             data.append(np.load(os.path.join('logs/basis00/figure1', 'loss_8_8_'+str(p)+'_'+str(m)+'_'+str(subworker)+'.npy')))
#         x = range(len(data[0]))
#         data = np.mean(data, axis=0)[::5]
#         ax.plot(x[::5], data, label='p={}, M={}'.format(p, m), color=color[cnt], marker=marker[cnt])
#         ax.tick_params(labelsize=20)
#         cnt += 1
# ax.set_title('Q=8, W=8', fontsize=20)
# ax.set_xlabel('Epoch', fontsize=20)
# ax.set_ylabel('Loss', fontsize=20)
# ax.legend(loc='best', fontsize=15)
# plt.tight_layout()
# plt.savefig('figure/loss1.pdf')

# loss
cnt = 0
for i, p in enumerate([0.0001, 0.1]):
    for m in [5, 100]:
        worker = 8
        data = []
        for subworker in range(worker):
            grad = np.load(os.path.join('logs/basis00/figure1', 'grad_8_8_'+str(p)+'_'+str(m)+'_'+str(subworker)+'.npy'))
            index = []
            for i in range(100):
                index = index + (np.array([7, 15, 23, 24])+25*i).tolist()
            grad = grad[index]
            data.append(grad)
        x = range(len(data[0]))
        data = np.mean(data, axis=0)
        data = data[::8].tolist() + [data[-1]]
        ax[2].plot(data, label='p={}, M={}'.format(p, m), color=color[cnt], marker=marker_grad[cnt]) # list(x[::10])+[x[-1]], 
        ax[2].tick_params(labelsize=20)
        cnt += 1
ax[2].set_title('(c)', fontsize=20)
ax[2].set_xlabel('Global step', fontsize=20)
ax[2].set_ylabel('Grad L2 norm', fontsize=20)
ax[2].legend(loc='best', fontsize=15)
ax[2].grid(True)
plt.tight_layout()
plt.savefig('figure/qnn.pdf')

# fig.legend(loc='best')
plt.show()