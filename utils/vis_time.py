import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189)]
marker = ["o", "s", '+', "v", "^", "<", ">"]
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

xticks = [1, 2, 4, 8, 16, 32]

fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))

# # time
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
log_dir = 'logs/basis00/e200_'
log_dir_ideal = 'logs/ideal/e200_'
print('Noisy:\n')
for i, h in enumerate(xticks):
    data = []
    bases, base_accs = [], []
    for seed in range(5):
        base = np.load(os.path.join(log_dir+str(seed), 'time'+str(1)+'_'+str(h)+'_0.0001_100.npy'))[0]
        bases.append(base)
        base_acc = np.load(os.path.join(log_dir+str(seed), 'acc_test_'+str(1)+'_'+str(h)+'_0.0001_100.npy'))
        for k in range(len(base_acc)):
            if base_acc[k] >= 0.9:
                base_acc = k
                base_accs.append(k)
                break
    bases = np.mean(bases)
    base_accs = np.mean(base_accs)
    for worker in xticks:
        ts, accs = [], []
        for seed in range(5):
            t = np.load(os.path.join(log_dir+str(seed), 'time'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))[0]
            ts.append(t)
            acc = np.load(os.path.join(log_dir+str(seed), 'acc_test_'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))
            for k in range(len(acc)):
                if acc[k] >= 0.9:
                    acc = k
                    accs.append(k)
                    break
        ts = np.mean(ts)
        accs = np.mean(accs)
        data.append(bases*base_accs/(ts*accs))
        if worker == 32:
            print('h={}, Q=32, t={}'.format(h, ts*accs/100))
        if h == 32 and worker == 4:
            ax.text(worker-0.5, data[-1]+1, str(round(data[-1], 3)), fontsize=20, bbox=dict(boxstyle='square,pad=0.3', ec='black'))
    # ax[0].plot(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
    ax.plot(xticks, data, label='W={} (N)'.format(h), color=np.array(color[i])/255, marker=marker[i], linewidth=2.0, alpha=0.5)

ax.plot(xticks, xticks, color='k', label='linear speedup')

## noiseless
print('Ideal:\n')
for i, h in enumerate(xticks):
    data = []
    bases, base_accs = [], []
    for seed in range(3, 4):
        base = np.load(os.path.join(log_dir_ideal+str(seed), 'time'+str(1)+'_'+str(h)+'_0.0001_100.npy'))[0]
        bases.append(base)
        base_acc = np.load(os.path.join(log_dir_ideal+str(seed), 'acc_test_'+str(1)+'_'+str(h)+'_0.0001_100.npy'))
        for k in range(len(base_acc)):
            if base_acc[k] >= 0.9:
                base_acc = k
                base_accs.append(k)
                break
    bases = np.mean(bases)
    base_accs = np.mean(base_accs)
    for worker in xticks:
        ts, accs = [], []
        for seed in range(5):
            t = np.load(os.path.join(log_dir_ideal+str(seed), 'time'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))[0]
            ts.append(t)
            if not os.path.exists(os.path.join(log_dir_ideal+str(seed), 'acc_test_'+str(worker)+'_'+str(h)+'_0.0001_100.npy')):
                print(os.path.join(log_dir_ideal+str(seed), 'acc_test_'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))
            else:
                acc = np.load(os.path.join(log_dir_ideal+str(seed), 'acc_test_'+str(worker)+'_'+str(h)+'_0.0001_100.npy'))
                for k in range(len(acc)):
                    if acc[k] >= 0.9:
                        acc = k
                        accs.append(k)
                        break
        ts = np.mean(ts)
        accs = np.mean(accs)
        data.append(bases*base_accs/(ts*accs))
        if worker == 32:
            print('h={}, Q=32, t={}'.format(h, ts*accs/100))
    ax.plot(xticks, data, '--', label='W={} (I)'.format(h), color=np.array(color[i])/255, marker=marker[i], linewidth=2.0)

ax.tick_params(labelsize=20)
# ax.set_title('(a)', fontsize=20)
ax.set_yticks(xticks)
ax.set_xticks(xticks)
ax.set_xlabel('Number of local nodes (Q)', fontsize=20)
ax.set_ylabel('Speedup to accuracy', fontsize=20)
ax.legend(loc='best', ncol=2, fontsize=20, frameon=True, columnspacing=0.5)
ax.grid(True)
plt.tight_layout()
# plt.savefig('figure/time.pdf')
# plt.savefig('figure/speedup.png')
plt.show()