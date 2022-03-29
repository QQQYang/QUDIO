import matplotlib.pyplot as plt
import numpy as np
import os

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

xticks = [1, 2, 4, 8, 16]
workers = [1, 2, 4, 8, 16, 32]

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(13, 6.4))

# # time
for i in range(2):
    ax[i].set_xscale('log', base=2)
    ax[i].set_yscale('log', base=2)
log_dir_ideal = 'logs/qnn_fix_batch/'

## noiseless
print('Ideal:\n')
lines, line_labels = [], []

for i, h in enumerate(xticks):
    bases, base_accs = [], []
    for seed in range(0,3):
        base = np.load(os.path.join(log_dir_ideal, 'time'+str(1)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed)))[0]
        bases.append(base)
        base_acc = np.load(os.path.join(log_dir_ideal, 'acc_test_'+str(1)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed)))
        find = False
        for k in range(len(base_acc)):
            if base_acc[k] >= 0.98:
                find = True
                base_acc = k
                base_accs.append(k+1)
                break
        if not find:
            base_accs.append(len(base_acc)+1)
    bases = np.mean(bases)
    base_accs = np.mean(base_accs)

    data, data_time = [], []
    for worker in workers:
        ts, accs = [], []
        for seed in range(0,3):
            t = np.load(os.path.join(log_dir_ideal, 'time'+str(worker)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed)))[0]
            ts.append(t)
            if not os.path.exists(os.path.join(log_dir_ideal, 'acc_test_'+str(worker)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed))):
                print(os.path.join(log_dir_ideal, 'acc_test_'+str(worker)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed)))
            else:
                acc = np.load(os.path.join(log_dir_ideal, 'acc_test_'+str(worker)+'_'+str(h)+'_0.0_0_{}.npy'.format(seed)))
                find = False
                for k in range(len(acc)):
                    if acc[k] >= 0.98:
                        find = True
                        acc = k
                        accs.append(k+1)
                        break
                if not find:
                    accs.append(len(acc)+1)
        ts = np.mean(ts)
        accs = np.mean(accs)
        data.append(bases*base_accs/(ts*accs))
        data_time.append(bases/(ts))
        # if worker == 32:
        print('h={}, Q={}, t={}, bases={}'.format(h, worker, ts, bases))
    ax[1].plot(workers, data, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12)

    ax[0].plot(workers, data, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12)

ax[0].plot(workers, workers, color='k', linewidth=2.5)#
ax[1].plot(workers, workers, color='k', linewidth=2.5)#

## noisy
log_dir = 'logs/basis00/e200_'
lines, line_labels = [], []
print('Noisy:\n')
for i, h in enumerate(xticks):
    data, data_time = [], []
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
    for worker in workers:
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
        data_time.append(bases/(ts))
        if worker == 32:
            print('h={}, Q=32, t={}, bases={}'.format(h, ts*accs/100, bases))
        if h == 16 and worker == 4:
            ax[1].text(worker-1.5, data[-1]+1, str(round(data[-1], 3)), fontsize=20, bbox=dict(boxstyle='square,pad=0.3', ec='black', facecolor=np.array(color[1])/255))
            ax[0].text(worker-1.5, data_time[-1]+1, str(round(data_time[-1], 3)), fontsize=20, bbox=dict(boxstyle='square,pad=0.3', ec='black', facecolor=np.array(color[1])/255))
    # ax[0].plot(xticks, data, label='K={}'.format(h), color=color[i], marker=marker[i])
    line, = ax[0].plot(workers, data_time, linestyle='-', label='W={}'.format(h), color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, alpha=1.0, markersize=12)
    lines.append(line)
    line_labels.append('W={}'.format(h))

    ax[1].plot(workers, data, linestyle='-', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, alpha=1.0, markersize=12)

for i in range(2):
    ax[i].tick_params(labelsize=20)
    # ax.set_title('(a)', fontsize=20)
    ax[i].set_yticks(workers)
    ax[i].set_xticks(workers)
    font = {
        'family':'Times New Roman',
        # 'serif':'Times New Roman',
        'style':'normal',
        'weight':'bold', #or 'bold'
    }
    ax[i].set_xlabel('Number of local nodes (Q)', font, fontsize=20)
    ax[i].grid(True)
    ax[i].spines['bottom'].set_linewidth(2)
    ax[i].spines['left'].set_linewidth(2)
    ax[i].spines['right'].set_linewidth(2)
    ax[i].spines['top'].set_linewidth(2)
ax[0].set_title('Speedup to time', font, fontsize=20)
ax[1].set_title('Speedup to accuracy', font, fontsize=20)
# ax.legend(loc='best', ncol=2, fontsize=20, frameon=True, columnspacing=0.5)
# leg = plt.legend(handles=lines, ncol=1, fontsize=20, frameon=True, columnspacing=0.5, bbox_to_anchor=(-1, -1))
# leg.get_frame().set_linewidth(2)
# leg.get_frame().set_edgecolor('k')

# def export_legend(legend, filename="figure/qnn_legend.png"):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)
# export_legend(leg)

plt.tight_layout()
# plt.savefig('figure/time.pdf')
# plt.savefig('figure/speedup.png')
# plt.savefig('figure/time.png')
# plt.savefig('figure/qnn_time_speedup.png')
plt.show()