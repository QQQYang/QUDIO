import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
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

xticks = [4, 8, 16, 32]

fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4)) # (14.4, 6.4)

boxes = []
boxes_label = []
save = True

# ax.set_xscale('log', base=2)
# ax.set_xticks(xticks)

x = range(100)
worker = 32
acc = []
for i, h in enumerate(xticks):
    for j, w in enumerate([1, 2, 4, 8, 16, 32]):
        data = []
        name = 'acc_test_'+str(w)+'_'+str(h)+'_0.0001_100.npy'
        for seed in range(5):
            data.append(np.load(os.path.join('logs/basis00/e200_'+str(seed), name))[-1])
        acc.append(data)
        if w == 4:
            box = ax.boxplot(data, positions=[(i+1)*2+j*0.3], labels=[str(h)], patch_artist=True, widths=0.3, boxprops=dict(facecolor=color[j]), medianprops=dict(color='k'), flierprops=dict(markerfacecolor=color[j], marker='D'))
        else:
            box = ax.boxplot(data, positions=[(i+1)*2+j*0.3], labels=[''], patch_artist=True, widths=0.3, boxprops=dict(facecolor=color[j]), medianprops=dict(color='k'), flierprops=dict(markerfacecolor=color[j], marker='D'))
        if save:
            boxes.append(box['boxes'][0])
            boxes_label.append('Q={}'.format(w))
    # ax.plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax.tick_params(labelsize=20)
    save = False

# bplot = ax.boxplot(acc, labels=[str(h) for h in xticks], patch_artist=True)
# for i,patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color[i])
ax.set_ylabel('Test accuracy', fontsize=20)
ax.set_xlabel('Number of local iterations (W)', fontsize=20)
ax.legend(boxes, boxes_label, fontsize=20, loc='upper left')
ax.grid(True)
plt.tight_layout()
# plt.savefig('figure/acc.pdf')
plt.savefig('figure/acc.png')
plt.show()