import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
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
    for j, w in enumerate([1, 2, 4, 8, 16]):
        data = []
        for seed in range(5):
            name = 'acc_test_'+str(w)+'_'+str(h)+'_0.0001_100_{}.npy'.format(seed)
            if not os.path.exists(os.path.join('logs/qnn_fix_batch_weighted/', name)):
                print(os.path.join('logs/qnn_fix_batch_weighted', name))
            else:
                data.append(np.load(os.path.join('logs/qnn_fix_batch_weighted/', name))[-1])
        acc.append(data)
        
        if w == 4:
            box = ax.boxplot(data, positions=[(i+1)*2+j*0.3], labels=[str(h)], patch_artist=True, widths=0.3, boxprops=dict(facecolor=color[j], linewidth=2), medianprops=dict(color='k', linewidth=2), flierprops=dict(markerfacecolor=color[j], marker='D', linewidth=2), whiskerprops=dict(linewidth=2))
        else:
            box = ax.boxplot(data, positions=[(i+1)*2+j*0.3], labels=[''], patch_artist=True, widths=0.3, boxprops=dict(facecolor=color[j], linewidth=2), medianprops=dict(color='k', linewidth=2), flierprops=dict(markerfacecolor=color[j], marker='D', linewidth=2), whiskerprops=dict(linewidth=2))
        if save:
            boxes.append(box['boxes'][0])
            boxes_label.append('Q={}'.format(w))
    # ax.plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax.tick_params(labelsize=20)
    save = False

# bplot = ax.boxplot(acc, labels=[str(h) for h in xticks], patch_artist=True)
# for i,patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color[i])
font = {
    'family':'Times New Roman',
    # 'serif':'Times New Roman',
    'style':'normal',
    'weight':'bold', #or 'bold'
}
ax.set_ylabel('Test accuracy', font, fontsize=20)
ax.set_xlabel('Number of local iterations (W)', font, fontsize=20)
leg = ax.legend(boxes, boxes_label, fontsize=20, loc='best')
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

ax.grid(True)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.tight_layout()
# plt.savefig('figure/acc.pdf')
plt.savefig('figure/acc_weighted.pdf')
plt.show()