import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52)]
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

fig, ax = plt.subplots(1, 1, figsize=(14.4, 4.8)) # (14.4, 6.4)

boxes = []
boxes_label = []
save = True

# ax.set_xscale('log', base=2)
# ax.set_xticks(xticks)
noise = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
x = range(100)
worker = 16
for j, w in enumerate([5, 20, 50, 100]):
    acc = []
    for i, h in enumerate(noise):
        data = []
        name = 'acc_test_'+str(worker)+'_'+str(2)+'_'+str(h/10000)+'_'+str(w)+'.npy'
        acc.append(np.load(os.path.join('logs/noisy/e200_4', name))[-1])
    if w==50:
        box = ax.bar((np.arange(len(noise))+1)*2.5+j*0.55, acc, tick_label=np.array(noise)/10000, width=0.5, fc=np.array(color[j])/255, label='K={}'.format(w), edgecolor='k')
    else:
        box = ax.bar((np.arange(len(noise))+1)*2.5+j*0.55, acc, width=0.5, fc=np.array(color[j])/255, label='K={}'.format(w), edgecolor='k')
    for x, y in zip((np.arange(len(noise))+1)*2.5+j*0.55, acc):
        ax.text(x-0.45, y+0.008, str(round(y, 2)), fontsize=15)
    # ax.plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax.tick_params(labelsize=20)
    
    boxes.append(box[0])
    boxes_label.append('K={}'.format(w))

# bplot = ax.boxplot(acc, labels=[str(h) for h in xticks], patch_artist=True)
# for i,patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color[i])
ax.set_ylim([0.6, 1.05])
ax.set_ylabel('Test accuracy', fontsize=20)
ax.set_xlabel('Noise scale (p)', fontsize=20)
# ax.legend(boxes, boxes_label, fontsize=15, loc='upper right')
plt.legend(loc='best', fontsize=15, ncol=4)
ax.grid(True)
plt.tight_layout()
plt.savefig('figure/noise.pdf')
plt.show()