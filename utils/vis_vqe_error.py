import matplotlib.pyplot as plt
import numpy as np
import os

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52)]
marker = ["o", "s", '+', "v", "^", "<", ">"]
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

xticks = [4, 8, 16, 32]

# plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.4)) # (14.4, 6.4)

boxes = []
boxes_label = []
save = True

# ax.set_xscale('log', base=2)
# ax.set_xticks(xticks)
x = range(100)
worker = [2, 4, 8]
updates = [1, 2, 4, 8]
for j, h in enumerate(updates):
    acc = []
    for i, w in enumerate(worker):
        data = []
        for seed in range(0, 1):
            if seed == 2 or seed == 3:
                continue
            acc_train = []
            for subworker in range(w):
                name = 'loss_'+str(w)+'_'+str(h)+'_1.0_100_'+str(subworker)+'_'+str(0.3)+'.npy'
                if not os.path.exists(os.path.join('logs/vqe/', str(seed), name)):
                    print(os.path.join('logs/vqe/', str(seed), name))
                    continue
                acc_train.append(np.load(os.path.join('logs/vqe/', str(seed), name)))
            data.append(np.abs(np.sum(acc_train, axis=0)[193]+0.60180371))
        acc.append(np.mean(data))
    if h==4:
        box = ax.bar((np.arange(len(worker))+1)*2.5+j*0.55, acc, tick_label=np.array(worker), width=0.5, fc=np.array(color[j])/255, label='W={}'.format(h), edgecolor='k')
    else:
        box = ax.bar((np.arange(len(worker))+1)*2.5+j*0.55, acc, width=0.5, fc=np.array(color[j])/255, label='W={}'.format(h), edgecolor='k')
    # for x, y in zip((np.arange(len(worker))+1)*2.5+j*0.55, acc):
    #     ax.text(x-0.28, y+0.01, str(round(y, 3)), fontsize=17)
    # ax.plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax.tick_params(labelsize=25)
    
    boxes.append(box[0])
    boxes_label.append('W={}'.format(w))

# bplot = ax.boxplot(acc, labels=[str(h) for h in xticks], patch_artist=True)
# for i,patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color[i])
ax.set_ylabel('App.err', fontsize=25)
ax.set_xlabel('Number of local nodes (Q)', fontsize=25)
# ax.legend(boxes, boxes_label, fontsize=15, loc='upper right')
plt.legend(loc='best', fontsize=25)
ax.grid(True)
plt.tight_layout()
plt.savefig('figure/vqe_error.png')
plt.show()