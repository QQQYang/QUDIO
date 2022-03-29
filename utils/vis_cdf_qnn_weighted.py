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
data = []
data_vanilla = []
for i, h in enumerate(xticks):
    for j, w in enumerate([1, 2, 4, 8, 16]):
        for seed in range(5):
            name = 'acc_test_'+str(w)+'_'+str(h)+'_0.0001_100_{}.npy'.format(seed)
            if not os.path.exists(os.path.join('logs/qnn_fix_batch_weighted/', name)):
                print(os.path.join('logs/qnn_fix_batch_weighted', name))
            else:
                data.append(np.load(os.path.join('logs/qnn_fix_batch_weighted/', name))[-1])

            name = 'acc_test_'+str(w)+'_'+str(h)+'_0.0001_100.npy'
            data_vanilla.append(np.load(os.path.join('logs/basis00/e200_'+str(seed), name))[-1])
    # ax.plot(x, data, label='W={}'.format(h), color=color[i], marker=marker[i])
    ax.tick_params(labelsize=20)

ax.hist(data, bins=100, density=True, color=color[0], cumulative=True, histtype='step', label='WS-QUDIO', linewidth=2.5)
ax.hist(data_vanilla, bins=100, density=True, color=color[1], cumulative=True, histtype='step', label='QUDIO', linewidth=2.5)

# bplot = ax.boxplot(acc, labels=[str(h) for h in xticks], patch_artist=True)
# for i,patch in enumerate(bplot['boxes']):
#     patch.set_facecolor(color[i])
font = {
    'family':'Times New Roman',
    # 'serif':'Times New Roman',
    'style':'normal',
    'weight':'bold', #or 'bold'
}
ax.set_ylabel('Probability', font, fontsize=20)
ax.set_xlabel('Test accuracy', font, fontsize=20)
leg = ax.legend(fontsize=20, loc='upper left')
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

ax.grid(True)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.tight_layout()
# plt.savefig('figure/acc.pdf')
# plt.savefig('figure/acc_cdf.pdf')
plt.show()