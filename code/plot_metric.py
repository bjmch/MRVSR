import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def load_metrics(pickle_path):
    with open(pickle_path, 'rb') as f:
        metrics = pickle.load(f)
        return metrics


metric = 'psnrs.pkl'

length = 377

metrics_rfs3 = np.zeros(length)
metrics_mrvsr = np.zeros(length)
metrics_rlsp = np.zeros(length)

cwd = os.getcwd()

for seq in ['Sequence1', 'Sequence2', 'Sequence3']:
    metrics_mrvsr += load_metrics(
        os.path.join(cwd, '../images/mrvsr/QuasiStaticVideoSet/', seq, metric)
    )
    metrics_rfs3 += load_metrics(
        os.path.join(cwd, '../images/rfs3/QuasiStaticVideoSet/', seq, metric)
    )
    metrics_rlsp += load_metrics(
        os.path.join(cwd, '../images/rlsp/QuasiStaticVideoSet/', seq, metric)
    )

metrics_mrvsr /= 3
metrics_rfs3 /= 3
metrics_rlsp /= 3
# fig, ax = plt.subplots(figsize=(20, 20))
fig, ax = plt.subplots(figsize=(16, 9))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.xlabel('Frame number', fontsize=20)

if metric == 'psnrs.pkl':
    Y = 'Frame PSNR'
elif metric == 'ssims.pkl':
    Y = 'Frame SSIM'

plt.ylabel(Y, fontsize=20)

#
ref = metrics_rfs3
X = np.arange(1, length + 1)
#
plt.plot(X, metrics_rlsp - np.array(ref), color='r', label='RLSP', linewidth=3)
plt.plot(X, metrics_rfs3 - ref, color='orange', label='RFS3', linewidth=4)
plt.plot(X, metrics_mrvsr - ref, color='k', label='MRVSR', linewidth=3)
#

plt.xlim(1, length)
plt.ylim(-2.7, 1.7)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
#
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#
ax.legend(loc='upper right', fontsize=16.5, ncol=2)
plt.xlabel('Frame number', fontsize=20)

Y = '\u0394' + ' PSNR'
plt.ylabel(Y, fontsize=20)

plt.tight_layout()

plt.show()
