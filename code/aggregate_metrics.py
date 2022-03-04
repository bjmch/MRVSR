import pickle
import numpy as np
import os

from options import parse_argument

args = parse_argument()


def load_metrics(pickle_path):
    with open(pickle_path, 'rb') as f:
        metrics = pickle.load(f)
        return metrics


# PSNR or SSIM
metric = args.metric + 's.pkl'

metrics = []

cwd = os.getcwd()

net = args.net

if args.data == 'QSV':
    dataset = 'QuasiStaticVideoSet'
    sequences = ['Sequence1', 'Sequence2', 'Sequence3', 'Sequence4']
elif args.data == 'Vid4':
    dataset = 'Vid4'
    sequences = ['calendar', 'city', 'foliage', 'walk']

for seq in sequences:
    metrics.append(load_metrics(
        os.path.join(cwd, '../images', net, dataset, seq, metric)))

if args.data == 'QSV':
    print('Mean ' + args.metric.upper() + ' of ' + net.upper() + ' on Quasi-Static Video Set:')

    metrics_mean = 0
    for index in range(4):
        start = 2
        end = 52

        metrics_mean += np.array(metrics[index][start: end]).mean()

    metrics_mean /= 4
    if metric[:4] == 'psnr':
        print("First 50: " + "{:.2f}".format(metrics_mean) + " dB")
    elif metric[:4] == 'ssim':
        print("First 50: " + "{:.4f}".format(metrics_mean))

    metrics_mean = 0
    for index in range(4):
        start = 2
        end = -2

        metrics_mean += np.array(metrics[index][start: end]).mean()

    metrics_mean /= 4
    if metric[:4] == 'psnr':
        print("All: " + "{:.2f}".format(metrics_mean) + " dB")
    elif metric[:4] == 'ssim':
        print("All: " + "{:.4f}".format(metrics_mean))

    metrics_mean = 0
    for index in range(4):
        start = -52
        end = -2
        metrics_mean += np.array(metrics[index][start: end]).mean()

    metrics_mean /= 4
    if metric[:4] == 'psnr':
        print("Last 50: " + "{:.2f}".format(metrics_mean) + " dB")
    elif metric[:4] == 'ssim':
        print("Last 50: " + "{:.4f}".format(metrics_mean))

elif args.data == 'Vid4':
    print('Mean ' + args.metric.upper() + ' of ' + net.upper() + ' on Vid4 dataset:')
    metrics_mean = 0
    for index in range(4):
        start = 1
        end = -1
        metrics_mean += np.array(metrics[index][start: end]).mean()

    metrics_mean /= 4
    if metric[:4] == 'psnr':
        print("{:.2f}".format(metrics_mean) + " dB")
    elif metric[:4] == 'ssim':
        print("{:.4f}".format(metrics_mean))
