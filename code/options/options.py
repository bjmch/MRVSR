import argparse


def parse_argument(task='vsr'):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", help="the dataset on which the inference is run",
                        type=str, default='QSV')
    parser.add_argument("--sequence", help="the sequence in the dataset to super-resolve",
                        type=str, default='1')
    parser.add_argument("--net", help="network to run",
                        type=str, default='rfs3')
    parser.add_argument("--metric", help="metric to compute at evaluation time. Either 'psnr' or 'ssim'",
                        type=str, default='psnr')

    args = parser.parse_args()

    return args
