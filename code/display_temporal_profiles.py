from utils import read_image
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_temporal_profiles(net='mrvsr', sequence='Sequence1', start=62, length=125, save=False):
    cwd = os.getcwd()
    images_prefix = os.path.join(cwd, '../images', net, 'QuasiStaticVideoSet', sequence) + '/im_'

    for i in range(start, start + length):
        image = read_image(images_prefix + str(i) + '.png')
        if i == start:
            h, w, _ = image.shape
            segments = np.zeros((length, w, 3), dtype='uint8')
        segments[i - start, :, :] = image[h // 2, :, :]

    segments = segments[:, :w // 4, :]

    if save:
        import PIL
        directory = os.path.join(cwd, '../temporal_profiles', net, 'QuasiStaticVideoSet', sequence)
        os.makedirs(directory, exist_ok=True)
        PIL.Image.fromarray(segments).save(os.path.join(directory, 'temporal_profile.png'))


if __name__ == "__main__":
    plot_temporal_profiles(net='mrvsr', save=True)
    plot_temporal_profiles(net='rfs3', save=True)
