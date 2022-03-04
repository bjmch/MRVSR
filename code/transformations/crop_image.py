# -*- coding: utf-8 -*-
import numpy as np


def crop_image(image, height=64, width=64, indice_height=-1, indice_width=-1, with_indices=False):
    """
    Crop an image randomly or according to a specific set of indexes
    :arg
        image (:class: 'np.ndarray'): image to be cropped
        height (:class: 'int'): height of the crop
        width (:class: 'int'): width of the crop
        indice_height (:class: 'int'): vertical origin of the crop, ramdom if not defined
        indice_width (:class: 'int'): horizontal origin of the crop, ramdom if not defined
        with_indices (:class: 'boolean'): if true, return the indexes of the crop applied
    :return
        image (:class: 'np.ndarray'): cropped image
    """

    if indice_height == -1 or indice_width == -1:
        indice_height = 2 * int((image.shape[0] - height) * np.random.uniform() / 2)
        indice_width = 2 * int((image.shape[1] - width) * np.random.uniform() / 2)

    if with_indices:
        return image[indice_height:indice_height + height, indice_width:indice_width + width], \
               indice_height, indice_width
    else:
        return image[indice_height:indice_height + height, indice_width:indice_width + width]
