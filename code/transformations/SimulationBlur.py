# -*- coding: utf-8 -*-

import copy
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter  # scipy separate nd gaussian filter
from scipy.ndimage.filters import convolve


class SimulationBlur():
    """
    Blurs an image
    The blur is either gaussian or given by the user as a numpy array
    """

    def __init__(self, blur_variance=1, blur_variance_min=1, blur_variance_max=3, random_variance=False, mode='mirror', filter='gaussian', filter_array=np.array([]), **kwargs):
        """
        *** Adds blur to an image.***

        Choice between 'gaussian' for gaussian blur and 'custom' for imported np.ndarray blurring kernel.
        If filter_array is a list, the kernel choice will be random in this list.

        Args:
            blur_variance: (:class:`float`):
            blur_variance_min: (:class:`float`):
            blur_variance_max: (:class:`float`):
            random_variance: (:class:`bool`):
            mode: (:class:`str`): 'mirror' or 'reflect' ...
            filter: (:class:`str`):
            filter_array: (:class:`np.ndarray`) or (:class:`list`)
        """

        super(SimulationBlur).__init__(**kwargs)

        self.blur_variance = blur_variance
        self.blur_variance_min = blur_variance_min
        self.blur_variance_max = blur_variance_max
        self.random_variance = random_variance
        self.mode = mode

        supported_filters = ['gaussian', 'custom']
        if filter not in supported_filters:
            raise NameError(f"filter '{filter}' not supported")
        self.filter = filter

        self.filter_array = filter_array

        if self.filter == "custom" and not isinstance(self.filter_array, list):
            self.filter_array = [self.filter_array]

    def __call__(self, im, **kwargs):

        if len(kwargs) > 0:
            self.__init__(**kwargs)

        image = copy.copy(im)

        if self.filter == "gaussian":
            if len(im.shape) < 3:
                image = gaussian_filter(image, self.blur_variance, mode=self.mode)

            else:
                for i in range(image.shape[2]):
                    image[:, :, i] = gaussian_filter(image[:, :, i], self.blur_variance, mode=self.mode)

        elif self.filter == "custom":
            weights = random.choice(self.filter_array)

            if len(im.shape) < 3:
                image = convolve(image, weights, output=None, mode=self.mode, cval=0.0, origin=0)

            else:
                for i in range(image.shape[2]):  # for each channel
                    image[:, :, i] = convolve(image[:, :, i], weights, output=None, mode=self.mode, cval=0.0, origin=0)

        return image

    def random(self):

        if self.random_variance:
            self.blur_variance = np.random.uniform(self.blur_variance_min, self.blur_variance_max)
