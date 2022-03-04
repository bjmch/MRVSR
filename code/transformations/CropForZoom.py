# -*- coding: utf-8 -*-
from .crop_image import crop_image


class CropForZoom():

    def __init__(self, factor=32):
        self.factor = factor

    def __call__(self, image, **kwargs):
        if len(kwargs) > 0:
            self.__init__(**kwargs)

        height = image.shape[0]
        width = image.shape[1]

        new_height = (height // self.factor) * self.factor
        new_width = (width // self.factor) * self.factor

        r_height = height - new_height
        r_width = width - new_width

        cropped = crop_image(image, new_height, new_width, indice_height=r_height // 2, indice_width=r_width // 2)

        return cropped
