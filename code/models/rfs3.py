import torch
from torch.nn import Module, Conv2d, ReLU, Sequential, PixelShuffle

from utils.utils import UpsamplingInterpolation, PixelUnshuffle
from utils.utils import rgb2ycbcr as rgb2ycbcr_torch

from utils.utils import xavier_init


class RFS3Cell(Module):
    def __init__(self, f, n, zoom_factor, nb_in_channels, nb_out_channels):
        super(RFS3Cell, self).__init__()
        self.zoom_factor = zoom_factor
        input_channel_size = nb_in_channels * 3
        self.nb_out_channels = nb_out_channels
        self.block1 = Sequential(Conv2d(input_channel_size, f, 3, 1, 1), ReLU())
        self.block1.apply(xavier_init)

        layers = []
        for i in range(n - 2):
            layers.append(Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        self.blocks = Sequential(*layers)
        self.blocks.apply(xavier_init)

        self.output_layer = Conv2d(f, self.nb_out_channels * self.zoom_factor * self.zoom_factor, 3, 1, 1)
        self.output_layer.apply(xavier_init)

    def forward(self, x, reference_frame):
        reference_frame = reference_frame.repeat(1, self.zoom_factor * self.zoom_factor, 1, 1)
        x = self.block1(x)
        x = self.blocks(x)

        x = self.output_layer(x)
        x = x + reference_frame
        return x


class RFS3(Module):

    def __init__(self, zoom_factor, f, n, nb_in_channels, nb_out_channels, first_index=1):
        super(RFS3, self).__init__()
        self.nb_in_channels = nb_in_channels
        self.zoom_factor = zoom_factor
        self.f = f
        self.nb_out_channels = nb_out_channels
        self.RFS3Cell = RFS3Cell(self.f, n, self.zoom_factor, self.nb_in_channels, self.nb_out_channels)
        self.PixelUnshuffle = PixelUnshuffle(self.zoom_factor)
        self.PixelShuffle = PixelShuffle(self.zoom_factor)
        self.bicubic = UpsamplingInterpolation(zoom_factor, 'bicubic')
        self.first_index = first_index

    def init_state(self, images):
        self.h = torch.zeros(images[0].size(0), self.f, images[0].size(2), images[0].size(3)).cuda()
        self.y = torch.zeros(images[0].size(0), self.nb_out_channels * self.zoom_factor * self.zoom_factor, images[0].size(2), images[0].size(3)).cuda()

    def forward(self, images):
        reference_frame = images[1]
        reference_frame_ycbcr = rgb2ycbcr_torch(reference_frame, 1)
        reference_frame_y = reference_frame_ycbcr[:, 0, :, :].unsqueeze(1)
        reference_frame_cb = reference_frame_ycbcr[:, 1, :, :].unsqueeze(1)
        reference_frame_cr = reference_frame_ycbcr[:, 2, :, :].unsqueeze(1)

        LRFlow = torch.cat((images[0], reference_frame, images[2]), 1)
        input = LRFlow
        y = self.RFS3Cell(input, reference_frame_y)
        y = self.PixelShuffle(y)
        cb = self.bicubic(reference_frame_cb)
        cr = self.bicubic(reference_frame_cr)

        return torch.cat((y, cb, cr), 1)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        self.RFS3Cell.load_state_dict(state_dict, strict)
