import torch
from torch.nn import Module, Conv2d, ReLU, Sequential, PixelShuffle

from utils.utils import UpsamplingInterpolation, PixelUnshuffle
from utils.utils import rgb2ycbcr as rgb2ycbcr_torch

from utils.utils import xavier_init


class RSLPCell(Module):

    def __init__(self, f, n, zoom_factor, nb_in_channels, nb_out_channels):
        super(RSLPCell, self).__init__()
        self.zoom_factor = zoom_factor
        input_channel_size = f + nb_in_channels * 3 + nb_out_channels * self.zoom_factor * self.zoom_factor
        self.nb_out_channels = nb_out_channels
        self.block1 = Sequential(Conv2d(input_channel_size, f, 3, 1, 1), ReLU())
        self.block1.apply(xavier_init)

        layers = []
        for i in range(n - 2):
            layers.append(Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        self.blocks = Sequential(*layers)
        self.blocks.apply(xavier_init)

        self.hidden_state_layer = Sequential(Conv2d(f, f, 3, 1, 1), ReLU())
        self.hidden_state_layer.apply(xavier_init)
        self.output_layer = Conv2d(f, self.nb_out_channels * self.zoom_factor * self.zoom_factor, 3, 1, 1)
        self.output_layer.apply(xavier_init)

    def forward(self, x, reference_frame):
        reference_frame = reference_frame.repeat(1, self.zoom_factor * self.zoom_factor, 1, 1)
        x = self.block1(x)
        x = self.blocks(x)
        hidden_state = self.hidden_state_layer(x)
        x = self.output_layer(x)
        x = x + reference_frame
        return x, hidden_state


class RLSP(Module):
    """
    Definition of the network RLSP
    """

    def __init__(self, zoom_factor, f, n, nb_in_channels, nb_out_channels, first_index=1):
        super(RLSP, self).__init__()
        self.nb_in_channels = nb_in_channels
        self.zoom_factor = zoom_factor
        self.f = f
        self.nb_out_channels = nb_out_channels
        self.RSLPCell = RSLPCell(self.f, n, self.zoom_factor, self.nb_in_channels, self.nb_out_channels)
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
        input = torch.cat((LRFlow, self.h, self.y), 1)
        y, self.h = self.RSLPCell(input, reference_frame_y)
        y = self.PixelShuffle(y)
        cb = self.bicubic(reference_frame_cb)
        cr = self.bicubic(reference_frame_cr)

        self.y = self.PixelUnshuffle(y)

        return torch.cat((y, cb, cr), 1)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        new_state_dict = dict()
        for k in state_dict.keys():
            new_state_dict[k[14:]] = state_dict[k]

        self.RSLPCell.load_state_dict(state_dict, strict)
