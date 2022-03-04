import torch
from torch.nn import Module, Conv2d, ReLU, Sequential, PixelShuffle
from utils import UpsamplingInterpolation
from utils import rgb2ycbcr as rgb2ycbcr_torch


class MRVSRCell(Module):
    def __init__(self, f, n_xi, n_phi, n_psi, zoom_factor, nb_in_channels, nb_out_channels):
        super(MRVSRCell, self).__init__()
        self.zoom_factor = zoom_factor
        self.nb_out_channels = nb_out_channels
        self.n_xi = n_xi

        layers = []
        for i in range(n_xi):
            if i == 0:
                layers.append(Conv2d(in_channels=nb_in_channels * 3, out_channels=f, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        self.xi_layers = Sequential(*layers)

        self.phi_hidden_state_layer = Sequential(Conv2d(f, f, 3, 1, 1), ReLU())

        layers = []
        in_channels = 2 * f
        if n_xi == 0:
            in_channels = nb_in_channels * 3 + f
        for i in range(n_phi):
            if i == 0:

                layers.append(Conv2d(in_channels, f, 3, 1, 1))
            else:

                layers.append(Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        self.phi_layers = Sequential(*layers)

        layers = []
        for i in range(n_psi - 1):
            if i == 0:
                layers.append(Conv2d(in_channels=2 * f, out_channels=f, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(Conv2d(in_channels=f, out_channels=f, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        self.psi_layers = Sequential(*layers)

        self.psi_output_layer = Conv2d(f, self.nb_out_channels * self.zoom_factor * self.zoom_factor, 3, 1, 1)

        self.feature_shift = True

    def forward(self, LRFlow, h, reference_frame):
        reference_frame = reference_frame.repeat(1, self.zoom_factor * self.zoom_factor, 1, 1)

        if self.n_xi != 0:
            x = self.xi_layers(LRFlow)
        else:
            x = LRFlow

        hidden_state = self.phi_hidden_state_layer(h)

        input = torch.cat((x, hidden_state), 1)

        h_out = self.phi_layers(input)

        if self.feature_shift:
            z = torch.cat((h_out, h), 1)
        else:
            z = h_out

        z = self.psi_layers(z)

        z = self.psi_output_layer(z)
        z = z + reference_frame
        return z, h_out


class MRVSR(Module):
    """
    Definition of the network MRVSR
    """

    def __init__(self, zoom_factor, f, n_xi, n_phi, n_psi, nb_in_channels, nb_out_channels, first_index=1):
        super(MRVSR, self).__init__()
        self.nb_in_channels = nb_in_channels
        self.zoom_factor = zoom_factor
        self.f = f
        self.nb_out_channels = nb_out_channels
        self.MRVSRCell = MRVSRCell(self.f, n_xi, n_phi, n_psi, self.zoom_factor, self.nb_in_channels, self.nb_out_channels)
        self.PixelShuffle = PixelShuffle(self.zoom_factor)
        self.bicubic = UpsamplingInterpolation(zoom_factor, 'bicubic')
        self.first_index = first_index

    def init_state(self, images):
        self.h = torch.zeros(images[0].size(0), self.f, images[0].size(2), images[0].size(3)).cuda()

    def forward(self, images):
        reference_frame = images[1]
        reference_frame_ycbcr = rgb2ycbcr_torch(reference_frame, 1)
        reference_frame_y = reference_frame_ycbcr[:, 0, :, :].unsqueeze(1)
        reference_frame_cb = reference_frame_ycbcr[:, 1, :, :].unsqueeze(1)
        reference_frame_cr = reference_frame_ycbcr[:, 2, :, :].unsqueeze(1)

        LRFlow = torch.cat((images[0], reference_frame, images[2]), 1)
        y, self.h = self.MRVSRCell(LRFlow, self.h, reference_frame_y)
        y = self.PixelShuffle(y)
        cb = self.bicubic(reference_frame_cb)
        cr = self.bicubic(reference_frame_cr)

        return torch.cat((y, cb, cr), 1)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        self.MRVSRCell.load_state_dict(state_dict, strict)


class MRVSR_train(Module):
    """
    Definition of the network MRVSR
    """

    def __init__(self, zoom_factor, f, n_xi, n_phi, n_psi, nb_in_channels, nb_out_channels, first_index=1):
        super(MRVSR_train, self).__init__()
        self.first_index = first_index
        self.mrvsr = MRVSR(zoom_factor, f, n_xi, n_phi, n_psi, nb_in_channels, nb_out_channels, first_index)

    def forward(self, images):
        output = []
        initialized = False
        for i in range(self.first_index, len(images) - self.first_index):
            reference_frame = images[i]
            if i == 0:
                previous = torch.zeros_like(images[i]).cuda()
            else:
                previous = images[i - 1]
            if i == len(images) - 1:
                next = torch.zeros_like(images[i]).cuda()
            else:
                next = images[i + 1]

            input = [previous, reference_frame, next]
            if not initialized:
                self.mrvsr.init_state(input)
                initialized = True
            output.append(self.mrvsr(input))

        return output

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        self.mrvsr.load_state_dict(state_dict, strict)
