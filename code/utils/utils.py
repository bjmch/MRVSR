import torch
import torch.nn.functional as torch_f
import warnings
import cv2
import numpy as np
import os
from transformations.SimulationBlur import SimulationBlur
from transformations.CropForZoom import CropForZoom


def rgb2ycbcr(img, maxVal=1):
    O = torch.Tensor([[16],
                      [128],
                      [128]]).cuda()
    O = O.squeeze(0)

    T = torch.Tensor([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                      [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                      [0.439215686274510, -0.367788235294118, -0.071427450980392]]).cuda()

    if maxVal == 1:
        O = O / 255.0

    t = torch.reshape(img, (img.size(0), img.size(1), img.size(2) * img.size(3)))

    t = t.permute(0, 2, 1)

    t = torch.matmul(t, torch.transpose(T, 1, 0))
    t[:, :, 0] += O[0]
    t[:, :, 1] += O[1]
    t[:, :, 2] += O[2]

    t = t.permute(0, 2, 1)

    ycbcr = t.reshape(img.size(0), img.size(1), img.size(2), img.size(3))

    return ycbcr


class UpsamplingInterpolation(torch.nn.Module):
    """
    Upsampling layer : increases the resolution of feature maps by an integer factor, with given interpolation method.
    Spatial size : output_size = input_size * factor
    Phasing : output[..., ::factor, ::factor] = input
    """

    def __init__(self, zoom_factor, mode='bilinear'):
        super(UpsamplingInterpolation, self).__init__()
        self.zoom_factor = zoom_factor
        self.mode = mode

    def forward(self, x):
        return upsampling_interpolation2d(x, zoom_factor=self.zoom_factor, mode=self.mode)


def upsampling_interpolation2d(x, zoom_factor, mode='bilinear'):
    input_size = torch.tensor(x.shape)

    # Interpolate inside corner pixels : center of the data
    interp_size = torch.tensor(x.shape)
    interp_size[-2:] = zoom_factor * (input_size[-2:] - 1) + 1
    interp_h, interp_w = interp_size[-2:]

    x_center = torch_f.interpolate(x, size=torch.Size(interp_size[-2:]), mode=mode, align_corners=True)

    # Initialize full-sized output
    output_size = torch.tensor(x.size())
    output_size[-2] = zoom_factor * input_size[-2]
    output_size[-1] = zoom_factor * input_size[-1]

    output = torch.zeros(torch.Size(output_size)).cuda()

    # Apply mirroring to recover lower and right borders
    # Copy interpolated center, then vertical mirror for lower border, then horizontal mirror for right border
    # todo : using torch padding
    output[:, :, :interp_h, :interp_w] = x_center
    for k in range(zoom_factor):
        output[:, :, interp_h + k - 1, :interp_w] = x_center[:, :, interp_h - k - 1, :]
    for k in range(zoom_factor):
        output[:, :, :, interp_w + k - 1] = output[:, :, :, interp_w - k - 1]

    return output


class PixelUnshuffle(torch.nn.Module):
    def __init__(self, downscale_factor):
        """
        Args:
            downscale_factor: k
        """
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, tensor):
        """
        Args:
            tensor: batchSize * c * k*w * k*h

        Returns:
            batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """

        return pixel_unshuffle(tensor, self.downscale_factor)


def pixel_unshuffle(tensor, downscale_factor):
    """
    Args:
        tensor: batchSize * c * k*w * k*h
        downscale_factor: k

    Returns:
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = tensor.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=tensor.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1

    return torch_f.conv2d(tensor, kernel, stride=downscale_factor, groups=c)


def xavier_init(layer):
    # Initialize a Conv2d or ConvTranspose2d or Sequential of them layers with Xavier initialization. Bias are initialized to zero.
    if isinstance(layer, torch.nn.Sequential):
        for l in layer:
            if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(l.weight, gain=1.)
                l.bias.data.zero_()
    elif isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(layer.weight, gain=1.)
        layer.bias.data.zero_()

    elif isinstance(layer, torch.nn.ModuleList):
        for l in layer:
            if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(l.weight, gain=1.)
                l.bias.data.zero_()


def load_weights(network, filename, prefix='', key_offset=0):
    """ **Function to load saved coefficients of a network**

    This function loads the coefficient of a network saved in a .tar file which can be done with the function :meth:`torch.save`.

    :arg
        network (:class: 'nn.module') PyTorch network
        filename (:class: 'str'): Relative or absolute path where the coefficient are saved. Note that the path must end by ".tar".
    >>> net = MyNet()
    >>> load_weights(network, 'coefficients.tar')
    .. warning::
     The entries of the dictionary of the coefficient saved must match the name of one of the network layers. If
     not the coefficients corresponding to the entry not matched won't be loaded. A warning will be raised in this case. Note that you can check the name of
     the layers with the function :meth:`state_dict`.

    """

    # print(key_offset)

    # Load weights
    pretrained_dict = torch.load(filename)

    print(pretrained_dict.keys())

    # Get current weights
    model_dict = network.state_dict()

    print(model_dict.keys())

    pretrained_dict_key_offsetted = dict()
    for k in pretrained_dict.keys():
        pretrained_dict_key_offsetted[k[key_offset:]] = pretrained_dict[k]

    # (Warning Only) Check that both dictionaries have the same elements
    for model_key in model_dict.keys():
        model_key_without_prefix = model_key[len(prefix):]
        # if model_key not in pretrained_dict.keys():
        if model_key_without_prefix not in pretrained_dict_key_offsetted.keys():
            warnings.warn("The layer : {} is not present in the weights loaded. Please check if this is conform.".format(model_key))

    # Replace matching current weights with loaded weights
    pretrained_dict = {prefix + k: v for k, v in pretrained_dict_key_offsetted.items() if prefix + k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)


def read_image(image_path, num_bits=8, normalize=False):
    """

    Args:
        image_path (str): path to image file
        num_bits (int): number of lsb bits for the data
        normalize (bool):
            if False [default], the output is the raw data, encoded as uint8 or uint16
            if True, the output is a floating point normalized array between 0 and 1

    Returns:
        image: numpy array, either integers or floats.
    """

    image = cv2.imread(image_path, -1)
    if image is None:
        raise OSError(f'cv2 is unable to read file "{image_path}"')

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if normalize:
        image = image / (2 ** num_bits - 1)
    else:
        if num_bits > 8:
            image = np.array(image).astype('uint16')
        else:
            image = np.array(image).astype('uint8')

    return image


def image_to_tensor_shape(image):
    """
    Modify image's shape (W, H, C) or (W, H) to fit torch tensor convention (C, W, H)

    :arg
        image (:class:'np.ndarray'): image in the form (W, H, C) or (W, H)
    :return
        image (:class: 'np.ndarray'): image in a torch tensor shape (C, W, H)
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    else:
        assert len(image.shape) == 3, "Input must be an 2d or 3d image"
        image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    return image


def write_image(image, saving_path, num_bits=8):
    if num_bits > 8:
        image = image.astype('uint16')
    else:
        image = image.astype('uint8')
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not os.path.exists(os.path.dirname(saving_path)) and not (os.path.dirname(saving_path) == ''):
        os.makedirs(os.path.dirname(saving_path))
    ok = cv2.imwrite(saving_path, image)
    if not ok:
        raise OSError(f'cv2 cannot write file {saving_path}')


def quantify_image(image, nb_bit=16):
    factor = 2 ** nb_bit - 1
    return np.rint(image * factor) / factor


def HR2LR(image, leave_border=False):
    """
    Generate from the input HR image its corresponding LR image. The HR image is generated by applying gaussian blur with standard deviation 1.5 and and sampling every s = 4 pixel in both spatial dimensions

    :param image: Input HR image
    :param leave_border: If True, exclude border (to ignore the border effect of the gaussian blur)
    :return: The corresponding LR image
    """
    blur = SimulationBlur(blur_variance=1.5)
    image = blur(image)
    if not leave_border:
        image = image[6:-6, 6:-6]

    # Crop the HR image so that the later becomes divisible by 32 (= zoom factor * (2 ** number of poolings in FRVSR))
    cropForZoom = CropForZoom(factor=32)
    image = cropForZoom(image)

    image = image[::4, ::4, :]
    image = np.clip(image, a_min=0, a_max=1)
    image = quantify_image(image, 16)  # Simulate 16-bit image

    return image


def process_HR_image(image, leave_border=False):
    """
    Make the HR image conform with the LR frame generated by the above functiion HR2LR

    :param image: Input HR image
    :param leave_border: If True, exclude border (to ignore the border effect of the gaussian blur)
    :return: Processed HR image
    """
    if not leave_border:
        image = image[6:-6, 6:-6]  # avoid border

    # Crop the HR image so that the later becomes divisible by 32 (= zoom factor * (2 ** number of poolings in FRVSR))
    cropForZoom = CropForZoom(factor=32)
    image = cropForZoom(image)

    return image


def convert_grayscale_to_rgb(image):
    if len(image.shape) == 3 and not (image.shape[2] == 1):
        raise Exception("Input image must have one channel.")
    else:
        if len(image.shape) == 3:
            image = image[:, :, 0]
        elif len(image.shape) != 2:
            raise Exception("Only 2 dimensional images are accepted")

    return np.stack((image,) * 3, axis=-1)
