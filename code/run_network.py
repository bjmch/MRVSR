from models import MRVSR
from models import RFS3
from models import RLSP
from utils import SequenceFilepathParser
from utils import read_image, write_image, image_to_tensor_shape, HR2LR, process_HR_image
from utils import rgb2ycbcr as rgb2ycbcr_torch
import torch
import os
from metrics import ssim
import pickle
from options import parse_argument

"""
This script super-resolves a sequence using a network. The sequence and the network are specified by command-line options.
"""

args = parse_argument()
cwd = os.getcwd()
first_index = 1  # firstly super-resolved image's index in the input sequence. As detailed in our paper, the first image with index = 0 is used as x_{-1}

###########
# Network #
###########

net = args.net

if net == 'mrvsr':
    network = MRVSR(4, n_xi=3, n_phi=1, n_psi=3, f=128, nb_in_channels=3, nb_out_channels=1,
                    first_index=first_index)
    weight_path = os.path.join(cwd, '../weights/mrvsr_weights.tar')
elif net == 'rfs3':
    network = RFS3(4, n=7, f=128, nb_in_channels=3, nb_out_channels=1, first_index=first_index)
    weight_path = os.path.join(cwd, '../weights/rfs3_weights.tar')

elif net == 'rlsp':
    network = RLSP(4, n=7, f=128, nb_in_channels=3, nb_out_channels=1, first_index=first_index)
    weight_path = os.path.join(cwd, '../weights/rlsp_weights.tar')

network.load_state_dict(torch.load(weight_path))
state_initialized = False
network.cuda()
network.eval()

###########
# Dataset #
###########

if args.data == 'QSV':
    dataset = 'QuasiStaticVideoSet'
    sequence = 'Sequence' + args.sequence
    leave_border = False
elif args.data == 'Vid4':
    dataset = 'Vid4'
    sequence = args.sequence
    leave_border = True

reader_test = SequenceFilepathParser(os.path.join(cwd, '../data/' + dataset + '/' + sequence),
                                     'png$')
images = reader_test[0]  # List of all HR frames of the sequence

##############
# Evaluation #
##############

save_img_directory = os.path.join(cwd, '../images', net, dataset, sequence)
os.makedirs(save_img_directory, exist_ok=True)

save_GT = False  # If True, save GT images
if save_GT:
    save_GT_directory = os.path.join(cwd, '../images', 'GT', dataset, sequence)
    os.makedirs(save_GT_directory, exist_ok=True)


def process(image_np, processing_func, *args):
    """
    Process the input image on the fly
    :param image_np: imput image in numpy array
    :param processing_func: function that is applied on the input image
    :return: Processed image in torch tensor
    """
    image_np_noise = processing_func(image_np, *args)
    image_torch_noise = image_to_tensor_shape(image_np_noise)
    return torch.tensor(image_torch_noise).unsqueeze(0).cuda()


# Metric per frame
sp_border = 8  # exclude border pixels
PSNRs = []
SSIMs = []

with torch.no_grad():
    for i in range(first_index, len(images) - first_index):

        # x_t and y_t
        image_np = read_image(images[i], 8, normalize=True)
        target = process(image_np, process_HR_image, leave_border)
        reference_frame = process(image_np, HR2LR, leave_border)

        # x_{t-1} and x_{t+1}
        if i == 0:
            previous = torch.zeros_like(reference_frame).cuda()
        else:
            previous = process(read_image(images[i - 1], 8, normalize=True), HR2LR, leave_border)
        if i == len(images) - 1:
            next = torch.zeros_like(reference_frame).cuda()
        else:
            next = process(read_image(images[i + 1], 8, normalize=True), HR2LR, leave_border)

        input = [previous, reference_frame, next]

        if not state_initialized:
            network.init_state(input)
            state_initialized = True

        output = network(input)

        # Compare prediction and target on the Y channel

        prediction = output[:, 0, :, :].unsqueeze(1)
        target = rgb2ycbcr_torch(target)[:, 0, :, :].unsqueeze(1)
        target *= 255

        prediction = torch.clip(prediction, 16 / 255, 235 / 255)
        prediction = torch.round(prediction * 255)

        if save_img_directory is not None:
            write_image(prediction.clone().detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), os.path.join(save_img_directory, 'im_' + str(i) + '.png'))
        if save_GT:
            write_image(target.clone().detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), os.path.join(save_GT_directory, 'im_' + str(i) + '.png'))

        mse = torch.nn.MSELoss()(prediction[:, :, sp_border:-sp_border, sp_border:-sp_border],
                                 target[:, :, sp_border:-sp_border, sp_border:-sp_border])

        psnr = 20. * torch.log10(255. / torch.sqrt(mse))
        ssim_value = ssim(prediction[:, :, sp_border:-sp_border, sp_border:-sp_border] / 255,
                          target[:, :, sp_border:-sp_border, sp_border:-sp_border] / 255)

        # Save metrics per frame
        if save_img_directory is not None:
            PSNRs.append(psnr.item())
            SSIMs.append(ssim_value.item())
            pickle_path = os.path.join(save_img_directory, 'psnrs.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(PSNRs, f)
            pickle_path = os.path.join(save_img_directory, 'ssims.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(SSIMs, f)

    print('Inference of ' + net.upper() + ' on sequence ' + args.sequence + ' of ' + dataset + ' finished.')
