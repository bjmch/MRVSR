import torch
import torch.optim as optim
from utils import convert_grayscale_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

seed = 0
random_network = False
torch.manual_seed(seed)
np.random.seed(seed % 2 ** 32)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class InputSequence(torch.nn.Module):
    def __init__(self, tau, H, W):
        super(InputSequence, self).__init__()
        self.x = torch.nn.Parameter(torch.rand(size=(tau * 2 + 1, 1, 3, H, W)))

    def forward(self):
        return list(torch.unbind(self.x, dim=0))

    def clip_constrain(self):
        self.x.data.clamp_(0, 1)


H = 64
W = 64
tau = 40

lr = 1

from models import MRVSR_train

first_index = 1  # firstly super-resolved image's index in input sequence. As detailed in our paper, the image with index = 0 is used as x_{-1}
network = MRVSR_train(4, n_xi=3, n_phi=1, n_psi=3, f=128, nb_in_channels=3, nb_out_channels=1,
                      first_index=first_index)

cwd = os.getcwd()
weight_path = os.path.join(cwd, '../weights/mrvsr_weights.tar')
num_iterations = 1500
e1 = 750
e2 = 1250

mode = 'STRF'  # or 'TRF'

network.load_state_dict(torch.load(weight_path))
network.cuda()
network.eval()
# create the input sequence as the parameter to be optimized
input_sequence = InputSequence(tau, H, W).cuda()
# create optimizer
optimizer = optim.Adam(input_sequence.parameters(), lr=lr)
lr_lambda = lambda epoch: 0.1 ** (int(e1 is not None and epoch >= e1) + int(e2 is not None and epoch >= e2))  # + int(e3 is not None and epoch >= e3))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=True)
# main optimization loop

for iter_idx in range(num_iterations):
    print('Iteration: ', iter_idx)
    input = input_sequence()
    # with torch.no_grad():

    y = network(input)  # list of [1,3,H,W]
    if mode == 'TRF':
        loss = - torch.linalg.norm(y[len(y) // 2][0, 0, :, :])
    elif mode == 'STRF':
        loss = - torch.abs(y[len(y) // 2][0, 0, (H * 4) // 2, (W * 4) // 2])
    print('loss = ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    input_sequence.clip_constrain()
    scheduler.step()

x = input_sequence.x

with torch.no_grad():
    y = network(input_sequence())
    y = torch.stack(y, dim=0)
    y = y[:, :, 0, :, :].unsqueeze(2)
    y.clamp_(16 / 255, 235 / 255)
    y = y.squeeze(dim=1)
    T, C, H, W = y.size()
    y = y.permute(2, 0, 3, 1)
    y = y.reshape(H, T * W, 1)
    y = y.detach().cpu().numpy()

x = x.squeeze()
T, C, H, W = x.size()
x = x.permute(2, 0, 3, 1)
x = x.reshape(H, T * W, 3)
x = x.detach().cpu().numpy()

f = 10
fontsize = 10

fig, axs = plt.subplots(2)

axs[0].imshow(np.rint(cv2.resize(x, dsize=(5184, 64 * f), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8))
axs[1].imshow(convert_grayscale_to_rgb(np.rint(cv2.resize(y, dsize=(20224, 256 * f), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)))

ticks = np.arange(0, 5184, 64)
labels = np.arange(0, 81) - 40
axs[0].set_xticks(ticks, minor=False)
axs[0].set_xticklabels(labels, fontdict=None, minor=False, fontsize=fontsize)  # , rotation=90)

ticks = np.arange(0, 64 * f, (64 * f) // 3)
labels = np.arange(0, 64, 64 // 3) + 1
axs[0].set_yticks(ticks, minor=False)
axs[0].set_yticklabels(labels, fontdict=None, minor=False, fontsize=fontsize)

ticks = np.arange(0, 20224, 64 * 4)
labels = np.arange(0, 79) - 39
axs[1].set_xticks(ticks, minor=False)
axs[1].set_xticklabels(labels, fontdict=None, minor=False, fontsize=fontsize)  # , rotation=90)

ticks = np.arange(0, 256 * f, (256 * f) // 3)
labels = np.arange(0, 256, 256 // 3) + 1
axs[1].set_yticks(ticks, minor=False)
axs[1].set_yticklabels(labels, fontdict=None, minor=False, fontsize=fontsize)

plt.tight_layout()

plt.show()
