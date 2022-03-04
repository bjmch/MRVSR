import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def SVD_Conv_Tensor_NP(filter, inp_size):
    # Adopted from https://github.com/brain-research/conv-sv
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    transform_coeff = np.fft.fft2(filter, inp_size, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    return np.linalg.svd(transform_coeff, compute_uv=False)


cwd = os.getcwd()
weight_path = os.path.join(cwd, '../weights/mrvsr_weights.tar')

pretrained_dict = torch.load(weight_path)

svd_list = []
for key in pretrained_dict.keys():
    if 'weight' in key:
        weight = pretrained_dict[key]
        weight_np = weight.permute(2, 3, 1, 0).detach().cpu().numpy()
        svd = SVD_Conv_Tensor_NP(weight_np, [128, 128])
        svd_list.append([key, np.flip(np.sort(svd.flatten()), 0)])

labels = {'xi_layers.0.weight': 'ξ_1', 'xi_layers.2.weight': 'ξ_2', 'xi_layers.4.weight': 'ξ_3', 'phi_hidden_state_layer.0.weight': 'Φ_0 (hidden state layer)', 'phi_layers.0.weight': 'Φ_1',
          'psi_layers.0.weight': 'Ψ_1', 'psi_layers.2.weight': 'Ψ_2', 'psi_output_layer.weight': 'Ψ_3'}

svd_list = svd_list[2:-3] + svd_list[0:2] + svd_list[-3:]
colors = ['orange', 'r', 'g', 'c', 'k', 'b', 'm', 'y']

for i in range(len(svd_list)):
    name, svds = svd_list[i]
    normalized_layer_number = (0.1 + i) / (0.2 + len(svd_list))
    this_color = colors[i]
    plt.plot(np.array(range(len(svds))) / len(svds), svds, label=labels[name], color=this_color)
    # plt.show()
axes = plt.gca()
plt.legend(fontsize='14', ncol=2)
plt.xlabel('Singular value percentile', fontsize=15)
plt.ylabel('Singular value', fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.tight_layout()

plt.show()
