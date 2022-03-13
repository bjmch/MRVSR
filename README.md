# Stable Long-Term Recurrent Video Super-Resolution
This is the official repository for the work [Stable Long-Term Recurrent Video Super-Resolution](https://arxiv.org/abs/2112.08950).
# Environment
The conda environment with libraries we used can be created with the following command:

```
$ conda create --name <env> --file conda_environment.txt
```
We notably used the following libraries:
- python=3.9.2
- pytorch=1.8.1
- cv2=4.5.1
- numpy
- scipy
- matplotlib
- pickle

We verified that our code works on the linux-64 platform. We performed experiments with an Intel I9-10940X CPU and one NVIDIA TITAN RTX GPU.
# Data
The proposed Quasi-Static Video Set, Sequence 1-XL and the standard Vid4 dataset can be downloaded from [here](https://drive.google.com/drive/folders/1mbndVeCqdBs-flk-Z_s-ISJZISNYZ8XN?usp=sharing). Please unzip and place them in the `data` folder. They are used in evalution.
# Pre-trained models
Pre-trained MRVSR, RFS3 and RLSP can be downloaded from [here](https://drive.google.com/drive/folders/1EzMWzV2sIJ4WYQycvedBoCqCNZP0BOD9?usp=sharing). Please place them in the `weights` directory.
For MRVSR, we do not include the code of SRNL layer that is used at training time, for proprietary reasons. However, the weights of MRVSR are already normalized by our SRNL code, ready to be used for inference. 
# Evaluation
Please change the current working directory to `code`:
```
$ cd code 
```

To run pre-trained MRVSR, RFS3 and RLSP on the test sequences:
```
$ python evaluate.py
```
This saves reconstructed images and computed per-frame PSNR/SSIM values in the `images` folder.

To aggregate metrics and reproduce results related to MRVSR, RFS3 and RLSP on the tables 1 and 2 of the main paper:
```
$ python display_metrics.py
```

To reproduce results related to MRVSR, RFS3 and RLSP on Figure 3:
```
$ python plot_metric.py
```
## Spatio-Temporal Receptive Field
To reproduce Figure 7:
```
$ python display_strf.py
```
## Temporal profiles
To reproduce Figure 8:
```
$ python display_temporal_profiles.py
```
# Visualize singular values of convolutional layers 
To visualize singular values for each convolutional layer of MRVSR:
```
$ python vizualize_SVD_spectrum.py
```
This outputs the following picture:

<img src="https://github.com/anonymous-for-research/anonymous-repo/blob/main/pictures/SVD_MRVSR.png" width="500">

Each label in the legend indicates the n-th layer in one of the sub-networks ξ, Φ<sup>L</sup> or Ψ. We see that SRNL successfully worked in constraining the spectral norm of only recurrent layers of Φ<sup>L</sup> to 1. We adopted the code from ([Sedghi et al., 2018](https://github.com/brain-research/conv-sv)). 
# Demo
The video of entire Sequence 1-XL reconstructed from MRVSR can be viewed [here](https://drive.google.com/file/d/1xv6sSdm1Hkm-0KFFvvuunn7-oSKYtk_H/view?usp=sharing).
