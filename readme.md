# nnUNet based segmentation networks

This repository contains following networks with nnUNet preprocessing and post processing pipelines. 

1. [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
2. [SwinUNETR](https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/swin_unetr.html)
3. [UNet++](https://docs.monai.io/en/stable/networks.html#basicunetplusplus)
4. [VNet](https://docs.monai.io/en/stable/networks.html#vnet)
5. [U-mamba](https://github.com/bowang-lab/U-Mamba)
6. [CoTr](https://github.com/YtongXie/CoTr/tree/main)
7. [UNETR](https://docs.monai.io/en/0.7.0/_modules/monai/networks/nets/unetr.html)

# Setup the environment


For running the respective commands, visit the nnUNet documenation ([nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet))


# What will change with introduced network architectures
nnUNetv2 uses Residual Encoder UNet (ResEncUNet) as the backbone network for the segmentation. So its current experiment planners optimizes the parameters in this ResEncUNet. 

However, we haven't written experimen planners for the each type of network. So it just uses the preprocessed images from the preprocessing pipeline as inputs. If you want to do the optimizations for the network, you have to write your own experiment planner. But the easiest hack is to directly, change the parameters in the trainer. 

