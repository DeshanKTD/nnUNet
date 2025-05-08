# nnUNet based segmentation networks

This repository contains following networks with nnUNet preprocessing and post processing pipelines. 

1. nnUNet
2. SwinUNETR
3. UNet++
4. VNet
5. U-mamba
6. CoTr
7. UNETR

# Setup the environment





# What will change with introduced network architecture
nnUNetv2 uses Residual Encoder UNet (ResEncUNet) as the backbone network for the segmentation. So its current experiment planners optimizes the parameters in this ResEncUNet. 

However, we haven't written experimen planners for the each type of network. So it just uses the preprocessed images from the preprocessing pipeline as inputs. If you want to do the optimizations for the network, you have to write your own experiment planner. But the easiest hack is to directly, change the parameters in the trainer. 

