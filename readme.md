# nnUNet based segmentation networks

This repository contains following networks with nnUNet preprocessing and post processing pipelines. 

1. [x] [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
2. [x] [SwinUNETR](https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/swin_unetr.html)
3. [x] [UNet++](https://docs.monai.io/en/stable/networks.html#basicunetplusplus)
4. [x] [VNet](https://docs.monai.io/en/stable/networks.html#vnet)
5. [ ] [U-mamba](https://github.com/bowang-lab/U-Mamba)
6. [ ] [CoTr](https://github.com/YtongXie/CoTr/tree/main)
7. [x] [UNETR](https://docs.monai.io/en/0.7.0/_modules/monai/networks/nets/unetr.html)

# Setup the environment

We have to install the required packages for added networks.

1. Create a virtual environment using python venv or conda. Then activate it.
2. Install pytorch 2.0.1 
    - This specifica version used because to match the versions of U-mamba dependancies.
    - ```pip install torch==2.0.1 torchvision==0.15.2```
3. Install mamba packages - [Mamba](https://pypi.org/project/mamba-ssm/)
    - ```pip install mamba-ssm==2.2.2```
    - This needs causal-conv1d>=1.4.0 or higher for better performance. Check the link for installation guide.
    - Cuda should be installed and the version should be 11.7. However, with pytorch 2.0.1, it install python packages for cuda11.7.
4. Install Monai. Monai will provide some networks needed.
    - ```pip install monai```
5. Install nnUNet.
    -     
    ```
    git clone https://github.com/DeshanKTD/nnUNet
    cd nnUNet
    pip install -e .
    ```


For running the respective commands, visit the nnUNet documenation ([nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet))


# What will change with introduced network architectures
nnUNetv2 uses Residual Encoder UNet (ResEncUNet) as the backbone network for the segmentation. So its current experiment planners optimizes the parameters in this ResEncUNet. 

However, we haven't written experiment planners for the each type of network. So it just uses the preprocessed images from the preprocessing pipeline as inputs. If you want to do the optimizations for the network, you have to write your own experiment planner. But the easiest hack is to directly, change the parameters in the trainer. 

The trainers respective to each network is located in following path.
[/nnunetv2/training/nnUNetTrainer](/nnunetv2/training/nnUNetTrainer)
