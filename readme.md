# nnUNet based segmentation networks

This repository includes the following networks, along with nnUNet preprocessing and postprocessing pipelines.

1. [x] [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
2. [x] [SwinUNETR](https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/swin_unetr.html)
3. [x] [UNet++](https://docs.monai.io/en/stable/networks.html#basicunetplusplus)
4. [x] [VNet](https://docs.monai.io/en/stable/networks.html#vnet)
5. [ ] [U-mamba](https://github.com/bowang-lab/U-Mamba)
6. [x] [CoTr](https://github.com/YtongXie/CoTr/tree/main)
7. [x] [UNETR](https://docs.monai.io/en/0.7.0/_modules/monai/networks/nets/unetr.html)

# Setup the environment

It is necessary to install the required packages for the additional networks.

<!-- 1. Create a virtual environment using python venv or conda. Then activate it.
2. Install pytorch 2.0.1 
    - This specifica version used because to match the versions of U-mamba dependancies.
    - ```pip install torch==2.0.1 torchvision==0.15.2```
3. Install mamba packages - [Mamba](https://pypi.org/project/mamba-ssm/)
    - ```pip install mamba-ssm==2.2.2```
    - This needs causal-conv1d>=1.4.0 or higher for better performance. Check the link for installation guide.
    - Cuda should be installed and the version should be 11.7. However, with pytorch 2.0.1, it install python packages for cuda11.7.
4. Install Monai. Monai will provide some networks needed. -->

1. Install nnUNet.
    -     
    ```
    git clone https://github.com/DeshanKTD/nnUNet
    cd nnUNet
    pip install -e .
    ```
2. Mamba is not yet supported because it requires mamba-ssm, which currently does not support torch==2.6.0 (the latest version).


For running the respective commands, visit the nnUNet documenation ([nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet))


# What will change with introduced network architectures
nnUNetv2 uses the Residual Encoder UNet (ResEncUNet) as its backbone network for segmentation. As a result, the current experiment planners are specifically optimized for ResEncUNet.

We have not implemented experiment planners for each individual network type. Therefore, these networks utilize the preprocessed images from the standard preprocessing pipeline without additional optimization. If you wish to optimize parameters for a specific network, you will need to implement a custom experiment planner.

However, the simplest workaround is to manually modify the parameters directly within the trainer.

The trainers corresponding to each network can be found at the following path:
[/nnunetv2/training/nnUNetTrainer](/nnunetv2/training/nnUNetTrainer)

# Usage Instructions

## Creating the datasets.
In nnUNet, datasets must be created following the structure described in the [Dataset format](/documentation/dataset_format.md). 
There are three required directories: `nnUNet_raw, nnUNet_preprocessed, nnUNet_results`. 
These directories should be set as environment variables. To simplify this setup, it's recommended to either create a script that can be sourced or add the environment variable definitions to your shell's startup file (e.g., `.bashrc`, `.zshrc`).

```
export nnUNet_raw="<nnUNet_raw path>"
export nnUNet_preprocessed="<nnUnet_preprocessed_path>"
export nnUNet_results="nnUNet_results_path"
```

The raw data should be put inside `nnUNet_raw` directory under `imagesTr`,`imagesTs`, and `labelsTr`. Follow the [documentation](/documentation/dataset_format.md) for more information.


## Preprocessing 

When provided with a new dataset, nnU-Net extracts a dataset fingerprint, which includes dataset-specific properties such as image dimensions, voxel spacings, intensity distributions, and more. This fingerprint is then used to automatically design three different U-Net configurations tailored to the dataset.

Each of these configurations uses its own preprocessed version of the dataset, ensuring that the training pipelines are optimized for varying resolutions and characteristics.

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

`DATASET_ID` is the number given for the dataset accoridng to the format. If dataset is named as `Dataset002_XXX`, `DATASET_ID` is `2`. 

This process can be divided in to 3 steps as extracting the fingerprint, plan the experiment and preprocessing. You can use commands `nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment` and `nnUNetv2_preprocess` (in that order) to complete the preprocessing. There is a option where you can change the experiment planner and use your own one. In such scenreos, this will be benificial.

nnUNetv2 comes with added experiment planner. To run it use, 

```
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEnc(M/L/XL)
```

In here, you have to select betweenm M, L and XL. More information is given on [Residual Encoder Presets in nnU-Net](/documentation/resenc_presets.md).

This step will create preprocessed data under `nnUNet_preprocessed` directory.  So initial preprocessing will not perform at training stage.

### Model Training

More details on model training is given in [documentation](/documentation/how_to_use_nnunet.md). Here I'll keep it short. 

During the preprocessing step, nnU-Net follows a selected plan—typically the default plan—which defines several configurations such as `2d`, `3d_fullres`, `3d_lowres`, or any custom-defined configurations. It's important to use the appropriate plan when initiating training. If you're using the default plan, there's no need to specify it explicitly.

By default, nnU-Net performs 5-fold cross-validation. Given the potentially long training times, it's often best to run each fold separately. If you have sufficient computational resources, you can also run multiple folds in parallel to speed up the process.


Here is the command for training with default plan.

```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
```
nnUNetv2_train 1 3d_fullres 0
```

This commands denotes, use dataset 1, 3d_fullres as the configuration and perform fold 0. 

If we use a plan, we have to specify the plan as and argument. 
```
nnUNetv2_train 2 3d_fullres 0 -p nnUNetResEncUNetMPlans
```

There are set of arguments that can be passed into the commands. You can use `-h` argument to check all availbale arguments.

### Running Custom Model Trainers
Currently, nnU-Net experiment plans are specifically designed for the UNet and Residual Encoder UNet (ResEncUNet) architectures. The parameters defined in these plans are optimized for these two networks based on the dataset characteristics.

However, the newly added network trainers are not yet optimized through custom experiment planners. Despite this, you can still use the preprocessed data from the default planner (or any other planner like ResEnc) to train these networks.

To do so, simply provide the appropriate trainer class as an argument when launching the training. This allows the network to utilize the existing preprocessing setup while using your custom trainer for model training.

```
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainerVNet
```

All the available trainers are listed in this [directory](/nnunetv2/training/nnUNetTrainer/). You can modify the network architecture to suit your specific needs. The best and most robust approach is to write a custom experiment planner, which allows you to optimize preprocessing and training parameters specifically for your network.

However, writing a new experiment planner can be time-consuming and requires a good understanding of the nnU-Net planning pipeline. If you're looking for a quicker solution, you can start by directly modifying the trainer or using existing plans with your custom network. This allows for flexibility without the overhead of implementing a full planner from scratch.

If you use a planner, you have to specify it while training.

```
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainerVNet -p nnUNetResEncUNetMPlans
```

All five folds has to be trained before, using it for inference.

## Inference

For each of the desired configurations, run (with default trainer):
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

`INPUT_FOLDER` contains the images that needs to predicted. Segmentation outputs will be put into `OUTPUT_FOLDER`. 

This is an example for `nnUNetTrainerVNet` trainer
```
nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $DATASET_ID -c 3d_fullres -tr nnUNetTrainerVNet
```

More details about how to run is on [documentation](/documentation/how_to_use_nnunet.md).
