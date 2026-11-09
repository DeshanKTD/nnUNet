from nnunetv2.training.nnUNetTrainer.variants.colon_improvement.nnUNetTrainerSkelTest import nnUNetTrainerSkelTest


class nnUNetTrainerSkelPatch(nnUNetTrainerSkelTest):
    """
    Image + skeleton (2 input channels) -> colon segmentation, trained on random patches.

    Everything (network, optimizer, data loading with nnUNetDataLoaderPatchWithMultiSeg, default nnU-Net
    augmentation, sliding-window validation) is inherited from nnUNetTrainerSkelTest. The only difference is that
    the training loop is enabled here (it is commented out in nnUNetTrainerSkelTest).
    """

    def run_training(self):
        # skip nnUNetTrainerSkelTest.run_training (loop disabled) and use the standard nnU-Net loop
        super(nnUNetTrainerSkelTest, self).run_training()
