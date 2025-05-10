from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
import torch


class nnUNetTrainerCustomSeg(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 0, 'do_bg': False}, {})