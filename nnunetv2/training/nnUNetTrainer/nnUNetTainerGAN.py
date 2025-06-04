
import torch
from torch import autocast
import torch.nn as nn
from torch import distributed as dist
from network_architectures.networks.unet.se_unet.se_unet_3d import SEUNet3D
from network_architectures.networks.gan.basic.discriminator import Discriminator3D
from typing import List, Tuple, Union
import numpy as np

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.collate_outputs import collate_outputs

class nnUNetTrainerGAN(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 1e-4
        self.initial_dlr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01
        self.num_epochs = 10
        self.lambda_adv = 0.01  # You can tune this
        
        def initialize(self):
            if not self.was_initialized:
                ## DDP batch size and oversampling can differ between workers and needs adaptation
                # we need to change the batch size in DDP because we don't use any of those distributed samplers
                self._set_batch_size_and_oversample()

                self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                    self.dataset_json)

                self.network, self.discriminator = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.configuration_manager.patch_size,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                )
                self.network.to(self.device)
                self.discriminator.to(self.device)
                # compile network for free speedup
                if self._do_i_compile():
                    self.print_to_log_file('Using torch.compile...')
                    self.network = torch.compile(self.network)

                self.optimizer, self.lr_scheduler, self.optimzer_d = self.configure_optimizers()
                # if ddp, wrap in DDP wrapper
                if self.is_ddp:
                    self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                    self.network = DDP(self.network, device_ids=[self.local_rank])

                self.loss = self._build_loss()
                self.bce_loss = nn.BCELoss()

                self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

                # torch 2.2.2 crashes upon compiling CE loss
                # if self._do_i_compile():
                #     self.loss = torch.compile(self.loss)
                self.was_initialized = True
            else:
                raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                                "That should not happen.")
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   patch_size: Tuple[int, ...],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:


        generator = SEUNet3D(in_channels=num_input_channels,out_channels=num_output_channels)
        discriminator = Discriminator3D(in_channels=num_output_channels, base_features=16)
        
        return generator, discriminator
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        #TODO - Use a scheduler 
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(),self.initial_dlr)
        
        return optimizer, lr_scheduler, optimizer_d
    
    def on_train_epoch_start(self):
        # self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    
    def train_step(self,batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        # ====================================================
        # 1. Train Discriminator
        # ====================================================
        self.network.eval()
        self.discriminator.train()
        with torch.no_grad():
            fake_mask = self.network(data)
            
        real_pair = torch.cat([data, target], dim=1)
        fake_pair = torch.cat([data, fake_mask], dim=1)
        
        real_labels = torch.ones_like(self.discriminator(real_pair))
        fake_labels = torch.zeros_like(self.discriminator(fake_pair))
        
        d_real = self.discriminator(real_pair)
        d_fake = self.discriminator(fake_pair)
        
        d_loss_real = self.bce_loss(d_real,real_labels)
        d_loss_fake = self.bce_loss(d_fake, fake_labels)
        
        d_loss = (d_loss_real+d_loss_fake)/2
        
                # Handle Optimizers
        
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optmizer_d.step()
        

        
        # ====================================================
        # 2. Train Generator
        # ====================================================
        
        self.network.train()
        self.discriminator.eval()
        
        fake_mask = self.network(data)
        # segmentation loss
        seg_loss = self.loss(fake_mask,target)
        
        fake_pair = torch.cat([data, fake_mask], dim=1)
        pred_fake = self.discriminator(fake_pair)
        adv_loss = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        
        g_loss = seg_loss + self.lambda_adv * adv_loss
        
        # Handle Optimizers
        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()
        
        return {"loss": g_loss.detach().cpu().numpy(), "dloss": d_loss.detach().cpu().numpy()}
        
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            dloss_here = np.mean(outputs['dloss'])

        self.logger.log(f'generator loss: {loss_here}, discriminator loss: {dloss_here}, epoch: {self.current_epoch}')
        
    def on_validation_epoch_start(self):
        self.network.eval()
        self.discriminator.eval()
        
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            real_pair = torch.cat([data,target], dim=1),
            fake_pair = torch.cat([data, output], dim=1)
            
            douput = self.dicriminator(fake_pair)
            routput = self.dicriminator(real_pair)
            
            del data
            l = self.loss(output, target)
            dl = self.bce_loss(douput,routput)
            

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'generator loss': l.detach().cpu().numpy(), 'discriminator loss': dl.detach().cpu().numpy(),'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}