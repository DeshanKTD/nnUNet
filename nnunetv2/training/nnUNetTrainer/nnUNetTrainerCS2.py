
import torch
from torch import autocast
import torch.nn as nn
from torch import distributed as dist
from network_architectures.networks.unet.se_unet.se_unet_3d import SEUNet3D
from network_architectures.networks.gan.basic.discriminator import Discriminator3D
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from typing import List, Tuple, Union
import numpy as np
from time import time, sleep

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.collate_outputs import collate_outputs

class nnUNetTrainerColonSeg2(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 1e-4
        self.initial_dlr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01
        self.num_epochs = 1000
        self.lambda_adv = 0.01  # You can tune this
        self.configuration_manager.configuration['patch_size'] = [128,128,128]
        
        
    def initialize(self):
            print('......... on initialiaze .........................')
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

                self.optimizer, self.lr_scheduler, self.optimizer_d, self.dlr_scheudler = self.configure_optimizers()
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

        print("number of output channels: ", num_output_channels)
        generator = SEUNet3D(in_channels=num_input_channels,out_channels=num_output_channels,feature_channels=[8, 16, 32, 64])
        discriminator = Discriminator3D(in_channels=num_output_channels+1, base_features=16)
        
        return generator, discriminator
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        #TODO - Use a scheduler 
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(),self.initial_dlr)
        dlr_scheudler = PolyLRScheduler(optimizer_d, self.initial_dlr, self.num_epochs)
        
        return optimizer, lr_scheduler, optimizer_d, dlr_scheudler
    
    def on_train_epoch_start(self):
        # self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.dlr_scheudler.step(self.current_epoch)
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
        
        # set generator to eval and discriminator to train
        self.network.eval()
        self.discriminator.train()
        
        # get the network output
        with torch.no_grad():
            fake_mask = self.network(data)
        
        # generally target is not one hot encoded. Convert it to one hot encoding
        if fake_mask.shape == target.shape:
            # if this is the case then gt is probably already a one hot encoding
            target_onehot = target
        else:
            target_onehot = torch.zeros(fake_mask.shape, device=fake_mask.device, dtype=torch.bool)
            target_onehot.scatter_(1, target.long(), 1)
        
        # Create the image and output pair for the discriminator
        real_pair = torch.cat([data, target_onehot], dim=1)
        fake_pair = torch.cat([data, fake_mask], dim=1)
        
        # get discriminaotr output for real and fake pair
        d_real = self.discriminator(real_pair)
        d_fake = self.discriminator(fake_pair)
        
        # genereate targets for real and fake pair outputs from the discriminator
        real_labels = torch.ones_like(d_real)
        fake_labels = torch.zeros_like(d_fake)
        
        # calculate the loss with the ouput and target
        d_loss_real = self.bce_loss(d_real,real_labels)
        d_loss_fake = self.bce_loss(d_fake, fake_labels)
        
        d_loss = (d_loss_real+d_loss_fake)/2
        
        # Handle Optimizers
        # loss backwards 
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()
        

        
        # ====================================================
        # 2. Train Generator
        # ====================================================
        
        # set generator to train and discriminator to eval
        self.network.train()
        self.discriminator.eval()
        
        # In here target automatically handles by nnUNet
        fake_mask = self.network(data)
        
        # get segmentation loss
        seg_loss = self.loss(fake_mask,target)
        
        # Generate fake pair
        fake_pair = torch.cat([data, fake_mask], dim=1)
        # Get discriminator output
        with torch.no_grad():
            pred_fake = self.discriminator(fake_pair)
        
        # Get adversarial loss --> forcing generator to predict ones for fake pair
        adv_loss = self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        
        # Compound loss
        # TODO - Check loss
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

        # self.logger.log(f'generator loss: {loss_here}, discriminator loss: {dloss_here}, epoch: {self.current_epoch}')
        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('disc_losses', dloss_here, self.current_epoch)
        
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
        # with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        output = self.network(data)
            
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("Output contains NaN or Inf values")
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target contains NaN or Inf values")
            
        # generally target is not one hot encoded. Convert it to one hot encoding
        if output.shape == target.shape:
            # if this is the case then gt is probably already a one hot encoding
            target_onehot = target
        else:
            target_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.bool)
            target_onehot.scatter_(1, target.long(), 1)
        
        real_pair = torch.cat([data,target_onehot], dim=1)
        fake_pair = torch.cat([data, output], dim=1)
        
        douput = self.discriminator(fake_pair)
        routput = self.discriminator(real_pair)
        
        del data
        
        
        # target = target.long()
        # if target.shape[1] == 1:
        #     target = target[:, 0]

        # output = output.float() 

        l = self.loss(output, target)
        dl = self.bce_loss(douput,routput)
        
        # print("Loss fn:", self.loss)
        # print("Output dtype/shape/min/max:", output.dtype, output.shape, output.min().item(), output.max().item())
        # print("Target dtype/shape/min/max:", target.dtype, target.shape, target.min().item(), target.max().item())
        
        # print('val seg loss -----', l)

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
        mask=None

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

        # TODO - Discriminator loss is not showing
        return {'loss': l.detach().cpu().numpy(), 'dloss': dl.detach().cpu().numpy(),'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        
        loss_here = np.mean(outputs_collated['loss'])
        dloss_here = np.mean(outputs_collated['dloss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_disc_losses',dloss_here,self.current_epoch)
        
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('generator train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('discriminator train_loss', np.round(self.logger.my_fantastic_logging['disc_losses'][-1], decimals=4))
        self.print_to_log_file('generator val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('discriminator val_loss', np.round(self.logger.my_fantastic_logging['val_disc_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
