
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
import multiprocessing
import warnings

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.sliding_window_prediction import compute_gaussian


class nnUNetTrainerColonSeg2(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 1e-4
        self.initial_dlr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01
        self.num_epochs = 2000
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
        generator = SEUNet3D(in_channels=2,out_channels=num_output_channels,feature_channels=[8, 16, 32, 64])
        discriminator = Discriminator3D(in_channels=num_output_channels+2, base_features=16)
        
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
        input_data = batch['data']
        target = batch['target']
        seg = batch['seg']
        
        data = torch.cat((input_data,seg), dim=1)
        del input_data
        del seg
        
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
        input_data = batch['data']
        target = batch['target']
        seg = batch['seg']

        data = torch.cat((input_data,seg), dim=1)
        del input_data
        del seg
        
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

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties, seg2 = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]
                seg2 = seg2[:]
                
                # Combine data and seg
                # print("data",data.shape)
                # print("seg2",seg2.shape)
                

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)
                    seg2 = torch.from_numpy(seg2)
                    data = torch.cat((data,seg2),dim=0)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                #      self.dataset_json, output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        # next stage may have a different dataset class, do not use self.dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
