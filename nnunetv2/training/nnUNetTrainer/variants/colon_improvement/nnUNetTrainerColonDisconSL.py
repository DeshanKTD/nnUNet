
import torch
from torch import autocast
import torch.nn as nn
from torch import distributed as dist
# from network_architectures.networks.auto_encoder.basic.auto_encoder import AutoEncoder
from network_architectures.networks.unet.se_unet.se_unet_3d import SEUNet3D
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from typing import List, Tuple, Union
import numpy as np
from time import time, sleep
import multiprocessing
import warnings
import sys

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.data_augmentation.custom_transforms.colon.random_disconnection_transform import RandomDisconnectionsTransform
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
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.dataloading.utils import crop_with_bbox, get_padded_3d_segmentation_box, resize_data


class nnUNetTrainerColonDisconSL(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.initial_lr = 1e-4
        self.initial_dlr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01
        self.num_epochs = 1000
        self.lambda_adv = 0.01  # You can tune this
        self.configuration_manager.configuration['patch_size'] = [192,192,192]
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   patch_size: Tuple[int, ...],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        print("number of output channels: ", num_output_channels)
        model = SEUNet3D(in_channels=2, out_channels=num_output_channels,feature_channels=[8,16,32,64])
        
        return model
    
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
            

        transforms.append(
                RandomDisconnectionsTransform(min_rad=10, max_rad=40, fill_voxels=15000)
        )

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        
        transforms.append(
                RandomDisconnectionsTransform(min_rad=10, max_rad=40, fill_voxels=15000)
        )
        
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    
    def on_train_epoch_start(self):
        # self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    
    def train_step(self,batch: dict) -> dict:
        input_img = batch['data']
        target = batch['target']
        disconnection_map = batch['disconnection_map']
        
        data = torch.cat([input_img, disconnection_map], dim=1)
        
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        data = data.float()
        
        print(f"data shape: {data.shape}, target shape: {target.shape}")
        
        # set generator to train and discriminator to eval
        self.network.train()
       
        fake_mask = self.network(data)
        
        # get segmentation loss
        l = self.loss(fake_mask,target)
        
        # Handle Optimizers
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        
        return {"loss": l.detach().cpu().numpy()}
        # return {"loss": 0.0}
        
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        # self.logger.log(f'generator loss: {loss_here}, discriminator loss: {dloss_here}, epoch: {self.current_epoch}')
        self.logger.log('train_losses', loss_here, self.current_epoch)
        
    def on_validation_epoch_start(self):
        self.network.eval()

        
    def validation_step(self, batch: dict) -> dict:
        input_data = batch['data']
        target = batch['target']
        disconnection_map = batch['disconnection_map']
        # disconnection_map = batch['disconnection_map']

        # combine input data and disconnection map
        data = torch.cat([input_data, disconnection_map], dim=1)
        
        # assgin device
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        data = data.float()

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
            
        l = self.loss(output, target)
        

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
        return {'loss': l.detach().cpu().numpy(),'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        # return {'loss': 0, 'dloss': 0,'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
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

        # predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        #                             perform_everything_on_device=True, device=self.device, verbose=False,
        #                             verbose_preprocessing=False, allow_tqdm=False)
        # predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
        #                                 self.dataset_json, self.__class__.__name__,
        #                                 self.inference_allowed_mirroring_axes)

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
                
                # self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                print(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                # prediction = predictor.predict_sliding_window_return_logits(data)
                
                # orignal shape
                original_shape = data.shape[1:]
                
                # implement prediction pipeline
                pad = 20
                target_size = (192,128,128)
                bbox_lbs, bbox_ubs = get_padded_3d_segmentation_box(seg2[0], pad)
                seg_cropped = crop_with_bbox(seg2, bbox_lbs, bbox_ubs)
                data_cropped = crop_with_bbox(data, bbox_lbs, bbox_ubs)
                seg_cropped_shape = seg_cropped.shape[1:]
                seg2_resized = resize_data(seg_cropped,target_size)
                data_resized = resize_data(data_cropped,target_size)
                
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    seg2 = torch.from_numpy(seg2_resized)
                    seg2 = seg2.unsqueeze(0).float()  # add batch dimension
                    data = torch.from_numpy(data_resized)
                    data = data.unsqueeze(0).float()  # add batch dimension
                    
                    data = torch.cat([data, seg2], dim=1)
                    data = data.to(self.device, non_blocking=True)
                    
                    data = data.float()
                    
                # print(f"seg2 shape: {seg2.shape}, original shape: {original_shape}, seg_cropped_shape: {seg_cropped_shape}")

                self.network.eval()
                with torch.no_grad():
                    # we do not use autocast here because it is not supported by DDP
                    prediction = self.network(data)
                
                prediction = prediction.cpu().numpy()
                prediction = np.squeeze(prediction, axis=0)  # remove batch dimension
                
                # resize to cropped shape
                prediction = resize_data(prediction, seg_cropped_shape)
                
                output = np.zeros((
                    prediction.shape[0],
                    original_shape[0],
                    original_shape[1],
                    original_shape[2]), 
                                  dtype=np.float32)
                
                output[:, bbox_lbs[0]:bbox_ubs[0]+1, bbox_lbs[1]:bbox_ubs[1]+1, bbox_lbs[2]:bbox_ubs[2]+1] = prediction
                
                output = torch.from_numpy(output)
                output = output.unsqueeze(0).float()
                
                
                print(f"Prediction shape: {prediction.shape}, Output shape: {output.shape}, ")
                
                
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

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
