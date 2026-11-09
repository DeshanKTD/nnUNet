import multiprocessing
import warnings
from time import sleep
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import distributed as dist
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.dataloading.multiple_segmentation.data_loader_full_vol import \
    nnUNetDataLoaderFullVolumeWithMultiSeg
from nnunetv2.training.dataloading.multiple_segmentation.nnunet_dataset_seg2 import nnUNetDatasetBlosc2MultiSeg
from nnunetv2.training.dataloading.utils import crop_with_bbox, get_padded_3d_square_xy_segmentation_box, \
    pad_with_all_directions, resize_data, resize_data_with_scaling_factor
from nnunetv2.training.nnUNetTrainer.variants.colon_improvement.nnUNetTrainerSkelTest import nnUNetTrainerSkelTest
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


class nnUNetTrainerSkelFullVol(nnUNetTrainerSkelTest):
    """
    Image + skeleton (2 input channels) -> colon segmentation, trained on the whole colon region.

    Each case is cropped to the bounding box of the skeleton (+ self.fullvol_pad voxels), isotropically scaled to fit
    self.fullvol_target_size and padded/cropped to exactly that size (nnUNetDataLoaderFullVolumeWithMultiSeg).
    Validation applies the same crop/scale, predicts in one forward pass and maps the prediction back.

    Network, optimizer, loss, train_step and validation_step are inherited from nnUNetTrainerSkelTest.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.fullvol_target_size = (128, 128, 192)
        # margin around the skeleton bounding box (in preprocessed voxels, before scaling). The skeleton is the
        # centreline, so this must be at least the max colon radius or the colon wall gets cut off.
        self.fullvol_pad = 20

    # ---------------------------------------------------------------- augmentation
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
        # no SpatialTransform: it would crop the full volume down to patch_size.
        # MirrorTransform flips image, target and skeleton (segmentation_out_1) together.
        transforms = []
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        transforms.append(RemoveLabelTansform(-1, 0))
        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    # ---------------------------------------------------------------- data loading
    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # patch_size args are required by the signature but unused by the full-volume loader
        dl_kwargs = dict(label_manager=self.label_manager, sampling_probabilities=None, pad_sides=None,
                         oversample_foreground_percent=self.oversample_foreground_percent,
                         probabilistic_oversampling=self.probabilistic_oversampling,
                         target_size=self.fullvol_target_size, pad=self.fullvol_pad)
        dl_tr = nnUNetDataLoaderFullVolumeWithMultiSeg(dataset_tr, self.batch_size, patch_size, patch_size,
                                                       transforms=tr_transforms, **dl_kwargs)
        dl_val = nnUNetDataLoaderFullVolumeWithMultiSeg(dataset_val, self.batch_size, patch_size, patch_size,
                                                        transforms=val_transforms, **dl_kwargs)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val, transform=None,
                                                      num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # ---------------------------------------------------------------- training loop
    def run_training(self):
        # skip nnUNetTrainerSkelTest.run_training (loop disabled) and use the standard nnU-Net loop
        super(nnUNetTrainerSkelTest, self).run_training()

    # ---------------------------------------------------------------- full-volume inference
    @torch.no_grad()
    def predict_full_volume(self, data: np.ndarray, skel: np.ndarray) -> torch.Tensor:
        """
        data: (C, X, Y, Z) preprocessed image, skel: (1, X, Y, Z) preprocessed skeleton.
        Returns logits (num_classes, X, Y, Z) in the preprocessed image space.
        Same crop/scale/pad as nnUNetDataLoaderFullVolumeWithMultiSeg, then inverted.
        """
        original_shape = data.shape[1:]
        box = get_padded_3d_square_xy_segmentation_box(skel[0], self.fullvol_target_size, self.fullvol_pad)
        if box is None:
            raise RuntimeError('Empty skeleton, cannot determine the region to predict')
        pads = (box["x_min_pad"], box["y_min_pad"], box["z_min_pad"],
                box["x_max_pad"], box["y_max_pad"], box["z_max_pad"])
        lbs, ubs = box['bbox_min_padded'], box['bbox_max_padded']

        padded_data = pad_with_all_directions(resize_data_with_scaling_factor(data, box['scaling_ratio'], order=1),
                                              *pads)
        padded_skel = pad_with_all_directions(resize_data_with_scaling_factor(skel, box['scaling_ratio'], order=0),
                                              *pads)
        padded_shape = padded_skel.shape[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # non-writable numpy array
            net_in = torch.cat([torch.from_numpy(crop_with_bbox(padded_data, lbs, ubs)).float(),
                                torch.from_numpy(crop_with_bbox(padded_skel, lbs, ubs)).float()], dim=0)
        prediction = self.network(net_in[None].to(self.device, non_blocking=True))[0].cpu().numpy()

        # put back into the padded, scaled volume; outside the box stays 0 logits for every class
        output = np.zeros((prediction.shape[0], *padded_shape), dtype=np.float32)
        output[:, lbs[0]:ubs[0] + 1, lbs[1]:ubs[1] + 1, lbs[2]:ubs[2] + 1] = prediction
        s, e = box['min_removed_pad'], box['max_removed_pad']
        output = output[:, s[0]:e[0] + 1, s[1]:e[1] + 1, s[2]:e[2] + 1]
        return torch.from_numpy(resize_data(output, original_shape, order=0))

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.load_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
        self.network.eval()

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDatasetBlosc2MultiSeg(self.preprocessed_dataset_folder, val_keys,
                                                      folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            results = []
            for i, k in enumerate(dataset_val.identifiers):
                while check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                   allowed_num_queued=2):
                    sleep(0.1)

                self.print_to_log_file(f"predicting {k}")
                data, _, _, properties, seg2 = dataset_val.load_case(k)
                data, skel = data[:], seg2[:]  # blosc2 -> numpy
                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')

                prediction = self.predict_full_volume(data, skel)

                results.append(segmentation_export_pool.starmap_async(
                    export_prediction_from_logits,
                    ((prediction, properties, self.configuration_manager, self.plans_manager,
                      self.dataset_json, join(validation_output_folder, k), save_probabilities),)
                ))

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
