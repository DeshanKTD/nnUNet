import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.custom.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.custom.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.utils import crop_with_bbox, get_padded_3d_segmentation_box, get_padded_3d_square_xy_segmentation_box, pad_with_all_directions, resize_data, resize_data_with_scaling_factor


class nnUNetDataLoaderFullVolumeWithMultiSeg(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.target_size = (128,128,192)
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties,seg2 = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.target_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.target_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        
        # print(f"data shape: {self.data_shape}, seg shape: {self.seg_shape}, batch size: {self.batch_size}")
        # preallocate memory for data and seg
        # Here they set the patch size [crop with bbox do it for patch size]
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        seg2_all = np.zeros(self.seg_shape, dtype=np.int16)
        disconnection_map_all = np.zeros(self.seg_shape, dtype=np.int16)
        

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties, seg2 = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            # print('Shape of data: ', data.shape)

            # bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            pad = 20
            bbox_lbs, bbox_ubs = get_padded_3d_segmentation_box(seg2[0], pad)
            data_dict = get_padded_3d_square_xy_segmentation_box(seg2[0], self.target_size,pad=pad)
            scaling_ratio = data_dict['scaling_ratio']
           
            scaled_data = resize_data_with_scaling_factor(data, scaling_ratio,order=1)
            scaled_seg = resize_data_with_scaling_factor(seg, scaling_ratio, order=0)
            scaled_seg2 = resize_data_with_scaling_factor(seg2, scaling_ratio, order=0)
            
            padded_target = pad_with_all_directions(scaled_data,data_dict["x_min_pad"],
                                                        data_dict["y_min_pad"], data_dict["z_min_pad"],
                                                        data_dict["x_max_pad"], data_dict["y_max_pad"], data_dict["z_max_pad"])
            padded_seg = pad_with_all_directions(scaled_seg,data_dict["x_min_pad"],
                                                        data_dict["y_min_pad"], data_dict["z_min_pad"],
                                                        data_dict["x_max_pad"], data_dict["y_max_pad"], data_dict["z_max_pad"])
       
            padded_seg2 = pad_with_all_directions(scaled_seg2,data_dict["x_min_pad"],
                                                        data_dict["y_min_pad"], data_dict["z_min_pad"],
                                                        data_dict["x_max_pad"], data_dict["y_max_pad"], data_dict["z_max_pad"])
           
            bbox_lbs = data_dict['bbox_min_padded']
            bbox_ubs = data_dict['bbox_max_padded']
            
            data_cropped = crop_with_bbox(padded_target, bbox_lbs, bbox_ubs)
            seg_cropped = crop_with_bbox(padded_seg, bbox_lbs, bbox_ubs)
            seg2_cropped = crop_with_bbox(padded_seg2, bbox_lbs, bbox_ubs)
           
            del padded_target, padded_seg, padded_seg2, scaled_data, scaled_seg, scaled_seg2
            
            data_all[j] = data_cropped
            seg_all[j] = seg_cropped
            seg2_all[j] = seg2_cropped

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    seg2_all = torch.from_numpy(seg2_all).float()
    
                    images = []
                    segs = []
                    seg2s = []
                    disconnection_maps = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(
                            **{'image': data_all[b], 
                                'segmentation': seg_all[b], 
                                'segmentation_out_1': seg2_all[b],
                            })
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        seg2s.append(tmp['segmentation_out_1'])
                        if 'disconnection_map' in tmp:
                            disconnection_maps.append(tmp['disconnection_map'])
                        else:
                            disconnection_maps.append(torch.zeros_like(tmp['segmentation']))
                        
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        seg2_all = [torch.stack([s[i] for s in seg2s]) for i in range(len(seg2s[0]))]
                        disconnection_map_all = [torch.stack([s[i] for s in disconnection_maps]) for i in range(len(disconnection_maps[0]))]
                    else:
                        seg_all = torch.stack(segs)
                        seg2_all = torch.stack(seg2s)
                        disconnection_map_all = torch.stack(disconnection_maps)
                    del segs, images, seg2s, disconnection_maps
            return {'data': data_all, 'target': seg_all, 'seg': seg2_all, 'disconnection_map': disconnection_map_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all,'seg': seg2_all, 'disconnection_map': disconnection_map_all,  'keys': selected_keys}


if __name__ == '__main__':
    folder = join(nnUNet_preprocessed, 'Dataset002_Heart', 'nnUNetPlans_3d_fullres')
    ds = nnUNetDatasetBlosc2(folder)  # this should not load the properties!
    pm = PlansManager(join(folder, os.pardir, 'nnUNetPlans.json'))
    lm = pm.get_label_manager(load_json(join(folder, os.pardir, 'dataset.json')))
    dl = nnUNetDataLoader(ds, 5, (16, 16, 16), (16, 16, 16), lm,
                          0.33, None, None)
    a = next(dl)
