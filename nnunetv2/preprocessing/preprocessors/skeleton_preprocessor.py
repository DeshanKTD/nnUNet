import multiprocessing
import shutil
from time import sleep
from typing import List, Union

import blosc2
import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json, maybe_mkdir_p
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class SkeletonPreprocessor(DefaultPreprocessor):
    """
    Same as DefaultPreprocessor, but additionally preprocesses a skeleton mask per training case.

    Skeletons are read from nnUNet_raw/DatasetXXX/skeletonTr/<case><file_ending> (same naming as labelsTr) and
    undergo exactly the same transpose -> crop (image bbox) -> resample as the label. They are saved as
    <case>_seg2.b2nd next to <case>.b2nd / <case>_seg.b2nd, which is what nnUNetDatasetBlosc2MultiSeg loads.

    Select it by setting "preprocessor_name": "SkeletonPreprocessor" in the plans configuration.
    """
    skeleton_folder_name = 'skeletonTr'
    skeleton_suffix = '_seg2'
    # radius (in raw voxels) of the spherical dilation applied to the skeleton before preprocessing. 0 disables it.
    skeleton_dilation_radius = 2

    @staticmethod
    def _dilate(skel: np.ndarray, radius: int) -> np.ndarray:
        """
        skel: (C, X, Y, Z) binary. Dilates every channel with a ball of the given radius (in voxels).
        2D data is read as (C, 1, X, Y); the ball then effectively acts in-plane only.
        """
        if radius <= 0:
            return skel
        r = np.arange(-radius, radius + 1)
        ball = (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2) <= radius ** 2
        return np.stack([binary_dilation(c, structure=ball) for c in skel]).astype(skel.dtype)

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str], skel: Union[np.ndarray, None] = None):
        data, seg, properties = super().run_case_npy(data, seg, properties, plans_manager, configuration_manager,
                                                     dataset_json)
        if skel is None:
            return data, seg, properties

        skel = (np.asarray(skel) > 0).astype(np.int8)  # binarize: loader expects skeleton == 1

        # thicken the 1-voxel line so it survives resampling without gaps (done in raw voxel space)
        skel = self._dilate(skel, self.skeleton_dilation_radius)

        # transpose (same as parent)
        skel = skel.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        assert skel.shape[1:] == tuple(properties['shape_before_cropping']), \
            f"Skeleton shape {skel.shape[1:]} does not match image shape {properties['shape_before_cropping']}"

        # crop with the exact bbox used for the image
        slicer = (slice(None),) + bounding_box_to_slice(properties['bbox_used_for_cropping'])
        skel = skel[slicer]

        # resample to the shape of the preprocessed image (spacings computed as in the parent)
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(skel.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        skel = configuration_manager.resampling_fn_seg(skel, data.shape[1:], original_spacing, target_spacing)
        skel = skel.astype(np.int8)

        return data, seg, skel, properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager, dataset_json: Union[dict, str],
                 skel_file: Union[str, None] = None):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)
        seg = rw.read_seg(seg_file)[0] if seg_file is not None else None
        skel = rw.read_seg(skel_file)[0] if skel_file is not None else None

        if self.verbose:
            print(seg_file, skel_file)
        return self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager, dataset_json,
                                 skel=skel)

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str], skel_file: str = None):
        out = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json,
                            skel_file=skel_file)
        if skel_file is not None:
            data, seg, skel, properties = out
        else:
            (data, seg, properties), skel = out, None

        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16, copy=False)
        patch_size = tuple(configuration_manager.patch_size)
        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(data.shape, patch_size,
                                                                                  data.itemsize)
        block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(seg.shape, patch_size, seg.itemsize)

        nnUNetDatasetBlosc2.save_case(data, seg, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg)

        if skel is not None:
            # same dtype / compression as the regular _seg.b2nd
            skel = skel.astype(np.int16, copy=False)
            block_size_skel, chunk_size_skel = nnUNetDatasetBlosc2.comp_blosc2_params(skel.shape, patch_size,
                                                                                      skel.itemsize)
            blosc2.set_nthreads(1)
            blosc2.asarray(np.ascontiguousarray(skel),
                           urlpath=output_filename_truncated + self.skeleton_suffix + '.b2nd',
                           chunks=chunk_size_skel, blocks=block_size_skel,
                           cparams={'codec': blosc2.Codec.ZSTD, 'clevel': 8})

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans_manager = PlansManager(load_json(plans_file))
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
            print(configuration_manager)

        dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset.json'))
        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # locate skeletons and fail early if any are missing
        skeleton_folder = join(nnUNet_raw, dataset_name, self.skeleton_folder_name)
        skel_files = {k: join(skeleton_folder, k + dataset_json['file_ending']) for k in dataset.keys()}
        missing = [k for k, f in skel_files.items() if not isfile(f)]
        if len(missing) > 0:
            raise FileNotFoundError(f'{len(missing)} skeleton file(s) missing in {skeleton_folder}, e.g. '
                                    f'{missing[:5]}. Expected <case>{dataset_json["file_ending"]}')

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)
        if isdir(output_directory):
            shutil.rmtree(output_directory)
        maybe_mkdir_p(output_directory)

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            workers = [j for j in p._pool]
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager, dataset_json, skel_files[k]),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    if not all([j.is_alive() for j in workers]):
                        raise RuntimeError('One of the background workers died. This is usually caused by an error '
                                           '(look for an error message above) or by running out of RAM. If you '
                                           'don\'t see an error message, try reducing the number of processes.')
                    done = [i for i in remaining if r[i].ready()]
                    for i in done:
                        r[i].get()  # re-raises errors from workers
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
