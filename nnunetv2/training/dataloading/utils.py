from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes

from scipy.ndimage import zoom


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify] * len(npz_files))
                  )


def get_padded_3d_segmentation_box(segmentation_map, pad=0):
    """
    Returns the min and max coordinates of the non-zero region in a 3D segmentation map,
    padded by a given number of voxels in each direction (with bounds checking).
    
    Parameters:
        segmentation_map (np.ndarray): 3D ndarray (Z, Y, X)
        pad (int): Number of voxels to pad around the bounding box
    
    Returns:
        tuple or None: (min_coord, max_coord), where each is a (z, y, x) tuple
    """
    assert segmentation_map.ndim == 3, "Expected a 3D array"
    assert pad >= 0, "Pad must be non-negative"

    nonzero_indices = np.argwhere(segmentation_map)
    if nonzero_indices.size == 0:
        return None

    min_coord = nonzero_indices.min(axis=0)
    max_coord = nonzero_indices.max(axis=0)

    # Pad while keeping within array bounds
    shape = segmentation_map.shape
    min_padded = np.maximum(min_coord - pad, 0)
    max_padded = np.minimum(max_coord + pad, np.array(shape) - 1)

    return tuple(min_padded), tuple(max_padded)

def crop_with_bbox(segmentation_map, min_coord, max_coord):
    z_min, y_min, x_min = min_coord
    z_max, y_max, x_max = max_coord
    
    cropped = segmentation_map[:,z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    return cropped

def resize_data(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    assert data.ndim == 4, "Expected input of shape (C, Z, Y, X)"
    assert len(target_shape) == 3, "target_size should be (Z, Y, X)"
    
    c, z, y, x = data.shape
    target_z, target_y, target_x = target_shape
    
    zoom_factors = (target_z / z, target_y / y, target_x / x)
    
    resized = np.stack([
        zoom(data[i], zoom_factors, order=1)  # linear interpolation
        for i in range(c)
    ])
    
    return resized

if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')