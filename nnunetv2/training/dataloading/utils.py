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
        
        
def get_padded_3d_square_xy_segmentation_box(segmentation_map, target_size, pad=0):
    """
    Returns a padded, square-in-XY bounding box around the non-zero region of a 3D segmentation map.
    
    The cropped region in X and Y directions will have the same size, centered around the object.

    Parameters:
        segmentation_map (np.ndarray): 3D ndarray (X, Y, Z)
        pad (int): Number of voxels to pad in each direction

    Returns:
        tuple or None: (min_coord, max_coord), where each is a (x, y, z) tuple
    """
    assert segmentation_map.ndim == 3, "Expected a 3D array"
    assert pad >= 0, "Pad must be non-negative"
    
    data_dict = {}

    nonzero_indices = np.argwhere(segmentation_map)
    if nonzero_indices.size == 0:
        return None

    min_coord = nonzero_indices.min(axis=0)
    max_coord = nonzero_indices.max(axis=0)

    shape = segmentation_map.shape
    x,y,z = shape
    min_coord = min_coord.astype(int)
    max_coord = max_coord.astype(int)
    
    print(f"intial shape: {shape}")
    print(f"min_coord_of_foreground: {min_coord}, max_coord_of_foreground: {max_coord}")

    # Apply padding
    min_coord_padded = np.maximum(min_coord - pad, 0)
    max_coord_padded = np.minimum(max_coord + pad, np.array(shape) - 1)
    
    
    center_x = (min_coord_padded[0] + max_coord_padded[0]) // 2
    center_y = (min_coord_padded[1] + max_coord_padded[1]) // 2
    center_z = (min_coord_padded[2] + max_coord_padded[2]) // 2
    
    data_dict['center'] = tuple((center_x, center_y, center_z))
    
    print(f"min_coord_padded: {min_coord_padded}, max_coord_padded: {max_coord_padded}")
    
    tartet_x, target_y, target_z = target_size
    
    x_min, y_min, z_min = min_coord_padded
    x_max, y_max, z_max = max_coord_padded
    x_size = x_max - x_min + 1
    y_size = y_max - y_min + 1
    z_size = z_max - z_min + 1
    
    print(f"length_x_size: {x_size}, length_y_size: {y_size}")
    
    # Select the larger dimension to ensure square cropping
    if x_size >= y_size:
        # assume x is larger, make y equal to x
        
        # resize ratio 
        xy_resize_ratio = tartet_x / x_size
        z_size_after_resize = int(z_size * xy_resize_ratio)
        
        # check scaling for z axis
        if z_size_after_resize < target_z:
            data_dict['scaling_ratio'] = xy_resize_ratio
             
        elif z_size_after_resize > target_size[2]:
            # This is not OK, as we cannot crop removing foreground slices
            # rescale base on z axis
            z_scale = target_size[2] / z_size
            data_dict['scaling_ratio'] = z_scale
        
    elif y_size > x_size:
        # assume y is larger, make x equal to y
        
        # resize ratio 
        xy_resize_ratio = target_y / y_size
        z_size_after_resize = int(z_size * xy_resize_ratio)
        
        # check scaling for z axis
        if z_size_after_resize < target_z:
            data_dict['scaling_ratio'] = xy_resize_ratio
            
        elif z_size_after_resize > target_size[2]:
            # This is not OK, as we cannot crop removing foreground slices
            # rescale base on z axis
            z_scale = target_size[2] / z_size
            data_dict['scaling_ratio'] = z_scale
        
    print(f"scaling ratio: {data_dict['scaling_ratio']}")
        
    scaled_x_shape = int(x * data_dict['scaling_ratio'])
    scaled_y_shape = int(y * data_dict['scaling_ratio'])
    scaled_z_shape = int(z * data_dict['scaling_ratio'])
    
    scaled_center_x = center_x * data_dict['scaling_ratio']
    scaled_center_y = center_y * data_dict['scaling_ratio']
    scaled_center_z = center_z * data_dict['scaling_ratio']
    
    print(f"scaled shape: ({scaled_x_shape}, {scaled_y_shape}, {scaled_z_shape})")
    
    # Calculate new min and max coordinates based on the scaled center
    x_min = int(scaled_center_x - tartet_x // 2)
    x_max = int(scaled_center_x + (tartet_x - 1) // 2)
    y_min = int(scaled_center_y - target_y // 2)
    y_max = int(scaled_center_y + (target_y - 1) // 2)
    z_min = int(scaled_center_z - target_z // 2)
    z_max = int(scaled_center_z + (target_z - 1) // 2)  
    
    data_dict["x_min_pad"] = 0
    data_dict["y_min_pad"] = 0
    data_dict["z_min_pad"] = 0
    data_dict["x_max_pad"] = 0      
    data_dict["y_max_pad"] = 0
    data_dict["z_max_pad"] = 0
    
    # Ensure bounds are respected and all segmentation is included
    if x_min < 0:
        data_dict["x_min_pad"] = abs(x_min)
        x_min = 0
    if y_min < 0:
        data_dict["y_min_pad"] = abs(y_min)
        y_min = 0
    if z_min < 0:
        data_dict["z_min_pad"] = abs(z_min)
        z_min = 0
    if x_max >= scaled_x_shape:
        data_dict["x_max_pad"] = x_max - (scaled_x_shape - 1)
        x_max = scaled_x_shape - 1
    if y_max >= scaled_y_shape:
        data_dict["y_max_pad"] = y_max - (scaled_y_shape - 1)
        y_max = scaled_y_shape - 1
    if z_max >= scaled_z_shape:
        data_dict["z_max_pad"] = z_max - (scaled_z_shape - 1)
        z_max = scaled_z_shape - 1

    # Store the min and max before padding
    data_dict["x_min"] = x_min
    data_dict["y_min"] = y_min
    data_dict["z_min"] = z_min
    data_dict["x_max"] = x_max
    data_dict["y_max"] = y_max
    data_dict["z_max"] = z_max
    
    # reset scaled center after adding padding
    x_center_padded = scaled_center_x + data_dict["x_min_pad"]
    y_center_padded = scaled_center_y + data_dict["y_min_pad"]
    z_center_padded = scaled_center_z + data_dict["z_min_pad"]
    
    # bounding box coordinates after padding
    xb_min = int(x_center_padded - tartet_x // 2)
    xb_max = int(x_center_padded + (tartet_x - 1) // 2)
    yb_min = int(y_center_padded - target_y // 2)
    yb_max = int(y_center_padded + (target_y - 1) // 2)
    zb_min = int(z_center_padded - target_z // 2)
    zb_max = int(z_center_padded + (target_z - 1) // 2)
    
    # Store the final padded min and max coordinates
    min_coord_padded = (xb_min, yb_min, zb_min)
    max_coord_padded = (xb_max, yb_max, zb_max)
    
    # bounding box coordinates after padding
    data_dict["bbox_min_padded"] = min_coord_padded
    data_dict["bbox_max_padded"] = max_coord_padded
    
    print(f"padded min_coord: {min_coord_padded}, padded max_coord: {max_coord_padded}")
    
        
    # remove padding
    xc_min = data_dict["x_min_pad"]
    yc_min = data_dict["y_min_pad"]
    zc_min = data_dict["z_min_pad"]
    xc_max = data_dict["x_min_pad"] + scaled_x_shape 
    yc_max = data_dict["y_min_pad"] + scaled_y_shape
    zc_max = data_dict["z_min_pad"] + scaled_z_shape
    
    data_dict["min_removed_pad"] = (xc_min, yc_min, zc_min)
    data_dict["max_removed_pad"] = (xc_max, yc_max, zc_max)
    
    print(f"min_removed_pad: {data_dict['min_removed_pad']}, max_removed_pad: {data_dict['max_removed_pad']}")
    
    return data_dict

   

def get_padded_3d_segmentation_box(segmentation_map, pad=0):
    """
    Returns the min and max coordinates of the non-zero region in a 3D segmentation map,
    padded by a given number of voxels in each direction (with bounds checking).
    
    Parameters:
        segmentation_map (np.ndarray): 3D ndarray (X, Y, Z)
        pad (int): Number of voxels to pad around the bounding box
    
    Returns:
        tuple or None: (min_coord, max_coord), where each is a (x, y, z) tuple
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
    x_min, y_min, z_min  = min_coord
    x_max, y_max, z_max = max_coord
    
    cropped = segmentation_map[:, x_min:x_max+1,y_min:y_max+1, z_min:z_max+1]
    return cropped

def resize_data(data: np.ndarray, target_shape: Tuple[int, ...],order: int =1) -> np.ndarray:
    assert data.ndim == 4, "Expected input of shape (C, X, Y, Z)"
    assert len(target_shape) == 3, "target_size should be (X, Y, Z)"
    
    c, x, y, z = data.shape
    target_x, target_y , target_z,  = target_shape
    
    zoom_factors = ( target_x / x, target_y / y,  target_z / z)
    
    resized = np.stack([
        zoom(data[i], zoom_factors, order=order)  # linear interpolation
        for i in range(c)
    ])
    
    return resized

if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')