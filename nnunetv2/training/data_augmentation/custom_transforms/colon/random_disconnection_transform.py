from typing import Tuple

import torch
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, dilation
from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage.morphology import ball
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class RandomDisconnectionsTransform(BasicTransform):
    def __init__(self):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()

    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()

        bin_seg = (seg_all > 0).astype(np.int16)  #
        disconnection_map = np.zeros_like(bin_seg, dtype=np.int16)
        
        if np.sum(bin_seg[0]) == 0:
            data_dict["disconnection_map"] = disconnection_map
            return data_dict
        
        # Ensure the input is a 3D binary mask
        skel = skeletonize(bin_seg[0].astype(bool))
        
        random_point = self._get_random_voxel_from_skeleton(skel)
        # random_point = [102,41,204]
        
        # randomly create a blob around the random point
        shape = bin_seg.shape[1:]  # Exclude batch dimension
        steps = np.random.randint(10,15)
        
        # Select random blob type
        blob_type = np.random.choice(['rectangle', 'circular', 'irregular'])
        if blob_type == 'rectangle':
            removing_blob = self._create_random_rectangle(random_point, shape)
        elif blob_type == 'circular':
            removing_blob = self._create_random_circular_blob(random_point, shape)
        else:  # irregular shape
            removing_blob = self._create_random_blob(random_point, shape)

        
        # add batch dimession to the removing blob
        removing_blob = removing_blob[np.newaxis, ...]
        
        # create disconnection map
        disconnection_map = bin_seg.astype(np.int16)
        disconnection_map[removing_blob] = 0
        del removing_blob
        disconnection_map = disconnection_map.astype(np.int16)
        data_dict["disconnection_map"] = torch.from_numpy(disconnection_map)
        
        # Add skeleton to the data_dict
        skel = skel[np.newaxis, ...]  # Add batch dimension
        data_dict["skel"] = torch.from_numpy(skel.astype(np.int16))
        
        return data_dict
        
    def _get_random_voxel_from_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Returns a random voxel coordinate from a binary 3D skeleton mask.

        Args:
            skeleton (np.ndarray): Binary 3D skeleton mask (0/1)

        Returns:
            np.ndarray: (x, y, z) coordinate of a random voxel, or None if empty
        """
        skel_coords = np.argwhere(skeleton > 0)

        if skel_coords.shape[0] == 0:
            return None  # or raise error

        random_index = np.random.randint(0, len(skel_coords))
        return skel_coords[random_index]
        
    def _create_random_blob(self, seed_point: np.ndarray, shape: Tuple[int], max_size: int = 150) -> np.ndarray:
        """
        Creates a randomly shaped, filled blob around a seed point by region growing.

        Args:
            seed_point (np.ndarray): Starting (x, y, z) point.
            shape (tuple): Shape of the 3D volume.
            max_size (int): Max number of voxels in the blob.

        Returns:
            np.ndarray: Binary 3D blob mask.
        """
        blob = np.zeros(shape, dtype=bool)
        visited = np.zeros(shape, dtype=bool)
        
        queue = deque()
        queue.append(tuple(seed_point))
        count = 0
        
        # 6-connected neighbors
        directions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

        while queue and count < 20000:
            current = queue.popleft()
            x, y, z = current

            if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                continue
            if visited[x, y, z]:
                continue

            visited[x, y, z] = True
            blob[x, y, z] = True
            count += 1

            # Shuffle directions to introduce randomness
            np.random.shuffle(directions)

            for d in directions:
                neighbor = (x + d[0], y + d[1], z + d[2])
                if (
                    0 <= neighbor[0] < shape[0] and
                    0 <= neighbor[1] < shape[1] and
                    0 <= neighbor[2] < shape[2] and
                    not visited[neighbor]
                ):
                    if np.random.rand() < 0.8:  # adjust growth randomness
                        queue.append(neighbor)

        return blob
    
    def _create_random_rectangle(self, seed_point: np.ndarray, shape: Tuple[int], 
                             max_extent: Tuple[int, int, int] = (70, 70, 70)) -> np.ndarray:
        """
        Creates a randomly sized 3D rectangular block around a seed point.

        Args:
            seed_point (np.ndarray): Starting point (x, y, z).
            shape (tuple): Shape of the 3D volume (Dx, Dy, Dz).
            max_extent (tuple): Max size (dx, dy, dz) of the rectangle in each direction.

        Returns:
            np.ndarray: Binary 3D mask of the rectangle.
        """
        blob = np.zeros(shape, dtype=bool)
        x, y, z = seed_point

        # Random sizes (at least 1, up to max_extent)
        size_x = np.random.randint(20, max_extent[0])
        size_y = np.random.randint(20, max_extent[1])
        size_z = np.random.randint(20, max_extent[2])

        # Compute bounding box
        x0 = max(0, x - size_x // 2)
        x1 = min(shape[0], x + (size_x + 1) // 2)
        y0 = max(0, y - size_y // 2)
        y1 = min(shape[1], y + (size_y + 1) // 2)
        z0 = max(0, z - size_z // 2)
        z1 = min(shape[2], z + (size_z + 1) // 2)

        # Fill the region
        blob[x0:x1, y0:y1, z0:z1] = 1
        return blob
        
    
    def _create_random_circular_blob(self, seed_point: np.ndarray, shape: Tuple[int], 
                                  max_radius: Tuple[int, int, int] = (35, 35, 35)) -> np.ndarray:
        """
        Creates a randomly sized 3D ellipsoidal blob around a seed point.

        Args:
            seed_point (np.ndarray): Center point (x, y, z).
            shape (tuple): 3D volume shape (Dx, Dy, Dz).
            max_radius (tuple): Maximum radii in each dimension (rx, ry, rz).

        Returns:
            np.ndarray: Binary 3D blob mask.
        """
        blob = np.zeros(shape, dtype=bool)
        cx, cy, cz = seed_point

        # Random radii in each axis (at least 1)
        rx = np.random.randint(15, max_radius[0])
        ry = np.random.randint(15, max_radius[1])
        rz = np.random.randint(15, max_radius[2])

        # Bounding box
        x0 = max(0, cx - rx)
        x1 = min(shape[0], cx + rx + 1)
        y0 = max(0, cy - ry)
        y1 = min(shape[1], cy + ry + 1)
        z0 = max(0, cz - rz)
        z1 = min(shape[2], cz + rz + 1)

        # Grid of coordinates within the bounding box
        x, y, z = np.ogrid[x0:x1, y0:y1, z0:z1]

        # Ellipsoid equation
        ellipsoid = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1

        # Assign to blob
        blob[x0:x1, y0:y1, z0:z1] = ellipsoid

        return blob
  