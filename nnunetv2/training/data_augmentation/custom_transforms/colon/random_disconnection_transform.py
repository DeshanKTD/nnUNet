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
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        disconnection_map = np.zeros_like(bin_seg, dtype=np.int16)
        
        if np.sum(bin_seg[0]) == 0:
            data_dict["disconnection_map"] = torch.from_numpy(disconnection_map)
            return data_dict
        
        # Create skeleton
        skel = skeletonize(bin_seg[0])
        
        # get orderd coordinates of the longest path in the skeleton
        skel_coords = self._get_ordered_longest_path_coords(skel)
        
        if len(skel_coords) == 0:
            data_dict["disconnection_map"] = torch.from_numpy(disconnection_map)
            return data_dict
        
        # select a random point in the skeleton from 0.25 to 0.75 of the path length
        path_length = len(skel_coords)
        start_idx = int(0.25 * path_length)
        end_idx = int(0.75 * path_length)
        random_idx = np.random.randint(start_idx, end_idx)
        random_point = skel_coords[random_idx]
        
        # randomly create a blob around the random point
        shape = bin_seg.shape[1:]  # Exclude batch dimension
        steps = np.random.randint(1,7)
        removing_blob = self._create_random_blob(random_point, shape, max_radius=5, steps=steps)
        
        # add batch dimession to the removing blob
        removing_blob = removing_blob[np.newaxis, ...]
        
        # create disconnection map
        disconnection_map = bin_seg.astype(np.int16)
        disconnection_map[removing_blob] = 0
        disconnection_map = disconnection_map.astype(np.int16)
        data_dict["disconnection_map"] = torch.from_numpy(disconnection_map)
        
        # Add skeleton to the data_dict
        skel = skel[np.newaxis, ...]  # Add batch dimension
        data_dict["skel"] = torch.from_numpy(skel.astype(np.int16))
        
        return data_dict
        
        
        
    def _create_random_blob(seed_point: np.ndarray, shape: Tuple[int], max_radius: int = 5, steps: int = 5) -> np.ndarray:
        """
        Grows a random blob around a seed point.

        Args:
            seed_point (np.ndarray): (x, y, z) starting point
            shape (tuple): 3D shape of the volume
            max_radius (int): Max dilation radius
            steps (int): Number of dilation steps with randomness

        Returns:
            np.ndarray: Binary mask of the blob (same shape as volume)
        """
        blob = np.zeros(shape, dtype=bool)
        x, y, z = seed_point
        if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
            return blob  # Out of bounds

        blob[x, y, z] = 1
        struct = ball(radius=1)

        for _ in range(steps):
            # Dilate and randomly keep only a fraction
            blob = binary_dilation(blob, structure=struct)
            rand_mask = np.random.rand(*shape) > 0.3  # adjust randomness
            blob = np.logical_and(blob, rand_mask)

            # Stop growing if nothing left
            if blob.sum() == 0:
                break

        return blob
        
    
    def _get_ordered_longest_path_coords(skel: np.ndarray) -> np.ndarray:
        """
        Extracts the longest connected path from a 3D skeleton mask.

        Args:
            skel (np.ndarray): Binary 3D skeleton mask.

        Returns:
            np.ndarray: Coordinates of the longest skeleton path in voxel order.
        """
        # Define 26-connected structure
        struct = generate_binary_structure(3, 2)

        # Initialize graph
        G = nx.Graph()
        skel_coords = np.argwhere(skel)

        # Build the graph from skeleton voxels
        for voxel in skel_coords:
            for offset in np.argwhere(struct) - 1:
                neighbor = voxel + offset
                if (
                    np.all(neighbor >= 0) and
                    np.all(neighbor < skel.shape) and
                    skel[tuple(neighbor)]
                ):
                    G.add_edge(tuple(voxel), tuple(neighbor))

        if len(G.nodes) == 0:
            return np.empty((0, 3), dtype=int)

        # Find endpoints (degree == 1)
        endpoints = [n for n in G.nodes if G.degree[n] == 1]

        longest_path = []
        if len(endpoints) >= 2:
            # Check all endpoint pairs for the longest shortest path
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    try:
                        path = nx.shortest_path(G, endpoints[i], endpoints[j])
                        if len(path) > len(longest_path):
                            longest_path = path
                    except nx.NetworkXNoPath:
                        continue
        else:
            # Fallback: use any DFS path
            longest_path = list(nx.dfs_preorder_nodes(G, source=list(G.nodes)[0]))

        return np.array(longest_path, dtype=int)