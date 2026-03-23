# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

import os
import json
from typing import *
import numpy as np
import torch
import utils3d
import open3d as o3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import (
    StandardDatasetBase, 
    TextConditionedMixin, 
    ImageConditionedMixin
)
from .utils import (
    voxelize_mesh,
    rot_vertices,
    transform_vertices,
    renormalize_vertices
)
from .sparse_structure_latent import SparseStructureLatentVisMixin
from .. import models
       

class SparseStructureLatentRandRot(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        perturb_scale_min: float = 0.5,
        perturb_scale_max: float = 1.1,
        perturb_trans_factor: float = 0.05,
        perturb_rot_angles: float = 180.0,
        perturb_ratio: float = 0.5,
        max_mesh_size: float = 50.0,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        self.perturb_scale_min = perturb_scale_min
        self.perturb_scale_range = (perturb_scale_max - perturb_scale_min)
        self.perturb_trans_factor = perturb_trans_factor
        self.perturb_rot_angles = perturb_rot_angles
        self.perturb_ratio = perturb_ratio
        self.max_mesh_size = max_mesh_size
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        DEBUG_DATASET_ITEMS = os.environ.get('DEBUG_DATASET_ITEMS', None)
        if DEBUG_DATASET_ITEMS is not None:
            metadata = metadata[:int(DEBUG_DATASET_ITEMS)]

        stats = {}
        if 'readable' in metadata.columns:
            metadata = metadata[metadata['readable'] == True]
        else:
            metadata = metadata[metadata['rendered'] == True]
        stats['With readable ply'] = len(metadata)
        if 'mesh_size' in metadata.columns:
            metadata = metadata[metadata['mesh_size'] <= self.max_mesh_size]
            stats[f'With mesh smaller than {self.max_mesh_size}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        ply_path = os.path.join(root, 'renders', instance, 'mesh.ply')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        # normalize mesh
        vertices = np.asarray(mesh.vertices)
        vertices = np.nan_to_num(vertices)
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        origin_vertices = np.asarray(mesh.vertices)
        origin_triangles = mesh.triangles

        # if np.random.rand() < self.perturb_ratio:
        perturb_rot   = (np.random.rand(3) * 2.0 - 1.0) * np.deg2rad(self.perturb_rot_angles)
        perturb_trans = (np.random.rand(3) * 2.0 - 1.0) * self.perturb_trans_factor
        # perturb_scale = np.random.rand() * self.perturb_scale_range + self.perturb_scale_min
        # perturb_scale = perturb_scale if np.random.rand() < 0.8 else 0.5

        origin_vertices = rot_vertices(origin_vertices, perturb_rot, ['z', 'y', 'x'])
        # origin_vertices = transform_vertices(origin_vertices, ops=['scale', 'translation'],
        #                                                 params=[perturb_scale, perturb_trans])
        origin_vertices = transform_vertices(origin_vertices, ops=['translation'],
                                                        params=[perturb_trans])
        origin_vertices = renormalize_vertices(origin_vertices)

        x_0 = voxelize_mesh(origin_vertices, origin_triangles, clip_range_first=True, return_mask=False)

        pack = {
            'x_0': x_0,
        }
        if self.normalization is not None:
            pack['mean'] = self.mean
            pack['std'] = self.std
        return pack


class TextConditionedSparseStructureLatentRandRot(TextConditionedMixin, SparseStructureLatentRandRot):
    """
    Text-conditioned sparse structure dataset
    """
    pass


class ImageConditionedSparseStructureLatentRandRot(ImageConditionedMixin, SparseStructureLatentRandRot):
    """
    Image-conditioned sparse structure dataset
    """
    pass
    