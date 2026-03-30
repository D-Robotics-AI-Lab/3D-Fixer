# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

import os
import cv2
import json
from typing import *
from PIL import Image
import numpy as np
import torch
import utils3d
import open3d as o3d
from .components import (
    StandardDatasetBase, 
)
from .utils import (
    voxelize_mesh,
    rot_vertices,
    transform_vertices,
    get_cam_poses,
    get_rays,
    edge_mask_morph_gradient
)
from .sparse_structure_latent import SparseStructureLatentVisMixin
       

class ObjectSparseStructureVoxel(SparseStructureLatentVisMixin, StandardDatasetBase):
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
        perturb_scale_min, perturb_scale_max, perturb_trans_factor, perturb_rot_angles, perturb_ratio (float): perturbation parameters
        max_mesh_size (float): maximum of mesh size.
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

        pack = {
            'origin_vertices': origin_vertices,
            'origin_triangles': origin_triangles
        }
        if self.normalization is not None:
            pack['mean'] = self.mean
            pack['std'] = self.std
        return pack
    
    @staticmethod
    def collate_fn(batch):
        pack = {}        
        
        # collate other data
        keys = batch[0].keys()
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack
    


class ObjectInstanceConditionedMixin:
    def __init__(self, roots, *, image_size=518, 
                 erode_kernel_size=7,
                 erode_iters=2,
                 resolution=64, 
                 perturb_scale_min: float = 0.5,
                 perturb_scale_max: float = 1.1,
                 perturb_trans_factor: float = 0.05,
                 perturb_rot_angles: float = 180.0, 
                 resize_perturb: bool = False, 
                 resize_perturb_ratio: float = 0.5,
                 only_cond_renders: bool = False,
                 **kwargs):
        self.image_size = image_size
        self.erode_kernel_size = erode_kernel_size
        self.erode_iters = erode_iters
        self.resolution = resolution
        self.perturb_scale_min = perturb_scale_min
        self.perturb_scale_range = (perturb_scale_max - perturb_scale_min)
        self.perturb_trans_factor = perturb_trans_factor
        self.perturb_rot_angles = perturb_rot_angles
        self.only_cond_renders = only_cond_renders
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        self.resize_perturb = resize_perturb
        self.resize_perturb_ratio = resize_perturb_ratio
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        if self.only_cond_renders:
            metadata = metadata[metadata[f'cond_rendered']]
            stats['Cond rendered'] = len(metadata)
            return metadata, stats
        else:
            if 'cond_rendered' in metadata.columns:
                metadata = metadata[metadata[f'cond_rendered'].fillna(False) | metadata[f'rendered'].fillna(False)]
            else:
                metadata = metadata[metadata[f'rendered'].fillna(False)]
            stats['Cond rendered'] = len(metadata)
            return metadata, stats
        
    def voxelize_mesh(self, target_vertices_mesh, canonical_gt_mesh_faces):
        perturb_rot   = (np.random.rand(3) * 2.0 - 1.0) * np.deg2rad(self.perturb_rot_angles)
        perturb_trans = (np.random.rand(3) * 2.0 - 1.0) * self.perturb_trans_factor
        perturb_scale = np.random.rand() * self.perturb_scale_range + self.perturb_scale_min
        
        target_vertices_mesh = rot_vertices(target_vertices_mesh, perturb_rot, ['z', 'y', 'x'])
        target_vertices_mesh = transform_vertices(target_vertices_mesh, ops=['scale', 'translation'],
                                                  params=[perturb_scale, perturb_trans])
        target_vox = voxelize_mesh(target_vertices_mesh, canonical_gt_mesh_faces, clip_range_first=True, return_mask=False)
        return target_vox, perturb_rot, perturb_trans, perturb_scale
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        origin_vertices = pack.pop('origin_vertices')
        origin_triangles = pack.pop('origin_triangles')

        mesh = {
            'verts': torch.nan_to_num(torch.from_numpy(origin_vertices).float()),
            'faces': torch.nan_to_num(torch.from_numpy(np.asarray(origin_triangles))).long()
        }
        pack['mesh'] = mesh

        target_vox, perturb_rot, perturb_trans, perturb_scale = self.voxelize_mesh(origin_vertices, origin_triangles)
        est_depth_ratio = min(1.0, np.random.rand() * 1.01)
        pack['x_0'] = target_vox
        pack['perturb_rot'] = torch.nan_to_num(torch.from_numpy(perturb_rot).float())
        pack['perturb_trans'] = torch.nan_to_num(torch.from_numpy(perturb_trans).float())
        pack['perturb_scale'] = torch.tensor(perturb_scale, dtype=torch.float32)
        pack['est_depth_frac'] = torch.tensor(est_depth_ratio, dtype=torch.float32)

        if self.only_cond_renders:
            image_root = os.path.join(root, 'renders_cond', instance)
            with open(os.path.join(image_root, 'transforms.json')) as f:
                metadata = json.load(f)
        else:
            if os.path.exists(os.path.join(root, 'renders_cond', instance, 'transforms.json')):
                image_root = os.path.join(root, 'renders_cond', instance)
                with open(os.path.join(image_root, 'transforms.json')) as f:
                    metadata = json.load(f)
            else:
                image_root = os.path.join(root, 'renders', instance)
                with open(os.path.join(image_root, 'transforms.json')) as f:
                    metadata = json.load(f)

        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)
        pack['intrinsics'] = intrinsics
        pack['extrinsics'] = extrinsics

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        H, W = image.size
        K, c2w = get_cam_poses(metadata, H, W)
        K[0] = K[0] / (W / self.image_size)
        K[1] = K[1] / (H / self.image_size)
        i, j = torch.meshgrid(
            torch.arange(0, self.image_size).float(), 
            torch.arange(0, self.image_size).float()
        )
        rays_o, rays_d = get_rays(i, j, K, c2w)
        pack['rays_o'] = rays_o
        pack['rays_d'] = rays_d

        scene_image_rgba = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        scene_alpha = scene_image_rgba.getchannel(3)
        scene_image = scene_image_rgba.convert('RGB')
        instance_mask = (np.array(scene_alpha) > 127).astype(np.uint8)
        edge_mask = edge_mask_morph_gradient(instance_mask, self.kernel, self.erode_iters)
        fg_mask = (instance_mask > edge_mask).astype(np.uint8)

        scene_image = torch.tensor(np.array(scene_image)).permute(2, 0, 1).float() / 255.0
        scene_alpha = torch.tensor(np.array(scene_alpha)).float() / 255.0
        pack['cond_scene_masked'] = torch.cat([scene_image, scene_alpha.unsqueeze(0)])

        pack['fg_mask'] = torch.from_numpy(fg_mask)
        pack['edge_mask'] = torch.from_numpy(edge_mask)

        alpha = image.getchannel(3)
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        if self.resize_perturb and np.random.rand() < self.resize_perturb_ratio:
            rand_reso = np.random.randint(32, self.image_size)
            image = image.resize((rand_reso, rand_reso), Image.Resampling.LANCZOS)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        pack['cond_instance_masked'] = torch.cat([image, alpha.unsqueeze(0)])
       
        return pack


class ObjectImageConditionedSparseStructureVoxel(ObjectInstanceConditionedMixin, ObjectSparseStructureVoxel):
    """
    Image-conditioned sparse structure dataset
    """
    pass
    