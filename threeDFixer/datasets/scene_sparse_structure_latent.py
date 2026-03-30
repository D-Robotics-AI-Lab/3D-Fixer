import os
import cv2
import json
from typing import *
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from .components import (
    StandardSceneDatasetBase, 
)
from .utils import (
    vox2pts,
    voxelize_pcd,
    voxelize_mesh,
    rot_vertices,
    transform_vertices,
    normalize_vertices,
    get_std_cond,
    get_instance_mask,
    get_gt_depth,
    get_mix_est_depth,
    get_cam_poses,
    edge_mask_morph_gradient,
    process_scene_image,
    process_instance_image,
    lstsq_align_depth
)
from .sparse_structure_latent import SparseStructureLatentVisMixin

class SceneSparseStructureVoxel(SparseStructureLatentVisMixin, StandardSceneDatasetBase):
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
        max_mesh_size: float = 50.0,
        **kwargs
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.max_mesh_size = max_mesh_size
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            **kwargs
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
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

class SceneInstanceSingleConditionedMixin:
    def __init__(self, roots, *, image_size=1024, 
                 crop_size=518, erode_kernel_size=7,
                 stage='coarse', 
                 erode_iters=2,
                 min_pixel_ratio=0.001, 
                 resolution=64, est_depth_folder=['depth_MoGeV2'],
                 perturb_scale_min: float = 0.5,
                 perturb_scale_max: float = 1.1,
                 perturb_trans_factor: float = 0.05,
                 perturb_rot_angles: float = 180.0, 
                 resize_perturb: bool = False, 
                 resize_perturb_ratio: float = 0.5,
                 **kwargs):
        self.image_size = image_size
        self.erode_kernel_size = erode_kernel_size
        self.erode_iters = erode_iters
        self.min_pixel_ratio = min_pixel_ratio
        self.resolution = resolution
        self.crop_size = crop_size
        self.est_depth_folder = est_depth_folder
        self.perturb_scale_min = perturb_scale_min
        self.perturb_scale_range = (perturb_scale_max - perturb_scale_min)
        self.perturb_trans_factor = perturb_trans_factor
        self.perturb_rot_angles = perturb_rot_angles
        self.stage = stage
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        self.resize_perturb = resize_perturb
        self.resize_perturb_ratio = resize_perturb_ratio
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['pixel_ratio'] >= self.min_pixel_ratio]
        stats[f'Valid pixel ratio >= {self.min_pixel_ratio:.03f}'] = len(metadata)
        return metadata, stats
    
    def voxelize_all(self, partial_cond_points, partial_cond_points_mask, target_vertices_mesh, canonical_gt_mesh_faces):
        perturb_rot   = (np.random.rand(3) * 2.0 - 1.0) * np.deg2rad(self.perturb_rot_angles)
        perturb_trans = (np.random.rand(3) * 2.0 - 1.0) * self.perturb_trans_factor
        perturb_scale = np.random.rand() * self.perturb_scale_range + self.perturb_scale_min

        partial_cond_points = rot_vertices(partial_cond_points, perturb_rot, ['z', 'y', 'x'])
        partial_cond_points = transform_vertices(partial_cond_points, ops=['scale', 'translation'],
                                                        params=[perturb_scale, perturb_trans])
        
        target_vertices_mesh = rot_vertices(target_vertices_mesh, perturb_rot, ['z', 'y', 'x'])
        target_vertices_mesh = transform_vertices(target_vertices_mesh, ops=['scale', 'translation'],
                                                  params=[perturb_scale, perturb_trans])
        partial_cond_vox, partial_cond_vox_mask = voxelize_pcd(partial_cond_points, partial_cond_points_mask, clip_range_first=True, return_mask=True)
        target_vox = voxelize_mesh(target_vertices_mesh, canonical_gt_mesh_faces, clip_range_first=True, return_mask=False)
        return partial_cond_vox, partial_cond_vox_mask, target_vox
    
    def get_instance(self, instance: str):
        example_metadata = self.metadata.loc[instance]
        scene_sha256, frame_index, instance_sha256, selected_instance_index = instance.strip('/').split('/')
        selected_instance_index = int(selected_instance_index)
        transforms_path = os.path.join(self.scene_image_root, scene_sha256, 'transforms.json')
        image_path = os.path.join(self.scene_image_root, scene_sha256, frame_index + '.png')
        instance_mask_path = os.path.join(self.scene_image_root, scene_sha256, frame_index + '_index.png')
        gt_depth_path = os.path.join(self.scene_image_root, scene_sha256, frame_index + '_depth.png')
        est_depth_path_lists = [
           os.path.join(self.scene_image_root, scene_sha256, folder, frame_index + '.npz') for folder in self.est_depth_folder
        ]
        existing_est_depth_path = list(filter(lambda x: os.path.exists(x), est_depth_path_lists))
        est_depth_path = np.random.choice(existing_est_depth_path) if len(existing_est_depth_path) > 0 else None
        instance_ply_root = os.path.join(self.asset_root, example_metadata['from'])
        # load voxelized ply
        pack = super().get_instance(instance_ply_root, instance_sha256)

        std_cond = get_std_cond(instance_ply_root, instance_sha256, self.crop_size)

        # load image conditions
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        image = Image.open(image_path)
        index_mask, _ = get_instance_mask(instance_mask_path)
        frame_info = list(filter(lambda x: x['file_path'] == frame_index + '.png', transforms['frames']))[0]
        gt_depth = get_gt_depth(gt_depth_path, frame_info)
        H, W = gt_depth.shape[:2]
        
        try:
            if est_depth_path is not None:
                est_depth, est_depth_mask = get_mix_est_depth(est_depth_path, H)
                est_depth_ratio = min(1.0, np.random.rand() * 1.01)
            else:
                est_depth = gt_depth
                est_depth_mask = index_mask >= 3
                est_depth_ratio = 0.0
        except Exception:
            est_depth = gt_depth
            est_depth_mask = index_mask >= 3
            est_depth_ratio = 0.0

        K, c2w = get_cam_poses(frame_info, H, W)
        instance_mask = (np.logical_and(index_mask==selected_instance_index, est_depth_mask)).astype(np.uint8)
        edge_mask = edge_mask_morph_gradient(instance_mask, self.kernel, self.erode_iters)
        fg_mask = (instance_mask > edge_mask).astype(np.uint8)

        color_mask = fg_mask.astype(np.float32) + edge_mask.astype(np.float32) * 0.5

        est_depth_aligned = lstsq_align_depth(est_depth, gt_depth, 
                                              torch.from_numpy(fg_mask))
        mix_depth_map = est_depth_aligned * est_depth_ratio + gt_depth * (1.0 - est_depth_ratio)

        scene_image, scene_image_masked = process_scene_image(image, instance_mask, self.crop_size,
                                                              self.resize_perturb, self.resize_perturb_ratio)
        instance_image, instance_mask, instance_rays_o, instance_rays_d, instance_rays_c, \
            instance_rays_t = process_instance_image(image, instance_mask, color_mask, mix_depth_map, K, c2w, self.crop_size)
        
        pcd_points = (instance_rays_o + instance_rays_d * instance_rays_t[..., None]).detach().cpu().numpy() # pt2np
        pcd_colors = instance_rays_c

        # pack['pixel_points'] = pcd_points
        # pack['pixel_colors'] = repeat(pcd_colors, 'n -> n c', c=3)
        # pcd_points = pcd_points.detach().cpu().numpy()

        # prepare for GT
        gt_rot = transforms['instance'][f'{selected_instance_index}']['rand_rot']
        canonical_gt_mesh_vertices = rot_vertices(pack.pop('origin_vertices'), rot_angles=[gt_rot[2]], axis_list=['z'])
        canonical_gt_mesh_faces = pack.pop('origin_triangles')
        del gt_rot
        gt_trans = np.array([transforms['instance'][f'{selected_instance_index}']['rand_trans'][0], 
                             transforms['instance'][f'{selected_instance_index}']['rand_trans'][1], 
                             transforms['instance'][f'{selected_instance_index}']['rand_trans'][2]])
        gt_scale = transforms['instance'][f'{selected_instance_index}']['rand_scale']
        local_gt_mesh_vertices = transform_vertices(canonical_gt_mesh_vertices, ops=['scale', 'translation'],
                                                    params=[gt_scale, -gt_trans[None]])
        del gt_trans
        del gt_scale

        # scale_factor=2.0 to normalize instance into [-0.25, 0.25]
        valid_pcd_points = pcd_points[pcd_colors > 0.8]
        _, coarse_translation, coarse_scale = normalize_vertices(valid_pcd_points, scale_factor=2.0)
        coarse_cond_points = transform_vertices(pcd_points, ops=['translation', 'scale'],
                                                         params=[-coarse_translation, 1.0 / (coarse_scale + 1e-6)])
        invalid_points_mask = np.logical_or(
            (coarse_cond_points < -0.25).any(axis=-1),
            (coarse_cond_points > 0.25).any(axis=-1),
        )
        coarse_cond_points = coarse_cond_points[~invalid_points_mask]
        pcd_colors = pcd_colors[~invalid_points_mask]
        coarse_target_vertices_mesh = transform_vertices(local_gt_mesh_vertices, ops=['translation', 'scale'],
                                                         params=[-coarse_translation, 1.0 / (coarse_scale + 1e-6)])
        coarse_target_vertices_mesh = np.clip(coarse_target_vertices_mesh, a_min=-0.5+1e-6, a_max=0.5-1e-6)

        pack['cond_instance_masked'] = torch.cat([instance_image, instance_mask]) # DINO v2
        pack['cond_scene_masked'] = scene_image_masked # MoGe v2

        # 
        pack['est_depth_frac'] = torch.tensor(est_depth_ratio, dtype=torch.float32)
        pack['std_cond_instance'] = std_cond

        if self.stage == 'coarse':
            # voxelize points
            # coarse stage
            coarse_partial_cond_vox, coarse_partial_cond_vox_mask, coarse_target_vox = \
                self.voxelize_all(coarse_cond_points, pcd_colors, coarse_target_vertices_mesh, canonical_gt_mesh_faces)
            # coarse
            pack['x_0'] = coarse_target_vox
            pack['cond_vox'] = coarse_partial_cond_vox
            pack['cond_mask'] = coarse_partial_cond_vox_mask

            return pack

        elif self.stage == 'fine':

            # prepare for fine state conditions and GT
            coarse_target_vertices2vox = voxelize_mesh(coarse_target_vertices_mesh, canonical_gt_mesh_faces, clip_range_first=False, return_mask=False)
            coarse_target_vertices2vox2pts = vox2pts(coarse_target_vertices2vox, resolution=self.resolution)
            # scale_factor=1.0 to normalize instance into [-0.5, 0.5]
            _, fine_translation, fine_scale = normalize_vertices(coarse_target_vertices2vox2pts, scale_factor=1.0)
            fine_target_vertices_mesh = transform_vertices(coarse_target_vertices_mesh, ops=['translation', 'scale'],
                                                            params=[-fine_translation, 1.0 / (fine_scale + 1e-6)])
            fine_partial_cond_points = transform_vertices(coarse_cond_points, ops=['translation', 'scale'],
                                                            params=[-fine_translation, 1.0 / (fine_scale + 1e-6)])
            
            # voxelize points
            # fine stage
            fine_partial_cond_vox, fine_partial_cond_vox_mask, fine_target_vox = \
                self.voxelize_all(fine_partial_cond_points, pcd_colors, fine_target_vertices_mesh, canonical_gt_mesh_faces)
            
            # pack['verts'] = local_gt_mesh_vertices
            # pack['faces'] = canonical_gt_mesh_faces
            
            # fine
            pack['x_0'] = fine_target_vox
            pack['cond_vox'] = fine_partial_cond_vox
            pack['cond_mask'] = fine_partial_cond_vox_mask
    
            return pack


class SceneImageConditionedVoxel(SceneInstanceSingleConditionedMixin, SceneSparseStructureVoxel):
    """
    Image-conditioned sparse structure dataset
    """
    pass

