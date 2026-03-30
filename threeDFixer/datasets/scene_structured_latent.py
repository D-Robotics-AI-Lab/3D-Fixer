import json
import os
import pandas as pd
from PIL import Image
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardSceneDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.data_utils import load_balanced_group_indices
from .structured_latent import SLatVisMixin
from .utils import (
    map_rotated_slat2canonical_pose,
    get_std_cond,
    get_instance_mask,
    process_scene_image,
    process_instance_image_only,
    rot_vertices,
    transform_vertices
)


class SceneSLat(SLatVisMixin, StandardSceneDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        min_pixel_ratio: float = 0.005,
        resolution: int = 64,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.min_pixel_ratio = min_pixel_ratio
        self.resolution = resolution
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
            **kwargs
        )

        self.loads = [self.metadata.loc[sha256, 'num_voxels'] for sha256 in self.instances]
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

    def filter_metadata(self, metadata):
        
        def map_example_id_to_attrs(item: pd.Series, asset_metadata, attr):
            asset_source = item['from']
            instance_sha256 = item['example_id'].strip('/').split('/')[-2]
            try:
                return asset_metadata[asset_source].loc[instance_sha256, attr]
            except Exception:
                return False

        DEBUG_DATASET_ITEMS = os.environ.get('DEBUG_DATASET_ITEMS', None)
        if DEBUG_DATASET_ITEMS is not None:
            metadata = metadata[:int(DEBUG_DATASET_ITEMS)]
            
        stats = {}
        print (f'filtering pixel ratio >= {self.min_pixel_ratio:.03f}')
        metadata = metadata[metadata['pixel_ratio'] >= self.min_pixel_ratio]
        stats[f'Valid pixel ratio >= {self.min_pixel_ratio:.03f}'] = len(metadata)
        
        print (f'filtering latent_{self.latent_model}')
        if not (f'latent_{self.latent_model}' in metadata.columns):
            metadata[f'latent_{self.latent_model}'] = metadata.apply(lambda x: 
                                                                    map_example_id_to_attrs(x, 
                                                                    self.asset_metadata, 
                                                                    f'latent_{self.latent_model}'), axis=1)
        metadata = metadata[~metadata[f'latent_{self.latent_model}'].isna()]
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)

        print (f'filtering aesthetic_score')
        if not (f'aesthetic_score' in metadata.columns):
            metadata['aesthetic_score'] = metadata.apply(lambda x: 
                                                                    map_example_id_to_attrs(x, 
                                                                    self.asset_metadata, 
                                                                    'aesthetic_score'), axis=1)
        metadata = metadata[~metadata['aesthetic_score'].isna()]
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        
        print (f'filtering num_voxels')
        if not (f'num_voxels' in metadata.columns):
            metadata['num_voxels'] = metadata.apply(lambda x: 
                                                                 map_example_id_to_attrs(x, 
                                                                 self.asset_metadata, 
                                                                 'num_voxels'), axis=1)
        metadata = metadata[~metadata['num_voxels'].isna()]
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)

        print (f'filtering out blend files')
        if not (f'local_path' in metadata.columns):
            metadata['local_path'] = metadata.apply(lambda x: 
                                                                 map_example_id_to_attrs(x, 
                                                                 self.asset_metadata, 
                                                                 'local_path'), axis=1)
        metadata = metadata[~metadata['local_path'].isna()]
        metadata = metadata[~metadata['local_path'].str.endswith(".blend")]
        stats[f'Assets not endswith .blend'] = len(metadata)

        return metadata, stats
    
    def load_slat(self, fpath):
        data = np.load(fpath)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        vertices = (data['coords'] + 0.5) / self.resolution - 0.5 # np.ndarray
        return {
            'vertices': vertices,
            'coords': coords,
            'feats': feats,
        }
    
    def get_instance(self, root, instance):
        slat_pack = None
        if os.path.exists(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}/transforms.json')):
            with open(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}/transforms.json'), 'r') as f:
                transforms = json.load(f)
            if len(transforms) > 0:
                rot_slat_info = np.random.choice(transforms)
                slat_pack = self.load_slat(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}', rot_slat_info['file_path']))
                canonical_vertices = map_rotated_slat2canonical_pose(slat_pack['vertices'], rot_slat_info)
                slat_pack['vertices'] = canonical_vertices
        
        slat_pack = self.load_slat(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz')) if slat_pack is None else slat_pack
        slat_pack['vertices'] = torch.from_numpy(slat_pack['vertices']).float()
        return slat_pack
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            vis_ratio = []
            uv = []
            start = 0
            moge_c = 32
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                sample_rgba = torch.nn.functional.grid_sample(
                    b['cond_scene_masked'].unsqueeze(0),
                    b['uv'].unsqueeze(0).unsqueeze(0),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).squeeze(0).permute(1, 0)
                vis_ratio.append(torch.sum(sample_rgba[:, -1:]) / (torch.sum(torch.ones_like(sample_rgba[:, -1:])) + 1e-6))
                uv.append(b['uv'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            uv = torch.cat(uv)
            vis_ratio = torch.stack(vis_ratio)
            moge_feats = torch.zeros((feats.shape[0], moge_c), dtype=feats.dtype, device=feats.device) # hand-coded moge feature dims
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            pack['cond_voxel_feat'] = SparseTensor(
                coords=coords.clone(),
                feats=moge_feats,
            )
            pack['cond_voxel_feat']._shape = torch.Size([len(group), moge_c, *sub_batch[0]['feats'].shape[2:]])
            pack['cond_voxel_feat'].register_spatial_cache('layout', layout)

            pack['uv'] = SparseTensor(
                coords=coords.clone(),
                feats=uv,
            )
            pack['uv']._shape = torch.Size([len(group), 2, *sub_batch[0]['feats'].shape[2:]])
            pack['uv'].register_spatial_cache('layout', layout)

            pack['vis_ratio'] = vis_ratio
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats', 'uv']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
    
class SceneInstanceConditionedMixin:

    def __init__(self, roots, *, image_size=1024, crop_size=518,
                 resize_perturb: bool = False, 
                 resize_perturb_ratio: float = 0.5, **kwargs):
        self.image_size = image_size
        self.crop_size = crop_size
        self.resize_perturb = resize_perturb
        self.resize_perturb_ratio = resize_perturb_ratio
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        return metadata, stats
    
    def get_instance(self, instance: str):
        example_metadata = self.metadata.loc[instance]
        scene_sha256, frame_index, instance_sha256, selected_instance_index = instance.strip('/').split('/')
        selected_instance_index = int(selected_instance_index)
        transforms_path = os.path.join(self.scene_image_root, scene_sha256, 'transforms.json')
        image_path = os.path.join(self.scene_image_root, scene_sha256, frame_index + '.png')
        instance_mask_path = os.path.join(self.scene_image_root, scene_sha256, frame_index + '_index.png')
        instance_ply_root = os.path.join(self.asset_root, example_metadata['from'])
        instance_cond_root = os.path.join(self.asset_root, example_metadata['from'], 'renders_cond', instance_sha256)
        # load slat
        pack = super().get_instance(instance_ply_root, instance_sha256)

        std_cond = get_std_cond(instance_ply_root, instance_sha256, self.crop_size)

        # load image conditions
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        image = Image.open(image_path)
        H, W = image.size
        index_mask, _ = get_instance_mask(instance_mask_path)
        frame_info = list(filter(lambda x: x['file_path'] == frame_index + '.png', transforms['frames']))[0]

        c2w = torch.from_numpy(np.array(frame_info['transform_matrix'])).float()
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)
        fov = float(frame_info['camera_angle_x'])
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        instance_mask = (index_mask==selected_instance_index).astype(np.uint8)
        scene_image, scene_image_masked = process_scene_image(image, instance_mask, self.crop_size,
                                                              self.resize_perturb, self.resize_perturb_ratio)
        instance_image, instance_mask = process_instance_image_only(image, instance_mask, self.crop_size)

        # prepare for GT
        gt_rot = transforms['instance'][f'{selected_instance_index}']['rand_rot']
        canonical_gt_coords = pack.pop('coords')
        canonical_gt_vertices = pack.pop('vertices')
        cano2local_gt_vertices = rot_vertices(canonical_gt_vertices, rot_angles=[gt_rot[2]], axis_list=['z'])
        # translate back to scene position
        gt_trans = np.array([transforms['instance'][f'{selected_instance_index}']['rand_trans'][0], 
                             transforms['instance'][f'{selected_instance_index}']['rand_trans'][1], 
                             transforms['instance'][f'{selected_instance_index}']['rand_trans'][2]])
        gt_scale = transforms['instance'][f'{selected_instance_index}']['rand_scale']
        cano2local_gt_vertices = transform_vertices(cano2local_gt_vertices, ops=['scale', 'translation'],
                                                    params=[gt_scale, -gt_trans[None]])
        # uv coords to sample features
        uv = utils3d.torch.project_cv(torch.from_numpy(cano2local_gt_vertices).float(), extrinsics, intrinsics)[0] * 2 - 1

        pack['coords'] = canonical_gt_coords
        pack['uv'] = uv

        pack['cond_instance_masked'] = torch.cat([instance_image, instance_mask]) # DINO v2
        pack['cond_scene_masked'] = scene_image_masked # MoGe v2
        pack['std_cond_instance'] = std_cond # std cond

        return pack

class SceneConditionedSLat(SceneInstanceConditionedMixin, SceneSLat):
    """
    Image conditioned structured latent dataset
    """
    pass
