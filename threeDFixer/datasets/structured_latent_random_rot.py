# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics.

import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from .structured_latent import SLatVisMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.data_utils import load_balanced_group_indices


   
class SLatRandRot(SLatVisMixin, StandardDatasetBase):
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
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        perturb_ratio: float = 0.5,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.perturb_ratio = perturb_ratio
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_{self.latent_model}'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats
    
    def load_slat(self, fpath):
        data = np.load(fpath)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
        }

    def get_instance(self, root, instance):
        slat_pack = None
        if np.random.rand() < self.perturb_ratio:
            if os.path.exists(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}/transforms.json')):
                with open(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}/transforms.json'), 'r') as f:
                    transforms = json.load(f)
                if len(transforms) > 0:
                    rot_slat_info = np.random.choice(transforms)
                    slat_pack = self.load_slat(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}', rot_slat_info['file_path']))

        slat_pack = self.load_slat(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz')) if slat_pack is None else slat_pack
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
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
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
        
    
class TextConditionedSLatRandRot(TextConditionedMixin, SLatRandRot):
    """
    Text conditioned structured latent dataset
    """
    pass


class ImageConditionedSLatRandRot(ImageConditionedMixin, SLatRandRot):
    """
    Image conditioned structured latent dataset
    """
    pass
