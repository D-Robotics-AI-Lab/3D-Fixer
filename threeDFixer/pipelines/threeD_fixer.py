# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..datasets.utils import (
    transform_vertices,
    voxelize_pcd
)
from ..moge.model.v2 import MoGeModel
from einops import rearrange
import utils3d
from einops import repeat

class ThreeDFixerPipeline(Pipeline):
    """
    Pipeline for inferring 3D-Fixer models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
        scene_cond_model (str): The name of the scene conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
        scene_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.coarse_sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.coarse_sparse_structure_sampler_params = None
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        self._init_scene_cond_model(scene_cond_model)

    @staticmethod
    def from_pretrained(path: str, compile: bool = True) -> "ThreeDFixerPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(ThreeDFixerPipeline, ThreeDFixerPipeline).from_pretrained(path)
        new_pipeline = ThreeDFixerPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']
        if 'coarse_sparse_structure_sampler' in args:
            new_pipeline.coarse_sparse_structure_sampler = getattr(samplers, args['coarse_sparse_structure_sampler']['name'])(**args['coarse_sparse_structure_sampler']['args'])
            new_pipeline.coarse_sparse_structure_sampler_params = args['coarse_sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline._init_scene_cond_model(args['scene_cond_model'])

        if compile:
            new_pipeline.models['image_cond_model'] = torch.compile(new_pipeline.models['image_cond_model'])
            new_pipeline.models['scene_cond_model'] = torch.compile(new_pipeline.models['scene_cond_model'])
            new_pipeline.models['sparse_structure_flow_model'] = torch.compile(new_pipeline.models['sparse_structure_flow_model'])
            new_pipeline.models['scene_sparse_structure_flow_coarse_model'] = torch.compile(new_pipeline.models['scene_sparse_structure_flow_coarse_model'])
            new_pipeline.models['scene_sparse_structure_flow_fine_model'] = torch.compile(new_pipeline.models['scene_sparse_structure_flow_fine_model'])
            new_pipeline.models['sparse_structure_decoder'] = torch.compile(new_pipeline.models['sparse_structure_decoder'])
            new_pipeline.models['sparse_structure_encoder'] = torch.compile(new_pipeline.models['sparse_structure_encoder'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def _init_scene_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        moge_model = MoGeModel.from_pretrained(name)
        moge_model.eval()
        self.models['scene_cond_model'] = moge_model

    def _get_dino_feats(self, image):
        image = self.image_cond_model_transform(image).cuda()
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats(self, image):
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.models['scene_cond_model'].infer_feature_tokens(image, num_tokens=num_tokens)
        features = rearrange(features, 'b c (h n1) (w n2) -> b (h w) (n1 n2 c)', n1=16, n2=16)
        features = torch.nn.functional.adaptive_avg_pool1d(
            features, 1024
        )
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats_ss(self, image):
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.models['scene_cond_model'].infer_feature_tokens(image, num_tokens=num_tokens, tokens_layer=0)
        features = rearrange(features, 'b c h w -> b (h w) c')
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats_hw(self, image):
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.models['scene_cond_model'].infer_feature_tokens(image, num_tokens=num_tokens)
        h, w = features.shape[-2:]
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = F.layer_norm(features, features.shape[-1:])
        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)
        return features
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]], enc_fn) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        patchtokens = enc_fn(image)
        return patchtokens
    
    def get_ss_cond(self, cond_instance_masked: Union[torch.Tensor, list[Image.Image]],
                          cond_scene_masked: Union[torch.Tensor, list[Image.Image]],
                          x_0_partial_cond, x_0_partial_cond_mask
                          ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        batch_inputs = []
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)
        batch_inputs.append(
            cond_instance * cond_instance_mask
        )
        batch_inputs.append(
            cond_instance
        )
        batch_inputs.append(
            repeat(cond_instance_mask, 'b c ... -> b (n c) ...', n=3)
        )
        
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)

        batch_inputs.append(
            repeat(cond_scene_mask, 'b c ... -> b (n c) ...', n=3)
        )

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats_ss)

        batch_dino_feats = self.encode_image(torch.cat(batch_inputs), enc_fn=self._get_dino_feats)
        cond_instance_masked, cond_instance, cond_instance_mask, cond_scene_mask = batch_dino_feats.chunk(4)

        cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_scene = torch.cat([cond_scene, cond_scene_mask], dim=1)
        ##############

        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'cond_partial_vox': x_0_partial_cond, 
            'cond_partial_vox_mask': x_0_partial_cond_mask,
        }
        neg_cond = {k: torch.zeros_like(v) for k, v in cond.items()}
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    def get_slat_cond(self, coords: torch.Tensor, cond_instance_masked: Union[torch.Tensor, list[Image.Image]],
                          cond_scene_masked: Union[torch.Tensor, list[Image.Image]],
                          uv: torch.Tensor,
                          ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)

        cond_instance_masked = self.encode_image(cond_instance * cond_instance_mask, enc_fn=self._get_dino_feats)

        cond_instance = self.encode_image(cond_instance, enc_fn=self._get_dino_feats)
        cond_instance_mask = self.encode_image(repeat(cond_instance_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        cond_instance = (cond_instance + cond_instance_mask) / 2.0
        
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        cond_scene_hw = self._get_moge_feats_hw(cond_scene[:1])

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)
        cond_scene_mask = self.encode_image(repeat(cond_scene_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        cond_scene_mask[:, -cond_scene.shape[-2]:] = (cond_scene_mask[:, -cond_scene.shape[-2]:] + cond_scene) / 2.0
        cond_scene = cond_scene_mask

        sample_rgba = torch.nn.functional.grid_sample(
            cond_scene_masked,
            uv.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).squeeze(0).permute(1, 0)
        vis_ratio = torch.sum(sample_rgba[:, -1:]) / (torch.sum(torch.ones_like(sample_rgba[:, -1:])) + 1e-6)

        sample_feats = torch.nn.functional.grid_sample(
            cond_scene_hw,
            uv.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).squeeze(0).permute(1, 0)
        cond_voxel_feat = sp.SparseTensor(
            coords=coords,
            feats=sample_feats,
        )

        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'cond_voxel_feats': cond_voxel_feat,
        }
        neg_cond = {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v.replace(torch.zeros_like(v.feats))) for k, v in cond.items()}
        return {
            'cond': cond,
            'neg_cond': neg_cond,
            'vis_ratio': vis_ratio.unsqueeze(0),
        }
    
    def get_slat_cond_parallel(self, coords: torch.Tensor, cond_instance_masked: Union[torch.Tensor, list[Image.Image]],
                          cond_scene_masked: Union[torch.Tensor, list[Image.Image]],
                          uv_list: torch.Tensor
                          ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)

        cond_instance_masked = self.encode_image(cond_instance * cond_instance_mask, enc_fn=self._get_dino_feats)

        cond_instance = self.encode_image(cond_instance, enc_fn=self._get_dino_feats)
        cond_instance_mask = self.encode_image(repeat(cond_instance_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        cond_instance = (cond_instance + cond_instance_mask) / 2.0
        
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        cond_scene_hw = self._get_moge_feats_hw(cond_scene[:1])

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)
        cond_scene_mask = self.encode_image(repeat(cond_scene_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        cond_scene_mask[:, -cond_scene.shape[-2]:] = (cond_scene_mask[:, -cond_scene.shape[-2]:] + cond_scene) / 2.0
        cond_scene = cond_scene_mask

        vis_ratio = []
        for i, uv in enumerate(uv_list):
            sample_rgba = torch.nn.functional.grid_sample(
                cond_scene_masked[i][None],
                uv.unsqueeze(0).unsqueeze(0),
                mode='bilinear',
                align_corners=False,
            ).squeeze(2).squeeze(0).permute(1, 0)
            vis_ratio.append(torch.sum(sample_rgba[:, -1:]) / (torch.sum(torch.ones_like(sample_rgba[:, -1:])) + 1e-6))
        vis_ratio = torch.stack(vis_ratio)

        uv = torch.cat(uv_list)
        sample_feats = torch.nn.functional.grid_sample(
            cond_scene_hw,
            uv.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).squeeze(0).permute(1, 0)
        cond_voxel_feat = sp.SparseTensor(
            coords=coords,
            feats=sample_feats,
        )

        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'cond_voxel_feats': cond_voxel_feat,
        }
        neg_cond = {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v.replace(torch.zeros_like(v.feats))) for k, v in cond.items()}
        neg_cond['neg_infer'] = True
        return {
            'cond': cond,
            'neg_cond': neg_cond,
            'vis_ratio': vis_ratio,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        stage: str = 'fine',
        est_depth_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        scene_flow_model = self.models[f'scene_sparse_structure_flow_{stage}_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        est_depth_ratio = torch.ones(num_samples).to(self.device) * est_depth_ratio

        sampler = self.coarse_sparse_structure_sampler if (stage == 'coarse' and self.coarse_sparse_structure_sampler is not None) else self.sparse_structure_sampler
        sparse_structure_sampler_params = self.coarse_sparse_structure_sampler_params if (stage == 'coarse' and self.coarse_sparse_structure_sampler is not None) else self.sparse_structure_sampler_params
        sampler_params = {**sparse_structure_sampler_params, **sampler_params}
        z_s = sampler.sample(
            scene_flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            forzen_denoiser=flow_model,
            est_depth_ratio=est_depth_ratio * 1000.0,
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        ss = decoder(z_s) > 0
        coords = torch.argwhere(ss)[:, [0, 2, 3, 4]].int()
        return coords, ss

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        frozen_flow_model = self.models['slat_flow_model']
        flow_model = self.models['scene_slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        vis_ratio = cond.pop('vis_ratio') * 1000.0
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            vis_ratio=vis_ratio,
            forzen_denoiser=frozen_flow_model,
            stage='infer',
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        cond['vis_ratio'] = vis_ratio / 1000.0
        
        return slat
    
    def vox2pts(self, ss, resolution = 64):
        coords = torch.nonzero(ss[0, 0] > 0, as_tuple=False)
        position = (coords.float() + 0.5) / resolution - 0.5
        position = position.detach().cpu().numpy()
        return position
    
    def encode_positions(self, positions, positions_mask, ops=None, params=None):
        if ops is not None:
            positions = transform_vertices(positions, ops, params)
        positions = np.clip(positions, a_min=-0.5+1e-6, a_max=0.5-1e-6)
        ss, ss_mask = voxelize_pcd(positions, positions_mask, return_mask=True)
        latent = self.models['sparse_structure_encoder'](ss[None].float().cuda(), sample_posterior=False)
        ss_mask = ss_mask.to(latent.device)
        return latent, ss_mask[None]
    
    def project_uv(self, pts, extrinsics, intrinsics):
        pts = torch.from_numpy(pts).to(dtype=torch.float32, device=self.device)
        extrinsics = torch.inverse(extrinsics).float().to(pts)
        intrinsics = intrinsics.float().to(pts)
        uv = utils3d.torch.project_cv(pts.float(), extrinsics, intrinsics)[0] * 2 - 1
        return uv

    @torch.no_grad()
    def run(
        self,
        instance_image_masked: torch.Tensor, # RGBA for instance
        scene_image_masked: torch.Tensor, # RGBA for scene
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        points: np.array = None,
        points_mask: np.array = None,
        est_depth_ratio: float = 1.0,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            instance_image_masked (torch.Tensor): The cropped instance RGBA image in the scene.
            scene_image_masked (torch.Tensor): The scene RGBA image.
            extrinsics (torch.Tensor): The camera poses at the view.
            intrinsics (torch.Tensor): The camera intrinsics.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            points (np.array): the partial point cloud of the 3D instance.
            points_mask (np.array): the mask of the partial point cloud, 1 for foreground of the instance points, 0.5 for edge points, and 0.0 for background points.
            est_depth_ratio (float): value for the depth embedding self attention.

        Return:
            out_rf: predicted 3D assets of 3DGS, mesh, and NeRF.
            coarse_trans, coarse_scale, fine_trans, fine_scale: transform between the origin location and the canonical coordinate
        """
        torch.manual_seed(seed)        

        # coarse partial point clouds
        fg_mask = points_mask > 0.8
        points = points[fg_mask]
        points_mask = points_mask[fg_mask]
        min_pos, max_pos = np.min(points, axis=0), np.max(points, axis=0)
        coarse_trans = (min_pos + max_pos) / 2.0
        coarse_scale = np.max(max_pos - min_pos) * 2.0 + 1e-6 # scale to [-0.25, 0.25]
        coarse_ss_latents, coarse_ss_latents_mask = self.encode_positions(points, points_mask, ops=['translation', 'scale'],
                                                              params=[-coarse_trans[None], 1. / coarse_scale])
        ss_cond = self.get_ss_cond(instance_image_masked.unsqueeze(0),
                                scene_image_masked.unsqueeze(0), 
                                coarse_ss_latents,
                                coarse_ss_latents_mask)
                
        # coarse sample
        coords, coarse_ss = self.sample_sparse_structure(ss_cond, num_samples, sparse_structure_sampler_params, stage='coarse', est_depth_ratio=est_depth_ratio)
        # coarse sample

        # fine point cloud
        coarse_comp_vertices = self.vox2pts(coarse_ss)
        min_pos, max_pos = np.min(coarse_comp_vertices, axis=0), np.max(coarse_comp_vertices, axis=0)
        fine_trans = (min_pos + max_pos) / 2.0
        fine_scale = np.max(max_pos - min_pos) + 1e-6 # scale to [-0.5, 0.5]
        fine_part_ss_latents, fine_part_ss_latents_mask = self.encode_positions(points, points_mask, ops=['translation', 'scale', 'translation', 'scale'],
                                                                    params=[-coarse_trans[None], 1. / coarse_scale, -fine_trans[None], 1. / fine_scale])
        ss_cond['cond']['cond_partial_vox'] = fine_part_ss_latents
        ss_cond['cond']['cond_partial_vox_mask'] = fine_part_ss_latents_mask

        # fine sample
        coords, ss = self.sample_sparse_structure(ss_cond, num_samples, sparse_structure_sampler_params, stage='fine', est_depth_ratio=est_depth_ratio)
        # fine sample

        coords2vertices = (coords[..., 1:] + 0.5) / 64 - 0.5
        coords2vertices2local = transform_vertices(coords2vertices.detach().cpu().numpy(), ops=['scale', 'translation', 'scale', 'translation'], 
                                                   params=[fine_scale, fine_trans[None], coarse_scale, coarse_trans[None]])
        uv = self.project_uv(coords2vertices2local, extrinsics, intrinsics)

        slat_cond = self.get_slat_cond(coords, instance_image_masked.unsqueeze(0),
                                        scene_image_masked.unsqueeze(0), uv)

        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        out_rf = self.decode_slat(slat, formats)
        return out_rf, coarse_trans, coarse_scale, fine_trans, fine_scale

    @torch.no_grad()
    def run_parallel(
        self,
        instance_image_masked: torch.Tensor, # RGBA for instance
        scene_image_masked: torch.Tensor, # RGBA for scene
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        points: List[np.array] = None,
        points_mask: List[np.array] = None,
        est_depth_ratio: float = 1.0,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            instance_image_masked (torch.Tensor): The cropped instance RGBA image in the scene.
            scene_image_masked (torch.Tensor): The scene RGBA image.
            extrinsics (torch.Tensor): The camera poses at the view.
            intrinsics (torch.Tensor): The camera intrinsics.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            points (np.array): the partial point cloud of the 3D instance.
            points_mask (np.array): the mask of the partial point cloud, 1 for foreground of the instance points, 0.5 for edge points, and 0.0 for background points.
            est_depth_ratio (float): value for the depth embedding self attention.

        Return:
            out_rf: predicted 3D assets of 3DGS, mesh, and NeRF.
            coarse_trans, coarse_scale, fine_trans, fine_scale: transform between the origin location and the canonical coordinate
        """
        torch.manual_seed(seed)
                
        def encode_coarse_ss(points, points_mask):
            fg_mask = points_mask > 0.8
            points = points[fg_mask]
            points_mask = points_mask[fg_mask]
            min_pos, max_pos = np.min(points, axis=0), np.max(points, axis=0)
            coarse_trans = (min_pos + max_pos) / 2.0
            coarse_scale = np.max(max_pos - min_pos) * 2.0 + 1e-6 # scale to [-0.25, 0.25]
            coarse_ss_latents, coarse_ss_latents_mask = self.encode_positions(points, points_mask, ops=['translation', 'scale'],
                                                                params=[-coarse_trans[None], 1. / coarse_scale])
            return coarse_ss_latents, coarse_ss_latents_mask, coarse_trans, coarse_scale
        
        # coarse partial point clouds
        coarse_trans_list, coarse_scale_list = [], []
        coarse_ss_latents_list, coarse_ss_latents_mask_list = [], []

        for i in range(len(points)):
            coarse_ss_latents, coarse_ss_latents_mask, coarse_trans, coarse_scale = encode_coarse_ss(points[i], points_mask[i])
            coarse_ss_latents_list.append(coarse_ss_latents)
            coarse_ss_latents_mask_list.append(coarse_ss_latents_mask)
            coarse_trans_list.append(coarse_trans)
            coarse_scale_list.append(coarse_scale)
        
        coarse_ss_latents = torch.cat(coarse_ss_latents_list)
        coarse_ss_latents_mask = torch.cat(coarse_ss_latents_mask_list)
        
        ss_cond = self.get_ss_cond(instance_image_masked,
                                scene_image_masked, 
                                coarse_ss_latents,
                                coarse_ss_latents_mask)
        # coarse partial point clouds
        
        # coarse sample
        coords, coarse_ss = self.sample_sparse_structure(ss_cond, instance_image_masked.shape[0], 
                                                         sparse_structure_sampler_params, stage='coarse', 
                                                         est_depth_ratio=est_depth_ratio)
        # coarse sample

        # fine point cloud
        fine_trans_list, fine_scale_list = [], []
        fine_ss_latents_list, fine_ss_latents_mask_list = [], []
        for i in range(coarse_ss.shape[0]):
            coarse_comp_vertices = self.vox2pts(coarse_ss[i][None])
            min_pos, max_pos = np.min(coarse_comp_vertices, axis=0), np.max(coarse_comp_vertices, axis=0)
            fine_trans = (min_pos + max_pos) / 2.0
            fine_scale = np.max(max_pos - min_pos) + 1e-6 # scale to [-0.5, 0.5]
            fine_part_ss_latents, fine_part_ss_latents_mask = self.encode_positions(points[i], points_mask[i], ops=['translation', 'scale', 'translation', 'scale'],
                                                                        params=[-coarse_trans_list[i][None], 1. / coarse_scale_list[i], -fine_trans[None], 1. / fine_scale])
            fine_ss_latents_list.append(fine_part_ss_latents)
            fine_ss_latents_mask_list.append(fine_part_ss_latents_mask)
            fine_trans_list.append(fine_trans)
            fine_scale_list.append(fine_scale)
        fine_part_ss_latents = torch.cat(fine_ss_latents_list)
        fine_part_ss_latents_mask = torch.cat(fine_ss_latents_mask_list)

        ss_cond['cond']['cond_partial_vox'] = fine_part_ss_latents
        ss_cond['cond']['cond_partial_vox_mask'] = fine_part_ss_latents_mask

        # fine sample
        coords, ss = self.sample_sparse_structure(ss_cond, instance_image_masked.shape[0], sparse_structure_sampler_params, stage='fine', est_depth_ratio=est_depth_ratio)     
        # fine sample

        uv_list = []
        for i in range(ss.shape[0]):
            inst_coords = torch.argwhere(ss[i][None])[:, [0, 2, 3, 4]].int()
            coords2vertices = (inst_coords[..., 1:] + 0.5) / 64 - 0.5
            coords2vertices2local = transform_vertices(coords2vertices.detach().cpu().numpy(), ops=['scale', 'translation', 'scale', 'translation'], 
                                                    params=[fine_scale_list[i], fine_trans_list[i][None], coarse_scale_list[i], coarse_trans_list[i][None]])
            uv = self.project_uv(coords2vertices2local, extrinsics, intrinsics)
            uv_list.append(uv)
        slat_cond = self.get_slat_cond_parallel(coords, instance_image_masked,
                                        scene_image_masked, uv_list)

        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)

        out_rf = {
            k: [] for k in formats
        }

        for sub_slat in sp.sparse_unbind(slat, dim=0):
            sub_rf = self.decode_slat(sub_slat, formats)
            for k in formats:
                out_rf[k].append(sub_rf[k][0])

        return out_rf, coarse_ss, ss, coarse_trans_list, coarse_scale_list, fine_trans_list, fine_scale_list

    