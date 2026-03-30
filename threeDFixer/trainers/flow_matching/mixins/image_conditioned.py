# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics.

from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from ....datasets.utils import lstsq_align_depth, voxelize_pcd_pt, rot_vertices_torch, transform_vertices
from ....utils import dist_utils
from ....moge.model.v2 import MoGeModel
from ....modules.sparse.basic import SparseTensor, sparse_unbind
from ....representations import MeshExtractResult

from einops import rearrange, repeat

class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
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
        
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}

class BatchObjPreTrainImageConditionedMixin:
    """
    Mixin for pre-training of ss image-conditioned models with object-level data.
    
    Args:
        image_cond_model: The image conditioning model.
        moge_ckpts: The path to load MoGe v2 model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', moge_ckpts: str = '', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.moge_ckpts = moge_ckpts
        self.moge_model = None
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }

    def _init_moge_model(self):
        with dist_utils.local_master_first():
            moge_model = MoGeModel.from_pretrained(self.moge_ckpts)
        moge_model.eval().cuda()
        self.moge_model = moge_model

    def _get_dino_feats(self, image):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats(self, image):
        if self.moge_model is None:
            self._init_moge_model()
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.moge_model.infer_feature_tokens(image, num_tokens=num_tokens, tokens_layer=0)
        features = rearrange(features, 'b c h w -> b (h w) c')
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
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
    
    def _batch_encode_feats(self, cond_scene_masked, cond_instance_masked):
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

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)

        batch_dino_feats = self.encode_image(torch.cat(batch_inputs), enc_fn=self._get_dino_feats)
        cond_instance_masked, cond_instance, cond_instance_mask, cond_scene_mask = batch_dino_feats.chunk(4)

        cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_scene = torch.cat([cond_scene, cond_scene_mask], dim=1)

        pack = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
        }
        return pack
    
    def _render_gt_depth(self, meshes, intrinsics, extrinsics, renderer):
        
        gt_depths = []
        for i in range(intrinsics.shape[0]):
            mesh = MeshExtractResult(
                meshes[i]['verts'].to(self.device),
                meshes[i]['faces'].to(self.device)
            )
            rends = renderer.render(mesh, extrinsics[i].to(dtype=torch.float32, device=self.device),
                                    intrinsics[i].to(dtype=torch.float32, device=self.device), return_types=['depth'])
            gt_depths.append(rends['depth'])

        return gt_depths
    
    @torch.no_grad
    def _est_depth(self, cond_scene_masked: torch.Tensor):
        assert cond_scene_masked.ndim == 4
        output = self.moge_model.infer((cond_scene_masked[:, :3] * cond_scene_masked[:, -1:]).float())
        return output['depth']
        
    def get_cond(self, cond_scene_masked, cond_instance_masked,
                 mesh, perturb_rot, perturb_trans, 
                 perturb_scale, est_depth_frac, 
                 intrinsics, extrinsics, rays_o,
                 rays_d, fg_mask, edge_mask, renderer,
                 enc_vox_fn, 
                 **kwargs):
        """
        Get the conditioning data.
        """

        feats_pack = self._batch_encode_feats(cond_scene_masked, cond_instance_masked)
        gt_depths = self._render_gt_depth(mesh, intrinsics, extrinsics, renderer)
        est_depths = self._est_depth(cond_scene_masked)

        cond_vox, cond_vox_mask = [], []
        for i in range(len(gt_depths)):
            try:
                aligned_est_depth = lstsq_align_depth(est_depths[i].squeeze(),
                                                    gt_depths[i].squeeze(),
                                                    fg_mask[i].squeeze())
                mask_color = fg_mask[i].squeeze().float() + edge_mask[i].squeeze().float() * 0.5
                mix_depth = aligned_est_depth * est_depth_frac[i] + gt_depths[i].squeeze() * (1. - est_depth_frac[i])
                valid_mask = torch.nonzero(fg_mask[i])

                pcd_points = (rays_o[i][valid_mask[:, 1], valid_mask[:, 0]] + rays_d[i][valid_mask[:, 1], valid_mask[:, 0]] * mix_depth[valid_mask[:, 0], valid_mask[:, 1]][..., None])
                pcd_colors = mask_color[valid_mask[:, 0], valid_mask[:, 1]]

                valid_pcd_mask = ~torch.logical_or(
                    (pcd_points < -0.5).any(dim=-1),
                    (pcd_points > 0.5).any(dim=-1),
                )
                pcd_points = pcd_points[valid_pcd_mask]
                pcd_colors = pcd_colors[valid_pcd_mask]

                if pcd_points.shape[0] < 10:
                    vox = torch.zeros(1, 64, 64, 64, dtype=torch.long, device='cuda')
                    vox_mask = torch.zeros(64, 16, 16, 16, dtype=torch.float32, device='cuda')
                else:
                    partial_cond_points = rot_vertices_torch(pcd_points, perturb_rot[i], ['z', 'y', 'x'])
                    partial_cond_points = transform_vertices(partial_cond_points, ops=['scale', 'translation'],
                                                                    params=[perturb_scale[i], perturb_trans[i]])

                    vox, vox_mask = voxelize_pcd_pt(partial_cond_points, pcd_colors, clip_range_first=True, return_mask=True, resolution=64)
            except Exception as e:
                print (f'exception {e} in trainer.', flush=True)
                vox = torch.zeros(1, 64, 64, 64, dtype=torch.long, device='cuda')
                vox_mask = torch.zeros(64, 16, 16, 16, dtype=torch.float32, device='cuda')
            cond_vox.append(vox)
            cond_vox_mask.append(vox_mask)
        
        x_0_partial_cond = enc_vox_fn(torch.stack(cond_vox))
        x_0_partial_cond_mask = torch.stack(cond_vox_mask)

        cond = {
            **feats_pack,
            'cond_partial_vox': x_0_partial_cond, 
            'cond_partial_vox_mask': x_0_partial_cond_mask,
        }
        kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond_scene_masked, cond_instance_masked,
                 mesh, perturb_rot, perturb_trans, 
                 perturb_scale, est_depth_frac,
                 intrinsics, extrinsics, rays_o,
                 rays_d, fg_mask, edge_mask, renderer,
                 enc_vox_fn, **kwargs):
        """
        Get the conditioning data.
        """
        
        feats_pack = self._batch_encode_feats(cond_scene_masked, cond_instance_masked)

        gt_depths = self._render_gt_depth(mesh, intrinsics, extrinsics, renderer)
        est_depths = self._est_depth(cond_scene_masked)

        cond_vox, cond_vox_mask = [], []
        for i in range(len(gt_depths)):
            try:
                aligned_est_depth = lstsq_align_depth(est_depths[i].squeeze(),
                                                    gt_depths[i].squeeze(),
                                                    fg_mask[i].squeeze())
                mask_color = fg_mask[i].squeeze().float() + edge_mask[i].squeeze().float() * 0.5
                mix_depth = aligned_est_depth * est_depth_frac[i] + gt_depths[i].squeeze() * (1. - est_depth_frac[i])
                valid_mask = torch.nonzero(fg_mask[i])

                pcd_points = (rays_o[i][valid_mask[:, 1], valid_mask[:, 0]] + rays_d[i][valid_mask[:, 1], valid_mask[:, 0]] * mix_depth[valid_mask[:, 0], valid_mask[:, 1]][..., None])
                pcd_colors = mask_color[valid_mask[:, 0], valid_mask[:, 1]]

                valid_pcd_mask = ~torch.logical_or(
                    (pcd_points < -0.5).any(dim=-1),
                    (pcd_points > 0.5).any(dim=-1),
                )
                pcd_points = pcd_points[valid_pcd_mask]
                pcd_colors = pcd_colors[valid_pcd_mask]

                if pcd_points.shape[0] < 10:
                    vox = torch.zeros(1, 64, 64, 64, dtype=torch.long, device='cuda')
                    vox_mask = torch.zeros(64, 16, 16, 16, dtype=torch.float32, device='cuda')
                else:
                    partial_cond_points = rot_vertices_torch(pcd_points, perturb_rot[i], ['z', 'y', 'x'])
                    partial_cond_points = transform_vertices(partial_cond_points, ops=['scale', 'translation'],
                                                                    params=[perturb_scale[i], perturb_trans[i]])

                    vox, vox_mask = voxelize_pcd_pt(partial_cond_points, pcd_colors, clip_range_first=True, return_mask=True, resolution=64)
            except Exception as e:
                print (f'exception {e} in trainer.', flush=True)
                vox = torch.zeros(1, 64, 64, 64, dtype=torch.long, device='cuda')
                vox_mask = torch.zeros(64, 16, 16, 16, dtype=torch.float32, device='cuda')
            cond_vox.append(vox)
            cond_vox_mask.append(vox_mask)
        
        x_0_partial_cond = enc_vox_fn(torch.stack(cond_vox))
        x_0_partial_cond_mask = torch.stack(cond_vox_mask)
        
        cond = {
            **feats_pack,
            'cond_partial_vox': x_0_partial_cond, 
            'cond_partial_vox_mask': x_0_partial_cond_mask,
        }
        kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond_scene_masked, cond_instance_masked, **kwargs):
        """
        Visualize the conditioning data.
        """
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        return {'image_scene': {'value': cond_scene, 'type': 'image'}},\
               {'image_scene_masked': {'value': cond_scene * cond_scene_mask, 'type': 'image'}}, \
               {'image_instance': {'value': cond_instance, 'type': 'image'}}, \
               {'image_instance_masked': {'value': cond_instance * cond_instance_mask, 'type': 'image'}}

class BatchSceneImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', moge_ckpts: str = '', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.moge_ckpts = moge_ckpts
        self.moge_model = None
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }

    def _init_moge_model(self):
        with dist_utils.local_master_first():
            moge_model = MoGeModel.from_pretrained(self.moge_ckpts)
        moge_model.eval().cuda()
        self.moge_model = moge_model

    def _get_dino_feats(self, image):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats(self, image):
        if self.moge_model is None:
            self._init_moge_model()
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.moge_model.infer_feature_tokens(image, num_tokens=num_tokens, tokens_layer=0)
        features = rearrange(features, 'b c h w -> b (h w) c')
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
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
        
    def get_cond(self, cond_scene_masked, cond_instance_masked, std_cond_instance, 
                 x_0_partial_cond, x_0_partial_cond_mask, **kwargs):
        """
        Get the conditioning data.
        """

        batch_inputs = []
        if std_cond_instance is not None:
            batch_inputs.append(std_cond_instance)

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

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)

        batch_dino_feats = self.encode_image(torch.cat(batch_inputs), enc_fn=self._get_dino_feats)
        if len(batch_inputs) == 5:
            std_cond_instance, cond_instance_masked, cond_instance, cond_instance_mask, cond_scene_mask = batch_dino_feats.chunk(5)
        else:
            cond_instance_masked, cond_instance, cond_instance_mask, cond_scene_mask = batch_dino_feats.chunk(4)

        cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_scene = torch.cat([cond_scene, cond_scene_mask], dim=1)
        
        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'cond_partial_vox': x_0_partial_cond, 
            'cond_partial_vox_mask': x_0_partial_cond_mask,
        }
        if std_cond_instance is not None:
            cond['std_cond_instance'] = std_cond_instance
        kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond_scene_masked, cond_instance_masked, 
                 x_0_partial_cond, x_0_partial_cond_mask, **kwargs):
        """
        Get the conditioning data.
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

        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)

        batch_dino_feats = self.encode_image(torch.cat(batch_inputs), enc_fn=self._get_dino_feats)
        cond_instance_masked, cond_instance, cond_instance_mask, cond_scene_mask = batch_dino_feats.chunk(4)

        cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_scene = torch.cat([cond_scene, cond_scene_mask], dim=1)
        
        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'cond_partial_vox': x_0_partial_cond, 
            'cond_partial_vox_mask': x_0_partial_cond_mask,
        }
        kwargs['neg_cond'] = {k: torch.zeros_like(v) for k, v in cond.items()}
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond_scene_masked, cond_instance_masked, std_cond_instance=None, **kwargs):
        """
        Visualize the conditioning data.
        """
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        if std_cond_instance is None:
            return {'image_scene': {'value': cond_scene, 'type': 'image'}},\
                {'image_scene_masked': {'value': cond_scene * cond_scene_mask, 'type': 'image'}}, \
                {'image_instance': {'value': cond_instance, 'type': 'image'}}, \
                {'image_instance_masked': {'value': cond_instance * cond_instance_mask, 'type': 'image'}}
        else:
            return {'image_scene': {'value': cond_scene, 'type': 'image'}},\
                {'image_scene_masked': {'value': cond_scene * cond_scene_mask, 'type': 'image'}}, \
                {'image_instance': {'value': cond_instance, 'type': 'image'}}, \
                {'image_instance_masked': {'value': cond_instance * cond_instance_mask, 'type': 'image'}}, \
                {'std_image_instance': {'value': std_cond_instance, 'type': 'image'}}

class SceneImageConditionedProjectMixin:

    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', moge_ckpts: str = '', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.moge_ckpts = moge_ckpts
        self.moge_model = None
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }

    def _init_moge_model(self):
        with dist_utils.local_master_first():
            moge_model = MoGeModel.from_pretrained(self.moge_ckpts)
        moge_model.eval().cuda()
        self.moge_model = moge_model

    def _get_dino_feats(self, image):
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def _get_moge_feats(self, image):
        if self.moge_model is None:
            self._init_moge_model()
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.moge_model.infer_feature_tokens(image, num_tokens=num_tokens)
        features = rearrange(features, 'b c (h n1) (w n2) -> b (h w) (n1 n2 c)', n1=16, n2=16)
        features = torch.nn.functional.adaptive_avg_pool1d(
            features, 1024
        )
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
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

    def project_feats(self, cond_voxel_feat: SparseTensor, feats: torch.Tensor, uv):
        uvs = sparse_unbind(uv, dim=0)
        sample_feats = []
        for batch_index, batch_uv in enumerate(uvs):
            sample_feat = torch.nn.functional.grid_sample(
                    feats[batch_index].unsqueeze(0).contiguous(),
                    batch_uv.feats.unsqueeze(0).unsqueeze(0),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).squeeze(0).permute(1, 0)
            sample_feats.append(sample_feat)
        cond_voxel_feat = cond_voxel_feat.replace(torch.cat(sample_feats))
        return cond_voxel_feat
    
    def _get_moge_feats_hw(self, image):
        if self.moge_model is None:
            self._init_moge_model()
        patch_size = 14 # hand coded
        H, W = image.shape[-2:]
        num_tokens = (H // patch_size) * (W // patch_size)
        features = self.moge_model.infer_feature_tokens(image, num_tokens=num_tokens)
        h, w = features.shape[-2:]
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = F.layer_norm(features, features.shape[-1:])
        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)
        return features

    def get_cond(self, cond_scene_masked, cond_instance_masked, std_cond_instance, 
                 cond_voxel_feat, uv, **kwargs):
        """
        Get the conditioning data.
        """
        std_cond_instance = self.encode_image(std_cond_instance, enc_fn=self._get_dino_feats)

        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)

        cond_instance_masked = self.encode_image(cond_instance * cond_instance_mask, enc_fn=self._get_dino_feats)

        cond_instance = self.encode_image(cond_instance, enc_fn=self._get_dino_feats)
        cond_instance_mask = self.encode_image(repeat(cond_instance_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        # cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_instance = (cond_instance + cond_instance_mask) / 2.0
        
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        cond_scene_hw = self._get_moge_feats_hw(cond_scene)
        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)
        cond_scene_mask = self.encode_image(repeat(cond_scene_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        # cond_scene = (cond_scene + cond_scene_mask) / 2.0
        cond_scene_mask[:, -cond_scene.shape[-2]:] = (cond_scene_mask[:, -cond_scene.shape[-2]:] + cond_scene) / 2.0
        cond_scene = cond_scene_mask

        cond_voxel_feat = self.project_feats(cond_voxel_feat, cond_scene_hw, uv)
        
        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'std_cond_instance': std_cond_instance,
            'cond_voxel_feats': cond_voxel_feat
        }
        kwargs['neg_cond'] = {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v.replace(torch.zeros_like(v.feats))) for k, v in cond.items()}
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond_scene_masked, cond_instance_masked, 
                 cond_voxel_feat, uv, std_cond_instance, **kwargs):
        """
        Get the conditioning data.
        """
        std_cond_instance = self.encode_image(std_cond_instance, enc_fn=self._get_dino_feats)
        
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)

        cond_instance_masked = self.encode_image(cond_instance * cond_instance_mask, enc_fn=self._get_dino_feats)

        cond_instance = self.encode_image(cond_instance, enc_fn=self._get_dino_feats)
        cond_instance_mask = self.encode_image(repeat(cond_instance_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        # cond_instance = torch.cat([cond_instance, cond_instance_mask], dim=1)
        cond_instance = (cond_instance + cond_instance_mask) / 2.0
        
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        cond_scene_hw = self._get_moge_feats_hw(cond_scene)
        cond_scene = self.encode_image(cond_scene, enc_fn=self._get_moge_feats)
        cond_scene_mask = self.encode_image(repeat(cond_scene_mask, 'b c ... -> b (n c) ...', n=3), enc_fn=self._get_dino_feats)
        # cond_scene = (cond_scene + cond_scene_mask) / 2.0
        cond_scene_mask[:, -cond_scene.shape[-2]:] = (cond_scene_mask[:, -cond_scene.shape[-2]:] + cond_scene) / 2.0
        cond_scene = cond_scene_mask

        cond_voxel_feat = self.project_feats(cond_voxel_feat, cond_scene_hw, uv)
        
        cond = {
            'cond_scene': cond_scene,
            'cond_instance': cond_instance,
            'cond_instance_masked': cond_instance_masked,
            'std_cond_instance': std_cond_instance,
            'cond_voxel_feats': cond_voxel_feat
        }
        kwargs['neg_cond'] = {k: (torch.zeros_like(v) if isinstance(v, torch.Tensor) else v.replace(torch.zeros_like(v.feats))) for k, v in cond.items()}
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
    
    def vis_cond(self, cond_scene_masked, cond_instance_masked, std_cond_instance, **kwargs):
        """
        Visualize the conditioning data.
        """
        cond_instance, cond_instance_mask = cond_instance_masked.split([3, 1], dim=1)
        cond_scene, cond_scene_mask = cond_scene_masked.split([3, 1], dim=1)
        return {'image_scene': {'value': cond_scene, 'type': 'image'}},\
               {'image_scene_masked': {'value': cond_scene * cond_scene_mask, 'type': 'image'}}, \
               {'image_instance': {'value': cond_instance, 'type': 'image'}}, \
               {'image_instance_masked': {'value': cond_instance * cond_instance_mask, 'type': 'image'}}, \
               {'std_image_instance': {'value': std_cond_instance, 'type': 'image'}}
    