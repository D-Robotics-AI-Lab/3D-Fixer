# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .sparse_flow_matching import SparseFlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import SceneImageConditionedProjectMixin

class SceneSparseFlowMatchingTrainer(SparseFlowMatchingTrainer):

    def __init__(
        self,
        *args,
        align_loss_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.align_loss_weight = align_loss_weight

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond_voxel_feat: sp.SparseTensor,
        vis_ratio: torch.Tensor, 
        uv: sp.SparseTensor,
        cond_instance_masked: torch.Tensor,
        std_cond_instance: torch.Tensor,
        cond_scene_masked: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond_scene_masked, cond_instance_masked, std_cond_instance, cond_voxel_feat, uv, **kwargs)
        
        pred, align_loss = self.training_models['denoiser'](x_t, t * 1000, cond, vis_ratio * 1000,
                                                forzen_denoiser=self.frozen_models['frozen_denoiser'],
                                                stage='train', **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["align_loss"] = align_loss
        terms["loss"] = terms["mse"] + terms["align_loss"] * self.align_loss_weight

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        std_sample = []
        cond_vis = None
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            vis_ratio = data.pop('vis_ratio') * 1000.0
            sample_gt.append(data['x_0'])
            vis_cond = self.vis_cond(**data)
            vis_cond_len = 1 if isinstance(vis_cond, dict) else len(vis_cond)
            if cond_vis is None:
                cond_vis = {}
                for vis_cond_index in range(vis_cond_len):
                    cond_vis[f'{vis_cond_index}'] = [vis_cond[vis_cond_index]]
            else:
                for vis_cond_index in range(vis_cond_len):
                    cond_vis[f'{vis_cond_index}'].append(vis_cond[vis_cond_index])
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                vis_ratio=vis_ratio,
                forzen_denoiser=self.frozen_models['frozen_denoiser'],
                stage='infer',
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                vis_ratio=vis_ratio,
                forzen_denoiser=self.frozen_models['frozen_denoiser'],
                stage='infer_std',
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            std_sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        std_sample = sp.sparse_cat(std_sample)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
            'std_sample': {'value': std_sample, 'type': 'sample'},
        }
        for vis_cond_index in range(vis_cond_len):
            sample_dict.update(dict_reduce(cond_vis[f'{vis_cond_index}'], None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
        
        return sample_dict
    
class SceneSparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SceneSparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class SceneImageConditionedSparseFlowMatchingCFGTrainer(SceneImageConditionedProjectMixin, SceneSparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
