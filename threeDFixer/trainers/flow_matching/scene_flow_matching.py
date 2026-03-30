# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingTrainer
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import BatchSceneImageConditionedMixin


class SceneFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
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
    """

    def __init__(
        self,
        *args,
        align_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.align_loss_weight = align_loss_weight

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond_vox: torch.Tensor,
        cond_mask: torch.Tensor,
        est_depth_frac: torch.Tensor,
        cond_instance_masked: torch.Tensor,
        cond_scene_masked: torch.Tensor,
        std_cond_instance: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        if self.online_encode_z0:
            x_0 = self.frozen_models['encoder'](x_0.float())
            cond_vox = self.frozen_models['encoder'](cond_vox.float())
            mean = kwargs.pop('mean', None)
            std = kwargs.pop('std', None)
            if mean is not None:
                x_0 = (x_0 - mean) / std
                cond_vox = (cond_vox - mean) / std

        terms = edict()
        with torch.no_grad():
            noise = torch.randn_like(x_0)
            t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
            x_t = self.diffuse(x_0, t, noise=noise)
            cond = self.get_cond(cond_scene_masked, cond_instance_masked, 
                                 std_cond_instance if self.align_loss_weight > 0.0 else None, cond_vox,
                                cond_mask.float(), **kwargs)

        pred = self.training_models['denoiser'](
            x_t, t * 1000, cond, est_depth_ratio = est_depth_frac * 1000,
            forzen_denoiser=self.frozen_models['frozen_denoiser'],
            w_align_loss=self.align_loss_weight > 0.0, **kwargs)
        
        if self.align_loss_weight > 0.0:
            pred, align_loss = pred
        
        assert pred.shape == noise.shape == x_0.shape

        target = self.get_v(x_0, noise, t)
        terms["mse"] = F.mse_loss(pred, target)

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        terms["loss"] = terms["mse"] 
        if self.align_loss_weight > 0.0:
            terms["align_loss"] = align_loss.mean()
            terms["loss"] += terms["align_loss"] * self.align_loss_weight

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

        # fine
        sample_gt = []
        sample_part_cond_vox = []
        sample = []
        
        cond_vis = None
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}

            x_0 = data['x_0']
            x_0_partial_cond = data.pop('cond_vox')
            est_depth_frac = data.pop('est_depth_frac') * 1000
            sample_part_cond_vox.append(x_0_partial_cond.float().cuda())
            x_0_partial_cond_mask = data.pop('cond_mask')
            mean = data.pop('mean', None)
            std = data.pop('std', None)
            if self.online_encode_z0:
                x_0 = self.frozen_models['encoder'](x_0.float().cuda())
                x_0_partial_cond = self.frozen_models['encoder'](x_0_partial_cond.float().cuda())
                if mean is not None:
                    x_0 = (x_0 - mean) / std
                    x_0_partial_cond = (x_0_partial_cond - mean) / std

            noise = torch.randn_like(x_0)
            sample_gt.append(data['x_0'].float().cuda())
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
            del data['std_cond_instance']
            args = self.get_inference_cond(**data, x_0_partial_cond=x_0_partial_cond, 
                                           x_0_partial_cond_mask=x_0_partial_cond_mask)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                est_depth_ratio=est_depth_frac,
                steps=50, cfg_strength=3.0, verbose=verbose,
                forzen_denoiser=self.frozen_models['frozen_denoiser'],
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_part_cond_vox = torch.cat(sample_part_cond_vox, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
            'sample_part_cond_vox': {'value': sample_part_cond_vox, 'type': 'sample'},
        }
        for vis_cond_index in range(vis_cond_len):
            sample_dict.update(dict_reduce(cond_vis[f'{vis_cond_index}'], None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
        
        return sample_dict

    
class SceneFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SceneFlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
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



class SceneImageConditionedFlowMatchingCFGTrainer(BatchSceneImageConditionedMixin, SceneFlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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
