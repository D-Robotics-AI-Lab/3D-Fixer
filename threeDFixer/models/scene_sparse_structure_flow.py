from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import from_pretrained
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import SceneModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
from .sparse_structure_flow import (
    SparseStructureFlowModel,
    TimestepEmbedder
)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class SceneSparseStructureFlowModule(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_ss_flow_dit: str = None,
        resume_ckpts: str = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer_vox_partial = nn.Linear(in_channels * patch_size**3, model_channels)
        self.input_layer_mask_partial = nn.Linear(64, model_channels)

        self.dpt_ratio_embedder = TimestepEmbedder(model_channels)
            
        self.blocks = nn.ModuleList([
            SceneModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
        self.control_path = nn.Sequential(*[
            nn.Linear(model_channels, model_channels) for _ in range(num_blocks)
        ])

        self.neg_cache = {}
        self.cond_vox_cache = None

        self.initialize_weights()
        if pretrained_ss_flow_dit is not None:
            if pretrained_ss_flow_dit.endswith('.pt'):
                print (f'loading pretrained weight: {pretrained_ss_flow_dit}')
                model_ckpt = torch.load(pretrained_ss_flow_dit, map_location='cpu', weights_only=True)
                self.input_layer_vox_partial.load_state_dict(
                    {k.replace('input_layer.', ''): model_ckpt[k] for k in filter(lambda x: 'input_layer' in x, model_ckpt.keys())} 
                )
                self.dpt_ratio_embedder.load_state_dict(
                    {k.replace('t_embedder.', ''): model_ckpt[k] for k in filter(lambda x: 't_embedder' in x, model_ckpt.keys())} 
                )

                for block_index, module in enumerate(self.blocks):
                    module: SceneModulatedTransformerCrossBlock
                    module.load_state_dict(
                        {k.replace(f'blocks.{block_index}', ''): model_ckpt[k] for k in filter(lambda x: f'blocks.{block_index}' in x, model_ckpt.keys())}, strict=False
                    )
                    module.norm4.load_state_dict(module.norm1.state_dict())
                    module.norm5.load_state_dict(module.norm2.state_dict())
                    module.self_attn_dpt_ratio.load_state_dict(module.self_attn.state_dict())
                    module.cross_attn_extra.load_state_dict(module.cross_attn.state_dict())
                    nn.init.constant_(module.self_attn_dpt_ratio.to_out.weight, 0)
                    if module.self_attn_dpt_ratio.to_out.bias is not None:
                        nn.init.constant_(module.self_attn_dpt_ratio.to_out.bias, 0)
                    nn.init.constant_(module.cross_attn_extra.to_out.weight, 0)
                    if module.cross_attn_extra.to_out.bias is not None:
                        nn.init.constant_(module.cross_attn_extra.to_out.bias, 0)
                del model_ckpt
            else:
                print (f'loading pretrained weight: {pretrained_ss_flow_dit}')
                pre_trained_models = from_pretrained(pretrained_ss_flow_dit)
                pre_trained_models: SparseStructureFlowModel

                self.input_layer_vox_partial.load_state_dict(pre_trained_models.input_layer.state_dict())
                self.dpt_ratio_embedder.load_state_dict(pre_trained_models.t_embedder.state_dict())

                for block_index, module in enumerate(self.blocks):
                    module: SceneModulatedTransformerCrossBlock
                    module.load_state_dict(pre_trained_models.blocks[block_index].state_dict(), strict=False)
                    module.norm4.load_state_dict(module.norm1.state_dict())
                    module.norm5.load_state_dict(module.norm2.state_dict())
                    module.self_attn_dpt_ratio.load_state_dict(module.self_attn.state_dict())
                    module.cross_attn_extra.load_state_dict(module.cross_attn.state_dict())
                    nn.init.constant_(module.self_attn_dpt_ratio.to_out.weight, 0)
                    if module.self_attn_dpt_ratio.to_out.bias is not None:
                        nn.init.constant_(module.self_attn_dpt_ratio.to_out.bias, 0)
                    nn.init.constant_(module.cross_attn_extra.to_out.weight, 0)
                    if module.cross_attn_extra.to_out.bias is not None:
                        nn.init.constant_(module.cross_attn_extra.to_out.bias, 0)
                del pre_trained_models
        if resume_ckpts is not None:
            print (f'loading pretrained weight: {resume_ckpts}')
            model_ckpt = torch.load(resume_ckpts, map_location='cpu', weights_only=True)
            self.load_state_dict(model_ckpt, strict=False)
            del model_ckpt
        if use_fp16:
            self.convert_to_fp16()

    def clear_neg_cache(self):
        self.neg_cache = {}
    
    def clear_cond_vox_cache(self):
        self.cond_vox_cache = None

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        self.control_path.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        self.control_path.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.control_path:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation_dpt[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_dpt[-1].bias, 0)

        # Zero-out input layers:
        nn.init.constant_(self.input_layer_mask_partial.weight, 0)
        nn.init.constant_(self.input_layer_mask_partial.bias, 0)

    def input_voxel(self, x, input_layer, pos_emb):
        ########## voxel tokens
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = input_layer(h)
        h = h + pos_emb
        ########## voxel tokens
        return h
    
    def input_mask(self, x, input_layer):
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = input_layer(h)
        return h

    def forward(self, *args, **kwargs):
        if kwargs.pop("w_align_loss", False):
            return self._train_forward(*args, **kwargs, w_align_loss=True)
        else:
            return self._infer_forward(*args, **kwargs)
        
    def _train_forward(self, x: torch.Tensor, t: torch.Tensor, cond: Dict[str,torch.Tensor], 
                forzen_denoiser: SparseStructureFlowModel, est_depth_ratio: torch.Tensor, 
                w_align_loss: bool = False) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        
        h = self.input_voxel(x, forzen_denoiser.input_layer, forzen_denoiser.pos_emb[None])

        cond_vox = self.input_voxel(cond['cond_partial_vox'], self.input_layer_vox_partial, forzen_denoiser.pos_emb[None]) + \
                    self.input_mask(cond['cond_partial_vox_mask'], self.input_layer_mask_partial)

        cond_moge = cond['cond_scene']
        cond_dino = cond['cond_instance']
        cond_dino_masked = cond['cond_instance_masked']
        if w_align_loss:
            std_cond_dino = cond['std_cond_instance']
            std_cond_dino = std_cond_dino.type(self.dtype)
            std_h = h
            std_h = std_h.type(self.dtype)

        t_emb = forzen_denoiser.t_embedder(t)
        if self.share_mod:
            t_emb = forzen_denoiser.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        est_depth_ratio_emb = self.dpt_ratio_embedder(est_depth_ratio)
        est_depth_ratio_emb = est_depth_ratio_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond_control = cond_moge
        cond_control = cond_control.type(self.dtype)
        cond_vox = cond_vox.type(self.dtype)
        cond_dino = cond_dino.type(self.dtype)
        cond_dino_masked = cond_dino_masked.type(self.dtype)

        align_loss = 0.0
        acount = 0
        for block_index, frozen_block in enumerate(forzen_denoiser.blocks):
            h = frozen_block(h, t_emb, cond_dino_masked)
            if block_index < len(self.blocks):
                cond_vox = self.blocks[block_index](cond_vox, t_emb, est_depth_ratio_emb, cond_dino, cond_control)
                ctrl_feats = self.control_path[block_index](cond_vox)
                h = h + ctrl_feats

            if w_align_loss:
                with torch.no_grad():
                    std_h = frozen_block(std_h, t_emb, std_cond_dino)
                acount += 1
                reference = std_h
                source = h
                
                z_tilde_j = torch.nn.functional.normalize(source, dim=-1, eps=1e-6)
                z_j = torch.nn.functional.normalize(reference, dim=-1, eps=1e-6) 
                align_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))

        h = h.type(x.dtype)

        h = F.layer_norm(h, h.shape[-1:])
        h = forzen_denoiser.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        if w_align_loss:
            return h, align_loss / acount
        else:
            return h

    def _infer_forward(self, x: torch.Tensor, t: torch.Tensor, cond: Dict[str,torch.Tensor], 
                forzen_denoiser: SparseStructureFlowModel, est_depth_ratio: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
                
        h = self.input_voxel(x, forzen_denoiser.input_layer, forzen_denoiser.pos_emb[None])
        cond_vox = self.input_voxel(cond['cond_partial_vox'], self.input_layer_vox_partial, forzen_denoiser.pos_emb[None]) + \
                    self.input_mask(cond['cond_partial_vox_mask'], self.input_layer_mask_partial)

        cond_moge = cond['cond_scene']
        cond_dino = cond['cond_instance']
        cond_dino_masked = cond['cond_instance_masked']

        t_emb = forzen_denoiser.t_embedder(t)
        if self.share_mod:
            t_emb = forzen_denoiser.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        est_depth_ratio_emb = self.dpt_ratio_embedder(est_depth_ratio)
        est_depth_ratio_emb = est_depth_ratio_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond_control = cond_moge
        cond_control = cond_control.type(self.dtype)
        cond_vox = cond_vox.type(self.dtype)
        cond_dino = cond_dino.type(self.dtype)
        cond_dino_masked = cond_dino_masked.type(self.dtype)

        for block_index, frozen_block in enumerate(forzen_denoiser.blocks):
            h = frozen_block(h, t_emb, cond_dino_masked)
            if block_index < len(self.blocks):
                cond_vox = self.blocks[block_index](cond_vox, t_emb, est_depth_ratio_emb, cond_dino, cond_control)
                ctrl_feats = self.control_path[block_index](cond_vox)
                h = h + ctrl_feats 

        h = h.type(x.dtype)

        h = F.layer_norm(h, h.shape[-1:])
        h = forzen_denoiser.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h