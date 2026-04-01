# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics.

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock, ModulatedSceneSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .scene_sparse_structure_flow import mean_flat
from .structured_latent_flow import SparseResBlock3d, SLatFlowModel
from .sparse_elastic_mixin import SparseTransformerElasticMixin
from . import from_pretrained

class SceneSLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        cond_slat_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_flow_dit: str = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.cond_slat_channels = cond_slat_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.vis_ratio_embedder = TimestepEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        self.input_layer_cond = sp.SparseLinear(cond_slat_channels, model_channels if io_block_channels is None else io_block_channels[0])

        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )

        self.blocks = nn.ModuleList([
            ModulatedSceneSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.control_path = nn.Sequential(*[
            sp.SparseLinear(model_channels, model_channels) for _ in range(num_blocks)
        ])

        self.initialize_weights()
        if pretrained_flow_dit is not None:
            if pretrained_flow_dit.endswith('.pt'):
                print (f'loading pretrained weight: {pretrained_flow_dit}')
                model_ckpt = torch.load(pretrained_flow_dit, map_location='cpu', weights_only=True)
                self.input_layer.load_state_dict(
                    {k.replace('input_layer.', ''): model_ckpt[k] for k in filter(lambda x: 'input_layer' in x, model_ckpt.keys())} 
                )
                self.vis_ratio_embedder.load_state_dict(
                    {k.replace('t_embedder.', ''): model_ckpt[k] for k in filter(lambda x: 't_embedder' in x, model_ckpt.keys())} 
                )
                self.input_blocks.load_state_dict(
                    {k.replace('input_blocks.', ''): model_ckpt[k] for k in filter(lambda x: 'input_blocks' in x, model_ckpt.keys())} 
                )

                for block_index, module in enumerate(self.blocks):
                    module: ModulatedSceneSparseTransformerCrossBlock
                    module.load_state_dict(
                        {k.replace(f'blocks.{block_index}', ''): model_ckpt[k] for k in filter(lambda x: f'blocks.{block_index}' in x, model_ckpt.keys())}, strict=False
                    )
                    module.norm4.load_state_dict(module.norm1.state_dict())
                    module.norm5.load_state_dict(module.norm2.state_dict())
                    module.self_attn_vis_ratio.load_state_dict(module.self_attn.state_dict())
                    module.cross_attn_extra.load_state_dict(module.cross_attn.state_dict())
                    nn.init.constant_(module.self_attn_vis_ratio.to_out.weight, 0)
                    if module.self_attn_vis_ratio.to_out.bias is not None:
                        nn.init.constant_(module.self_attn_vis_ratio.to_out.bias, 0)
                    nn.init.constant_(module.cross_attn_extra.to_out.weight, 0)
                    if module.cross_attn_extra.to_out.bias is not None:
                        nn.init.constant_(module.cross_attn_extra.to_out.bias, 0)
                del model_ckpt
            else:
                print (f'loading pretrained weight: {pretrained_flow_dit}')
                pre_trained_models = from_pretrained(pretrained_flow_dit)
                pre_trained_models: SLatFlowModel

                self.input_layer.load_state_dict(pre_trained_models.input_layer.state_dict())
                self.vis_ratio_embedder.load_state_dict(pre_trained_models.t_embedder.state_dict())
                self.input_blocks.load_state_dict(pre_trained_models.input_blocks.state_dict())

                for block_index, module in enumerate(self.blocks):
                    module: ModulatedSceneSparseTransformerCrossBlock
                    module.load_state_dict(pre_trained_models.blocks[block_index].state_dict(), strict=False)
                    module.norm4.load_state_dict(module.norm1.state_dict())
                    module.norm5.load_state_dict(module.norm2.state_dict())
                    module.self_attn_vis_ratio.load_state_dict(module.self_attn.state_dict())
                    module.cross_attn_extra.load_state_dict(module.cross_attn.state_dict())
                    nn.init.constant_(module.self_attn_vis_ratio.to_out.weight, 0)
                    if module.self_attn_vis_ratio.to_out.bias is not None:
                        nn.init.constant_(module.self_attn_vis_ratio.to_out.bias, 0)
                    nn.init.constant_(module.cross_attn_extra.to_out.weight, 0)
                    if module.cross_attn_extra.to_out.bias is not None:
                        nn.init.constant_(module.cross_attn_extra.to_out.bias, 0)
                del pre_trained_models
        if use_fp16:
            self.convert_to_fp16()

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
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.control_path.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f16)
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

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.vis_ratio_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vis_ratio_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation_vis[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_vis[-1].bias, 0)

        for block in self.control_path:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

    def forward(self, *args, **kwargs):
        stage = kwargs.pop('stage', None)
        if stage == 'train':
            return self._train_forward(*args, **kwargs)
        elif stage == 'infer':
            return self._infer_forward(*args, **kwargs)
        elif stage == 'infer_std':
            return self._infer_std_forward(*args, **kwargs)
        
    def _input_slat(self, x: sp.SparseTensor, emb: torch.Tensor,
                       input_layer: Callable, input_blocks: Callable,
                       pos_embedder: Callable, residual_h: Callable = None
                       ):
        h = input_layer(x).type(self.dtype)
        skips = []
        # pack with input blocks
        for block in input_blocks:
            h = block(h, emb)
            skips.append(h.feats)

        if self.pe_mode == "ape" and pos_embedder is not None:
            h = h + pos_embedder(h.coords[:, 1:]).type(self.dtype)

        if residual_h is not None:
            h = residual_h(h)

        return h, skips
        
    def _train_forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Dict[str,torch.Tensor], vis_ratio: torch.Tensor,
                forzen_denoiser: SLatFlowModel) -> sp.SparseTensor:
        
        t_emb = forzen_denoiser.t_embedder(t)
        if forzen_denoiser.share_mod:
            t_emb = forzen_denoiser.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)

        # moge feats and image mask
        cond_moge = cond['cond_scene']
        cond_dino = cond['cond_instance']
        cond_dino_masked = cond['cond_instance_masked']
        std_cond_dino = cond['std_cond_instance']
        # voxels with projected feats
        x_feat = cond['cond_voxel_feats']

        cond_control = cond_moge
        cond_control = cond_control.type(self.dtype)
        cond_dino_masked = cond_dino_masked.type(self.dtype)
        cond_dino = cond_dino.type(self.dtype)
        std_cond_dino = std_cond_dino.type(self.dtype)

        vis_ratio_emb = self.vis_ratio_embedder(vis_ratio)
        vis_ratio_emb = vis_ratio_emb.type(self.dtype)

        # input layer of frozen part
        h, skips = self._input_slat(x, t_emb, self.input_layer,
                              forzen_denoiser.input_blocks, 
                              forzen_denoiser.pos_embedder if self.pe_mode == "ape" else None)
        # input layer of frozen part

        # condition branch
        ctrl_h, _ = self._input_slat(x_feat, vis_ratio_emb,
                               self.input_layer_cond, self.input_blocks, 
                               forzen_denoiser.pos_embedder if self.pe_mode == "ape" else None)
        # condition branch
        
        std_h = h
        align_loss = 0.0
        acount = 0
        for block_index, block in enumerate(forzen_denoiser.blocks):
            h = block(h, t_emb, cond_dino_masked)
            if block_index < self.num_blocks:
                ctrl_h = self.blocks[block_index](ctrl_h, t_emb, vis_ratio_emb, cond_dino, cond_control)
                h = h + self.control_path[block_index](ctrl_h)
            
            std_h = block(std_h, t_emb, std_cond_dino)

            std_h: sp.SparseTensor
            h: sp.SparseTensor
            for batch_std_h, batch_h in zip(sp.sparse_unbind(std_h, dim=0), sp.sparse_unbind(h, dim=0)):
                acount += 1
                reference_feats = batch_std_h.feats
                source_feats = batch_h.feats
                z_tilde_j = torch.nn.functional.normalize(source_feats, dim=-1, eps=1e-6)
                z_j = torch.nn.functional.normalize(reference_feats, dim=-1, eps=1e-6) 
                align_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        align_loss /= acount

        # unpack with output blocks
        for block, skip in zip(forzen_denoiser.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = forzen_denoiser.out_layer(h.type(x.dtype))
        return h, align_loss

    def _infer_forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Dict[str,torch.Tensor], vis_ratio: torch.Tensor,
                forzen_denoiser: SLatFlowModel) -> sp.SparseTensor:
        
        t_emb = forzen_denoiser.t_embedder(t)
        if forzen_denoiser.share_mod:
            t_emb = forzen_denoiser.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)

        # moge feats and image mask
        cond_moge = cond['cond_scene']
        cond_dino = cond['cond_instance']
        cond_dino_masked = cond['cond_instance_masked']
        # voxels with projected feats
        x_feat = cond['cond_voxel_feats']

        neg_infer = cond.pop("neg_infer", False)

        cond_control = cond_moge
        cond_control = cond_control.type(self.dtype)
        cond_dino = cond_dino.type(self.dtype)
        cond_dino_masked = cond_dino_masked.type(self.dtype)

        vis_ratio_emb = self.vis_ratio_embedder(vis_ratio)
        vis_ratio_emb = vis_ratio_emb.type(self.dtype)

        # input layer of frozen part
        h, skips = self._input_slat(x, t_emb, self.input_layer,
                              forzen_denoiser.input_blocks, 
                              forzen_denoiser.pos_embedder if self.pe_mode == "ape" else None)
        # input layer of frozen part

        # condition branch
        if not neg_infer:
            ctrl_h, _ = self._input_slat(x_feat, vis_ratio_emb, self.input_layer_cond,
                                forzen_denoiser.input_blocks, 
                                forzen_denoiser.pos_embedder if self.pe_mode == "ape" else None)
        # condition branch
        
        for block_index, block in enumerate(forzen_denoiser.blocks):
            h = block(h, t_emb, cond_dino_masked)
            if not neg_infer:
                if block_index < self.num_blocks:
                    ctrl_h = self.blocks[block_index](ctrl_h, t_emb, vis_ratio_emb, cond_dino, cond_control)
                    h = h + self.control_path[block_index](ctrl_h)

        # unpack with output blocks
        for block, skip in zip(forzen_denoiser.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = forzen_denoiser.out_layer(h.type(x.dtype))
        return h
    
    def _infer_std_forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: Dict[str,torch.Tensor], vis_ratio: torch.Tensor,
                forzen_denoiser: SLatFlowModel) -> sp.SparseTensor:
        
        t_emb = forzen_denoiser.t_embedder(t)
        if forzen_denoiser.share_mod:
            t_emb = forzen_denoiser.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)

        cond_dino = cond['std_cond_instance']
        cond_dino = cond_dino.type(self.dtype)

        # input layer of frozen part
        h, skips = self._input_slat(x, t_emb, forzen_denoiser.input_layer,
                              forzen_denoiser.input_blocks, 
                              forzen_denoiser.pos_embedder if self.pe_mode == "ape" else None)
        # input layer of frozen part
        
        for block_index, block in enumerate(forzen_denoiser.blocks):
            h = block(h, t_emb, cond_dino)

        # unpack with output blocks
        for block, skip in zip(forzen_denoiser.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = forzen_denoiser.out_layer(h.type(x.dtype))
        return h
    
class ElasticSceneSLatFlowModel(SparseTransformerElasticMixin, SceneSLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass
