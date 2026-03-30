# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

import importlib

__attributes = {
    'BasicTrainer': 'basic',
        
    'SLatVaeMeshDecoderTrainer': 'vae.structured_latent_vae_mesh_dec',
    
    'FlowMatchingTrainer': 'flow_matching.flow_matching',
    'FlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'ImageConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    
    'SparseFlowMatchingTrainer': 'flow_matching.sparse_flow_matching',
    'SparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'ImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',

    'SLatVaeGaussianDecoderTrainer': 'vae.structured_latent_vae_gaussian_dec',

    'SceneImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.scene_sparse_flow_matching',
    'SceneImageConditionedFlowMatchingCFGTrainer': 'flow_matching.scene_flow_matching',

    'SceneImageConditionedFlowMatchingObjPreTrainCFGTrainer': 'flow_matching.scene_flow_matching_obj_pretrain',

}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import BasicTrainer


    from .vae.structured_latent_vae_gaussian_dec import SLatVaeGaussianDecoderTrainer
    from .vae.structured_latent_vae_mesh_dec import SLatVaeMeshDecoderTrainer
    
    from .flow_matching.flow_matching import (
        FlowMatchingTrainer,
        FlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer,
        ImageConditionedSparseFlowMatchingCFGTrainer,
    )

    from .flow_matching.scene_flow_matching_obj_pretrain import (
        SceneImageConditionedFlowMatchingObjPreTrainCFGTrainer
    )

    from .flow_matching.scene_flow_matching import (
        SceneImageConditionedFlowMatchingCFGTrainer
    )

    from .flow_matching.scene_sparse_flow_matching import (
        SceneImageConditionedSparseFlowMatchingCFGTrainer
    )
