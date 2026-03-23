import importlib

__attributes = {
    'SparseStructure': 'sparse_structure',
    
    'SparseFeat2Render': 'sparse_feat2render',
    'SLat2Render':'structured_latent2render',
    'Slat2RenderGeo':'structured_latent2render',
    
    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',
    
    'SLat': 'structured_latent',
    'TextConditionedSLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',

    'ImageConditionedSparseStructureLatentRandRot': 'sparse_structure_latent_random_rot',
    'ImageConditionedSLatRandRot': 'structured_latent_random_rot',
    'SparseFeat2RenderRandRot': 'sparse_feat2render_random_rot',
    'Slat2RenderGeoRandRot': 'structured_latent2render_random_rot',
    'SceneImageConditionedVoxel': 'scene_sparse_structure_latent',
    'SceneConditionedSLat': 'scene_structured_latent',
    'SceneImageConditionedVoxelSingle': 'scene_sparse_structure_latent_single',
    'SceneImageConditionedVoxelSingleMixObject': 'scene_sparse_structure_latent_single_mix_obj',
    'ObjectImageConditionedSparseStructureVoxel': 'scene_sparse_structure_latent_obj_pretrain',
    'SceneImageConditionedVoxelSingleDPO': 'scene_sparse_structure_latent_single_dpo',
    'Scene3DFrontConditionedVoxel': 'scene_sparse_structure_latent_single_3dfront',
    'MixOursWith3DFront': 'scene_sparse_structure_latent_mix_data',
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
    from .sparse_structure import SparseStructure
    
    from .sparse_feat2render import SparseFeat2Render
    from .structured_latent2render import (
        SLat2Render,
        Slat2RenderGeo,
    )
    
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
    )
    
    from .structured_latent import (
        SLat,
        TextConditionedSLat,
        ImageConditionedSLat,
    )
    
    # rot mesh
    from .sparse_structure_latent_random_rot import (
        ImageConditionedSparseStructureLatentRandRot
    )

    # rot SLAT
    from .structured_latent_random_rot import (
        ImageConditionedSLatRandRot
    )

    # VAE gs dec
    from .sparse_feat2render_random_rot import (
        SparseFeat2RenderRandRot
    )

    # VAE mesh dec
    from .structured_latent2render_random_rot import (
        Slat2RenderGeoRandRot
    )

    # Scene ss
    from .scene_sparse_structure_latent import (
        SceneImageConditionedVoxel
    )
    # Scene ss single
    from .scene_sparse_structure_latent_single import (
        SceneImageConditionedVoxelSingle
    )
    # 3D-Front
    from .scene_sparse_structure_latent_single_3dfront import (
        Scene3DFrontConditionedVoxel
    )
    # 3D-Front mix ours
    from .scene_sparse_structure_latent_mix_data import (
        MixOursWith3DFront
    )
    # Scene ss mix obj
    from .scene_sparse_structure_latent_single_mix_obj import (
        SceneImageConditionedVoxelSingleMixObject
    )
    # Object data pre-training
    from .scene_sparse_structure_latent_obj_pretrain import (
        ObjectImageConditionedSparseStructureVoxel
    )
    # DPO training
    from .scene_sparse_structure_latent_single_dpo import (
        SceneImageConditionedVoxelSingleDPO
    )

    # Scene Slat
    from .scene_structured_latent import (
        SceneConditionedSLat
    )