import os
from PIL import Image
import json
import numpy as np
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase
from glob import glob
import open3d as o3d


class SLat2RenderRandRot(StandardDatasetBase):
    """
    Dataset for Structured Latent and rendered images.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        latent_model (str): latent model name
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        perturb_ratio: float = 0.5,
        max_mesh_size: float = 50.0,
    ):
        self.image_size = image_size
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.perturb_ratio = perturb_ratio
        self.max_mesh_size = max_mesh_size
        self.value_range = (0, 1)
        
        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_{self.latent_model}'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        if 'rotated_mv_num' in metadata.columns:
            metadata = metadata[metadata['rotated_mv_num'] >= 1]
            stats[f'Num rotated mv renders >= 1'] = len(metadata)
        if 'readable' in metadata.columns:
            metadata = metadata[metadata['readable'] == True]
        else:
            metadata = metadata[metadata['rendered'] == True]
        if 'mesh_size' in metadata.columns:
            metadata = metadata[metadata['mesh_size'] <= self.max_mesh_size]
            stats[f'With mesh smaller than {self.max_mesh_size}'] = len(metadata)
        return metadata, stats

    def _get_image(self, root, instance):
        metadata = None
        image_path_root = os.path.join(root, 'renders', instance)
        rotated_slat_name = None
        if np.random.rand() < self.perturb_ratio:
            # trans_list = glob(os.path.join(root, 'renders_with_rotated_slat', instance, '*/transforms.json'))
            trans_list_index = str(self.metadata.loc[instance]['valid_rots']).split(',')
            trans_list = [
                os.path.join(root, 'renders_with_rotated_slat', instance, f'{idx}/transforms.json') for idx in trans_list_index
            ]
            if len(trans_list) > 0:
                trans_path = np.random.choice(trans_list)
                with open(trans_path) as f:
                    metadata = json.load(f)
                image_path_root = os.path.dirname(trans_path)
                rotated_slat_name = trans_path.split('/')[-2]

        if metadata is None:
            with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
                metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        # image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image_path = os.path.join(image_path_root, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }, rotated_slat_name
    
    def _get_latent(self, root, instance, rotated_slat_name):
        if rotated_slat_name is None:
            data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
            coords = torch.tensor(data['coords']).int()
            feats = torch.tensor(data['feats']).float()
            transforms = None
        else:
            data = np.load(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}', f'{rotated_slat_name}.npz'))
            coords = torch.tensor(data['coords']).int()
            feats = torch.tensor(data['feats']).float()
            with open(os.path.join(root, 'latents_rot', self.latent_model, f'{instance}', 'transforms.json'), 'r') as f:
                js = json.load(f)
            transforms = list(filter(lambda x: x['file_path'] == f'./{rotated_slat_name}.npz', js))[0]
        return {
            'coords': coords,
            'feats': feats,
        }, transforms

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['latents'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        # collate other data
        keys = [k for k in batch[0].keys() if k not in ['coords', 'feats']]
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack

    def get_instance(self, root, instance):
        image, rotated_slat_name = self._get_image(root, instance)
        feat, _ = self._get_latent(root, instance, rotated_slat_name)
        return {
            **image,
            **feat,
        }
        

class Slat2RenderGeoRandRot(SLat2RenderRandRot):
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        perturb_ratio: float = 0.5,
        max_mesh_size: float = 50.0,
    ):
        super().__init__(
            roots,
            image_size,
            latent_model,
            min_aesthetic_score,
            max_num_voxels,
            perturb_ratio,
            max_mesh_size
        )
        
    def _get_geo(self, root, instance, transforms):
        if transforms is None:
            verts, face = utils3d.io.read_ply(os.path.join(root, 'renders', instance, 'mesh.ply'))
            verts = np.nan_to_num(verts)
        else:
            rand_rot = transforms['rotate']
            vertices_trans = np.array(transforms['translation'])
            vertices_scale = transforms['scale']

            mesh = o3d.io.read_triangle_mesh(os.path.join(root, 'renders', instance, 'mesh.ply'))
            vertices = np.nan_to_num(np.asarray(mesh.vertices))
            vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            R1 = mesh.get_rotation_matrix_from_xyz((rand_rot[0], 0, 0))
            R2 = mesh.get_rotation_matrix_from_xyz((0, rand_rot[1], 0))
            R3 = mesh.get_rotation_matrix_from_xyz((0, 0, rand_rot[2]))
            mesh.rotate(R1, center=(0., 0., 0.))
            mesh.rotate(R2, center=(0., 0., 0.))
            mesh.rotate(R3, center=(0., 0., 0.))

            vertices = np.asarray(mesh.vertices)
            vertices = vertices - vertices_trans
            vertices = vertices / vertices_scale
            vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
            
            verts = vertices.astype(np.float32)
            face = np.asarray(mesh.triangles).astype(np.int32)

        mesh = {
            "vertices" : torch.from_numpy(verts),
            "faces" : torch.from_numpy(face),
        }
        return  {
            "mesh" : mesh,
        }
        
    def get_instance(self, root, instance):
        image, rotated_slat_name = self._get_image(root, instance)
        latent, transforms = self._get_latent(root, instance, rotated_slat_name)
        geo = self._get_geo(root, instance, transforms)
        return {
            **image,
            **latent,
            **geo,
        }
        
        