# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin and Robot labs of Horizon Robotics.

from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            if os.path.exists(os.path.join(root, 'my_metadata/metadata.csv')):
                metadata = pd.read_csv(os.path.join(root, 'my_metadata/metadata.csv'))
            else:
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class StandardSceneDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
        asset_source_list: list[str] = None,
    ):
        super().__init__()
        self.roots = roots.split(',')
        assert len(self.roots) == 2 # 1st for scene images, 2nd for asset instances
        self.scene_image_root = self.roots[0]
        self.asset_root = self.roots[1]
        self.asset_metadata = {}
        if asset_source_list is not None:
            for asset_source in asset_source_list:
                print (f'loading {asset_source}')
                self.asset_metadata[asset_source] = pd.read_csv(os.path.join(self.asset_root, asset_source, 'my_metadata/metadata.csv')).set_index('sha256')
        assert os.path.exists(os.path.join(self.scene_image_root, 'metadata.csv'))
        metadata = pd.read_csv(os.path.join(self.scene_image_root, 'metadata.csv'))
        self._stats = {}
        self._stats['Total'] = len(metadata)
        metadata, stats = self.filter_metadata(metadata)
        self._stats.update(stats)
        self.metadata = metadata
        self.instances = list(self.metadata['example_id'].values)
        self.metadata.set_index('example_id', inplace=True)
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            instance = self.instances[index]
            return self.get_instance(instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}: {stats}')
        return '\n'.join(lines)

class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, only_cond_renders=True, 
                 resize_perturb=False, resize_perturb_ratio=0.5, **kwargs):
        self.image_size = image_size
        self.only_cond_renders = only_cond_renders
        self.resize_perturb = resize_perturb
        self.resize_perturb_ratio = resize_perturb_ratio
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        if self.only_cond_renders:
            metadata = metadata[metadata[f'cond_rendered']]
            stats['Cond rendered'] = len(metadata)
            return metadata, stats
        else:
            if 'cond_rendered' in metadata.columns:
                metadata = metadata[metadata[f'cond_rendered'].fillna(False) | metadata[f'rendered'].fillna(False)]
            else:
                metadata = metadata[metadata[f'rendered'].fillna(False)]
            stats['Cond rendered'] = len(metadata)
            return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        if self.only_cond_renders:
            image_root = os.path.join(root, 'renders_cond', instance)
            with open(os.path.join(image_root, 'transforms.json')) as f:
                metadata = json.load(f)
        else:
            if os.path.exists(os.path.join(root, 'renders_cond', instance, 'transforms.json')):
                image_root = os.path.join(root, 'renders_cond', instance)
                with open(os.path.join(image_root, 'transforms.json')) as f:
                    metadata = json.load(f)
            else:
                image_root = os.path.join(root, 'renders', instance)
                with open(os.path.join(image_root, 'transforms.json')) as f:
                    metadata = json.load(f)

        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        if self.resize_perturb and np.random.rand() < self.resize_perturb_ratio:
            rand_reso = np.random.randint(32, self.image_size)
            image = image.resize((rand_reso, rand_reso), Image.Resampling.LANCZOS)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
    