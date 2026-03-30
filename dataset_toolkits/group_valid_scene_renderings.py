# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import copy
import sys
import importlib
import argparse
import pandas as pd
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import copy
import fsspec
import numpy as np
from icecream import ic
import imageio
from concurrent.futures import ThreadPoolExecutor

def get_metadata(asset_metadata_dir: str):
    try:
        metadata = pd.read_csv(os.path.join(asset_metadata_dir, 'my_metadata/metadata.csv'))
    except FileNotFoundError:
        metadata = pd.read_csv(os.path.join(asset_metadata_dir, 'metadata.csv'))
    source_name = os.path.basename(asset_metadata_dir.strip('/'))
    if 'uid' in metadata.columns:
        metadata.drop('uid', axis=1, inplace=True)
    metadata = metadata[~metadata['local_path'].isna()]
    metadata = metadata[~metadata['rendered'].isna()]
    metadata['from'] = [source_name] * len(metadata)
    return metadata

def get_instance_mask(instance_mask_path):
    index_mask = imageio.v3.imread(instance_mask_path)
    index_mask = np.rint(index_mask.astype(np.float32) / 65535 * 100.0) # hand coded, max obj nums = 100
    instance_list = np.unique(index_mask).astype(np.uint8)
    instance_list = instance_list[instance_list != 0] # 0 for background
    instance_list = instance_list[instance_list != 1] # 1 for floor
    instance_list = instance_list[instance_list != 2] # 0 for wall
    return index_mask, instance_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_root_dir', type=str, required=True,
                        help='Directory to load the scene data.')
    parser.add_argument('--obj_asset_dir', type=str, required=True,
                        help='Directory to each object dataset.')
    parser.add_argument('--obj_root_dir', type=str, required=True,
                        help='Directory to the root path of object subsets.')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    obj_asset_dir = opt.obj_asset_dir
    root_dir = opt.obj_root_dir
    
    # get all metadata
    asset_sources = obj_asset_dir.split(',')
    all_asset_metadata = []
    asset_sha256_rec = {}
    for asset_dir in asset_sources:
        assert root_dir in asset_dir
        dataset_name = os.path.basename(asset_dir.strip('/'))
        asset_metadata = get_metadata(asset_dir)
        all_asset_metadata.append(asset_metadata)
        asset_sha256_rec[dataset_name] = set()
    metadata = pd.concat(all_asset_metadata)
    metadata.set_index('sha256', inplace=True)

    # rendered images
    scene_root_dir = opt.scene_root_dir
    fs, path = fsspec.core.url_to_fs(scene_root_dir)
    transforms_list = fs.glob(
        scene_root_dir + f"/*/transforms.json"
    )

    records = []
    max_workers = 256
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(transforms_list), desc="grouping valid training data ...") as pbar:
            def worker(transform_fpath: str):
                try:
                    scene_sha256 = transform_fpath.split('/')[-2]
                    scene_dir = os.path.dirname(transform_fpath)
                    with open(transform_fpath, 'r') as f:
                        js = json.load(f)
                    for frame_info in js['frames']:
                        frame_index = frame_info['file_path'][:3]
                        index_mask, instance_list = get_instance_mask(os.path.join(scene_dir, frame_index + '_index.png'))
                        if len(instance_list) == 0:
                            continue
                        H, W = index_mask.shape[:2]
                        for instance_idx in instance_list:
                            index_mask_ = copy.deepcopy(index_mask)
                            pixel_ratio = np.sum((index_mask_ == instance_idx).astype(np.float32)) / (H * W)           
                            instance_sha256 = js['instance'][f'{instance_idx}']['sha256']
                            item_rec = metadata.loc[instance_sha256]
                            asset_sha256_rec[item_rec['from']].add(instance_sha256)
                            rec = {
                                'example_id': f'{scene_sha256}/{frame_index}/{instance_sha256}/{instance_idx}',
                                'pixel_ratio': pixel_ratio,
                                'from': item_rec['from'],
                                'local_path': item_rec['local_path'],
                                'frame_path': os.path.join(scene_sha256, frame_index),
                                'rendered': item_rec['rendered'],
                                'readable': item_rec['readable'],
                                'mesh_size': item_rec['mesh_size'],
                            }
                            records.append(rec)
                    pbar.update()
                except Exception as e:
                    pbar.update()
                    print (f'Caught exception {e} during checking ' + rec['sha256'])
            
            executor.map(worker, transforms_list)
            executor.shutdown(wait=True)
        training_metadata = pd.DataFrame.from_records(records)
        training_metadata.set_index('example_id', inplace=True)
        if len(training_metadata) > 0:
            training_metadata.to_csv(os.path.join(scene_root_dir, "metadata.csv"))
            with open(os.path.join(scene_root_dir, 'statistics.txt'), 'w') as f:
                f.write('Statistics:\n')
                f.write(f'  - Number of scenes: {len(transforms_list)}\n')
                f.write(f'  - Number of examples: {len(training_metadata)}\n')
                total_asset = 0
                for dataset_name, rec in asset_sha256_rec.items():
                    total_asset += len(rec)
                    f.write(f'  - Number of assets from {dataset_name}: {len(rec)}\n')
                f.write(f'  - Number of assets in total: {total_asset}\n')
    except Exception as e:
        print (f'encountered exception: {e}')
