# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import cv2
import json
import torch
import imageio

from depth_anything_v2.dpt import DepthAnythingV2

import numpy as np
import fsspec
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse

def get_gt_depth(gt_depth_path, metadata):
    gt_depth = imageio.v3.imread(gt_depth_path).astype(np.float32) / 65535.
    depth_min, depth_max = metadata['depth']['min'], metadata['depth']['max']
    gt_depth = gt_depth * (depth_max - depth_min) + depth_min
    return gt_depth

def get_scene_mask(instance_mask_path):
    index_mask = imageio.v3.imread(instance_mask_path)
    index_mask = np.rint(index_mask.astype(np.float32) / 65535 * 100.0) # hand coded, max obj nums = 100
    scene_mask = (index_mask != 0).astype(np.float32)
    return scene_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpt_model_dir', type=str, required=True,
                        help='Directory to load the Depth-Anything v2.')
    parser.add_argument('--dpt_encoder', type=str, default='vitl',
                        help='Encoder version.')
    parser.add_argument('--scene_root_dir', type=str, required=True,
                        help='Directory to load the scene data.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    device = torch.device("cuda")
    # Load model and preprocessing transform
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = opt.dpt_encoder

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'{opt.dpt_model_dir}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    model_type = f"DAv2_{encoder}"

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    scene_root_dir = opt.scene_root_dir
    if os.path.exists(os.path.join(scene_root_dir, "transforms_list.json")):
        import json
        with open(os.path.join(scene_root_dir, "transforms_list.json"), "r") as f:
            transforms_list = json.load(f)
    else:
        fs, path = fsspec.core.url_to_fs(scene_root_dir)
        transforms_list = fs.glob(
            scene_root_dir + f"/*/transforms.json"
        )

    start = len(transforms_list) * opt.rank // opt.world_size
    end = len(transforms_list) * (opt.rank + 1) // opt.world_size
    transforms_list = transforms_list[start:end]

    max_workers = 6
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(transforms_list), desc="Est. ml-depth-pro depth ...") as pbar:

            def est_depth(transforms_path):
                instance_dir = os.path.dirname(transforms_path)
                try:
                    if not os.path.exists(os.path.join(instance_dir, f'depth_{model_type}/finish')):
                        with open(transforms_path, 'r') as f:
                            js = json.load(f)
                        frame_list = []
                        for frame in js['frames']:
                            current_frame = {
                                'file_name': frame['file_path'],
                                'file_path': os.path.join(instance_dir, frame['file_path']),
                            }
                            frame_list.append(current_frame)
                    else:
                        frame_list = None
                except Exception as e:
                    frame_list = None
                    print(f"Error loading frames for {instance_dir}: {e}", flush=True)

                try:
                    if frame_list is not None:
                        save_dir = os.path.join(instance_dir, f'depth_{model_type}')
                        os.makedirs(save_dir, exist_ok=True)
                        for frame in frame_list:
                            raw_img = cv2.imread(frame['file_path'])
                            depth = model.infer_image(raw_img)

                            npz_est_depth_dict = {
                                'lstsq_scale': -1,
                                'depth': depth.astype(np.float32),
                            }
                            np.savez_compressed(os.path.join(save_dir, frame['file_name'].replace('.png', '.npz')), **npz_est_depth_dict)
                        with open(os.path.join(instance_dir, f'depth_{model_type}/finish'), 'w') as f:
                            pass
                except Exception as e:
                    print(f"Error est depth for frames from {instance_dir}: {e}", flush=True)
                
                pbar.update()

            executor.map(est_depth, transforms_list)
            executor.shutdown(wait=True)
    except Exception as e:
        print (f'encountered exception: {e}')
