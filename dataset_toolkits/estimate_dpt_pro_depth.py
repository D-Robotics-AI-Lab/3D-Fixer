# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import json
import torch
import imageio

import depth_pro

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
    parser.add_argument('--scene_root_dir', type=str, required=True,
                        help='Directory to load the scene data.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    device = torch.device("cuda")
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    model_type = "ml-depth-pro"

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

    max_workers = 7
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
                            
                            image, _, f_px = depth_pro.load_rgb(frame['file_path'])
                            image = transform(image).to(device)

                            prediction = model.infer(image, f_px=f_px)

                            npz_est_depth_dict = {
                                'lstsq_scale': -1,
                                'depth': prediction['depth'].detach().cpu().numpy().astype(np.float32),
                                'focallength_px': prediction['focallength_px'].item()
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
