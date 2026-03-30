# SPDX-FileCopyrightText: 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import cv2
import json
import torch
import imageio
from threeDFixer.moge.model.v2 import MoGeModel # Let's try MoGe-2
from PIL import Image
import numpy as np
import fsspec
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse

def infer_image(model, input_image, device):
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Infer 
    output = model.infer(input_image)
    return output

def batch_infer_image(model, input_image, device):
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    # Infer 
    output = model.infer(input_image)
    return output

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
                        help='Directory to load the MoGe v2.')
    parser.add_argument('--scene_root_dir', type=str, required=True,
                        help='Directory to load the scene data.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    device = torch.device("cuda")
    # Load the model from huggingface hub (or load from local).
    model = MoGeModel.from_pretrained(opt.dpt_model_dir).to(device)
    model_type = "MoGeV2"

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

    max_workers = 16
    load_queue = Queue(maxsize=32)
    try:
        with ThreadPoolExecutor(max_workers=max_workers * 2) as loader_executor, \
            ThreadPoolExecutor(max_workers=max_workers) as saver_executor:

            def loader(transforms_path):
                instance_dir = os.path.dirname(transforms_path)
                try:
                    if not os.path.exists(os.path.join(instance_dir, f'depth_{model_type}/finish')):
                        with open(transforms_path, 'r') as f:
                            js = json.load(f)
                        frame_list = []
                        for frame in js['frames']:
                            current_frame = {
                                'file_name': frame['file_path'],
                                'rgb': cv2.cvtColor(cv2.imread(os.path.join(instance_dir, frame['file_path'])), cv2.COLOR_BGR2RGB)
                            }
                            if os.path.exists(os.path.join(instance_dir, frame['file_path'].replace('.png', '_depth.png'))):
                                current_frame['gt_depth'] = get_gt_depth(os.path.join(instance_dir, frame['file_path'].replace('.png', '_depth.png')), frame)
                            if os.path.exists(os.path.join(instance_dir, frame['file_path'].replace('.png', '_index.png'))):
                                current_frame['mask'] = get_scene_mask(os.path.join(instance_dir, frame['file_path'].replace('.png', '_index.png')))
                            frame_list.append(current_frame)
                        load_queue.put((instance_dir, frame_list))
                    else:
                        load_queue.put((instance_dir, None))
                except Exception as e:
                    load_queue.put((instance_dir, None))
                    print(f"Error loading frames for {instance_dir}: {e}")

            loader_executor.map(loader, transforms_list)

            def est_depth(instance_dir, frame_list):
                try:
                    if frame_list is not None:
                        save_dir = os.path.join(instance_dir, f'depth_{model_type}')
                        os.makedirs(save_dir, exist_ok=True)

                        for batch_idx in range(0, len(frame_list), int(opt.batch_size)):
                            batch_frames = frame_list[batch_idx: batch_idx + int(opt.batch_size)]

                            batch_outputs = batch_infer_image(model, np.stack([frame['rgb'] for frame in batch_frames], axis=0), device)

                            for frame_idx, frame in enumerate(batch_frames):
                                moge_depths = batch_outputs['depth'][frame_idx]
                                moge_mask = batch_outputs['mask'][frame_idx]
                                moge_mask = torch.logical_and(moge_mask, ~torch.isnan(moge_depths))
                                moge_mask = torch.logical_and(moge_mask, ~torch.isinf(moge_depths))
                                if 'mask' in frame:
                                    moge_mask = torch.logical_and(moge_mask, torch.from_numpy(frame['mask']).to(device=device) > 0.0)

                                if 'gt_depth' in frame:
                                    valid_coords = torch.nonzero(moge_mask)
                                    if valid_coords.shape[0] > 0:
                                        gt_depth = torch.from_numpy(frame['gt_depth']).to(device=device, dtype=torch.float32)
                                        gt_depth  = gt_depth[valid_coords[:, 0], valid_coords[:, 1]]
                                        est_depth = moge_depths[valid_coords[:, 0], valid_coords[:, 1]]
                                        X = torch.linalg.lstsq(est_depth[None, :, None], gt_depth[None, :, None]).solution
                                        lstsq_scale = X.item()
                                    else:
                                        lstsq_scale = -1.0
                                else:
                                    lstsq_scale = -1.0
                                
                                npz_est_depth_dict = {
                                    'depth': moge_depths.float().detach().cpu().numpy(),
                                    'mask': moge_mask.float().detach().cpu().numpy() > 0.0,
                                    'lstsq_scale': lstsq_scale
                                }
                                np.savez_compressed(os.path.join(save_dir, frame['file_name'].replace('.png', '.npz')), **npz_est_depth_dict)

                        with open(os.path.join(save_dir, 'finish'), 'w') as f:
                            pass
                except Exception as e:
                    print(f"Error est depth for frames from {instance_dir}: {e}")

            for _ in tqdm(range(len(transforms_list)), desc="Estimating depth maps ..."):
                instance_dir, frame_list = load_queue.get()
                est_depth(instance_dir, frame_list)
                saver_executor.submit(est_depth, instance_dir, frame_list)
                
            saver_executor.shutdown(wait=True)
    except Exception as e:
        print (f'encountered exception: {e}')
