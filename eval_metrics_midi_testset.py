# SPDX-FileCopyrightText: 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import json
import torch
from PIL import Image
import numpy as np
import open3d as o3d
import copy
from glob import glob
import random
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm

def get_midi_gt_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = depth[..., 0].astype(np.float32)
    mask = np.ones_like(depth)
    mask[depth > 1000.0] = 0.0  # depth = 65535 is the invalid value
    depth[~(mask > 0.5)] = 0.0
    return depth, mask

def get_midi_mask_valid(mask_path, instance_labels, small_image_proportion=0.005):
    height, width = 512, 512
    mask = np.array(Image.open(mask_path).resize((height, width), Image.NEAREST))
    valid_list = []
    for idx in range(1, len(instance_labels)):
        inst_mask = (mask == idx).astype(np.float32)
        if inst_mask.sum() <= small_image_proportion * height * width:
            valid_list.append(False)
        else:
            valid_list.append(True)
    return valid_list

def get_midi_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    instance_labels = np.unique(mask.reshape(-1), axis=0)
    masks = []
    valid_list = get_midi_mask_valid(mask_path, instance_labels)
    for idx, valid in zip(range(1, len(instance_labels)), valid_list):
        if valid:
            instance_mask = mask == idx
            masks.append(instance_mask)
        else:
            masks.append(None)
    fg_mask = []
    for mask in masks:
        if mask is not None:
            fg_mask.append(mask)
    if len(fg_mask) > 0:
        fg_mask = np.stack(fg_mask).any(axis=0)
    else:
        fg_mask = np.array(Image.open(mask_path)) < 0
    return masks, fg_mask

def voxelize(points, voxel_size, grid_size, min_bound):
    """
    Reference: https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/utils/metrics.py#L94
    Voxelize the point cloud.

    Args:
        points (torch.Tensor): Point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        min_bound (torch.Tensor): Minimum bounds of the grid

    Returns:
        torch.Tensor: Voxel grid of shape (B, grid_size, grid_size, grid_size) with boolean type
    """
    # Shift points to positive grid and scale
    scaled_points = (points - min_bound) / voxel_size
    indices = torch.floor(scaled_points).long()

    # Clamp indices to grid size
    indices = indices.clamp(0, grid_size - 1)

    # Create voxel grid
    voxel_grid = torch.zeros(
        (points.shape[0], grid_size, grid_size, grid_size),
        device=points.device,
        dtype=torch.bool,
    )
    for b in range(points.shape[0]):
        voxel_grid[b, indices[b, :, 0], indices[b, :, 1], indices[b, :, 2]] = True

    return voxel_grid

def compute_volume_iou(pred, gt, voxel_size=0.05, grid_size=64, mode="bbox"):
    """
    Reference: https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/utils/metrics.py#L126
    Compute Volume IoU between predicted and ground truth point clouds.

    Args:
        pred (torch.Tensor): Predicted point cloud of shape (B, N, 3)
        gt (torch.Tensor): Ground truth point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        mode (str): Mode of computing volume iou, either "pcd" or "bbox"

    Returns:
        torch.Tensor: Volume IoU for each batch
    """
    if mode == "pcd":
        # Define the grid bounds
        min_bound = torch.min(torch.min(pred, dim=1).values, dim=0).values
        max_bound = torch.max(torch.max(pred, dim=1).values, dim=0).values
        min_bound = torch.min(min_bound, torch.min(gt, dim=1).values.min(dim=0).values)
        max_bound = torch.max(max_bound, torch.max(gt, dim=1).values.max(dim=0).values)

        # Voxelize the point clouds
        pred_voxels = voxelize(pred, voxel_size, grid_size, min_bound)
        gt_voxels = voxelize(gt, voxel_size, grid_size, min_bound)

        # Compute intersection and union
        intersection = (pred_voxels & gt_voxels).sum(dim=(1, 2, 3)).float()
        union = (pred_voxels | gt_voxels).sum(dim=(1, 2, 3)).float()

        # Compute IoU
        iou = intersection / (union + 1e-8)

    elif mode == "bbox":
        # Compute bounding boxes
        pred_min = pred.min(dim=1).values
        pred_max = pred.max(dim=1).values
        gt_min = gt.min(dim=1).values
        gt_max = gt.max(dim=1).values

        # Compute intersection
        intersection_min = torch.max(pred_min, gt_min)
        intersection_max = torch.min(pred_max, gt_max)
        inter_dims = (intersection_max - intersection_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

        # Compute union
        pred_dims = (pred_max - pred_min).clamp(min=0)
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]  # (B,)
        gt_dims = (gt_max - gt_min).clamp(min=0)
        gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]

        # Compute IoU
        union_vol = pred_vol + gt_vol - inter_vol
        iou = inter_vol / (union_vol + 1e-8)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return iou

def compute_object_level_metrics(pred_vertices, gt_vertices, eps=1e-6):
    pred_vertices = copy.deepcopy(pred_vertices)
    gt_vertices = copy.deepcopy(gt_vertices)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_vertices)
    gt_vertices = pcd

    pred_vertices_pt = torch.from_numpy(np.array(pred_vertices.points).astype(np.float32))
    gt_vertices_pt = torch.from_numpy(np.array(gt_vertices.points).astype(np.float32))

    iou = compute_volume_iou(pred_vertices_pt.unsqueeze(0), gt_vertices_pt.unsqueeze(0)).item()

    def normalize_(tensor):
        min_vals = tensor.min(dim=1, keepdim=True)[0]
        max_vals = tensor.max(dim=1, keepdim=True)[0]

        ranges = max_vals - min_vals
        ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

        normalized_tensor = 1.9 * (tensor - min_vals) / ranges - 0.95

        return normalized_tensor
    
    pred_vertices_pt = normalize_(pred_vertices_pt)
    gt_vertices_pt = normalize_(gt_vertices_pt)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_vertices_pt.detach().cpu().numpy())
    gt_vertices = pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_vertices_pt.detach().cpu().numpy())
    pred_vertices = pcd

    # Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L71
    dist_gt_pred = np.array(gt_vertices.compute_point_cloud_distance(pred_vertices))
    dist_pred_gt = np.array(pred_vertices.compute_point_cloud_distance(gt_vertices))

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    metrics = {
        'IoU': iou,
        'Chamfer': (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return metrics

def compute_scene_level_metrics(all_pred_vertices, all_gt_vertice, eps=1e-6):
    gt_vertices = o3d.geometry.PointCloud()
    gt_vertices.points = o3d.utility.Vector3dVector(
        np.concatenate(
            [verts.astype(np.float32) for verts in all_gt_vertice]
        )
    )
    pred_vertices = o3d.geometry.PointCloud()
    pred_vertices.points = o3d.utility.Vector3dVector(
        np.concatenate(
            [np.array(verts.points).astype(np.float32) for verts in all_pred_vertices]
        )
    )

    # Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L71
    dist_gt_pred = np.array(gt_vertices.compute_point_cloud_distance(pred_vertices))
    dist_pred_gt = np.array(pred_vertices.compute_point_cloud_distance(gt_vertices))

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    precision = 100.0 * (dist_pred_gt < 0.1).mean()
    recall = 100.0 * (dist_gt_pred < 0.1).mean()
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    metrics = {
        'Chamfer': (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return metrics

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the generated results.')
    parser.add_argument('--testset_dir', type=str, required=True,
                        help='Directory to load the testset scenes.')
    parser.add_argument('--seed', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    set_random_seed(opt.seed)

    midi_test_src = opt.testset_dir
    save_root = opt.output_dir
    src_root = f'{midi_test_src}/3D-FRONT-TEST-RENDER'
    metadata_list = sorted(glob(os.path.join(src_root, '*/*/meta.json')))
    src_surf_root = f'{midi_test_src}/3D-FRONT-TEST-SCENE'
    with open(os.path.join(midi_test_src, 'midi_test_room_ids.json'), 'r') as f:
        midi_test_room_list = json.load(f)
    with open(os.path.join(midi_test_src, 'midi_test_furniture_ids.json'), 'r') as f:
        midi_test_furniture_list = json.load(f)

    eval_metrics = {}
    all_mean = {
        k: [] for k in ['IoU-B', 'Chamfer-O', 'F1-O', 'Chamfer-S', 'F1-S', 'T-S']
    }
    for room_id, room_furniture in tqdm(zip(midi_test_room_list, midi_test_furniture_list), total=len(midi_test_room_list)):
        
        scene_name_1, scene_name_2 = room_id.split('/')
        
        rgb_prefix = 'render'
        sample_frame = 0
        scene_root_dir = os.path.join(src_root, scene_name_1, scene_name_2)
        save_dir  = os.path.join(save_root, f'{scene_name_1}/{scene_name_2}/{rgb_prefix}_{sample_frame:04d}')

        save_mask_root = os.path.join(save_dir, 'masks')
        save_instance_image_root = os.path.join(save_dir, 'instances')
        save_pcd_root = os.path.join(save_dir, 'points')
        save_mesh_root = os.path.join(save_dir, 'objects')
        save_gt_pcd_root = os.path.join(save_dir, 'gt_pcd')

        with open(os.path.join(scene_root_dir, 'meta.json'), 'r') as f:
            metadata = json.load(f)

        gt_depth, gt_depth_mask = get_midi_gt_depth(os.path.join(scene_root_dir, f'depth_{sample_frame:04d}.exr'))
        H, W = gt_depth.shape

        ### get GT masks
        mask_pack, foreground_mask = get_midi_mask(os.path.join(scene_root_dir, f'semantic_{sample_frame:04d}.png'))
        if not foreground_mask.any():
            continue
        ### get GT masks

        pred_vertices = []
        gt_vertices = []
        num_points = 20480
        scene_metrics = {}
        for object_id, object_mask in enumerate(mask_pack):
            if object_mask is None:
                continue
            instance_name = f'{object_id+1}'
            if not os.path.exists(os.path.join(save_mesh_root, f'{instance_name}.glb')):
                continue

            try:
                pred_vertices.append(
                    o3d.io.read_triangle_mesh(os.path.join(save_mesh_root, f'{instance_name}.glb')).sample_points_uniformly(num_points)
                )
            except Exception as e:
                print (e, os.path.join(save_mesh_root, f'{instance_name}.glb'))
                continue

            ### gt vertices
            scene_object = room_furniture[object_id]
            gt_surface_path = os.path.join(
                src_surf_root, room_id, f"{scene_object}.npy"
            )
            data = np.load(gt_surface_path, allow_pickle=True).tolist()
            surface = data["surface_points"]  # Nx3
            rng = np.random.default_rng()
            ind = rng.choice(surface.shape[0], num_points, replace=False)
            gt_vertices.append(surface[ind])
            ### gt vertices

            ### pred vertices
            obj_metrics = compute_object_level_metrics(pred_vertices[-1],
                                                       gt_vertices[-1],
                                                      )
            scene_metrics[instance_name] = obj_metrics
            all_mean['Chamfer-O'].append(obj_metrics['Chamfer'])
            all_mean['F1-O'].append(obj_metrics['F1'])
            all_mean['IoU-B'].append(obj_metrics['IoU'])
            ### pred vertices
        if len(pred_vertices) > 0:
            scene_metrics['scene'] = compute_scene_level_metrics(pred_vertices, gt_vertices)
            all_mean['Chamfer-S'].append(scene_metrics['scene']['Chamfer'])
            all_mean['F1-S'].append(scene_metrics['scene']['F1'])
            if os.path.exists(os.path.join(save_mesh_root, 'time.txt')):
                with open(os.path.join(save_mesh_root, 'time.txt'), 'r') as f:
                    total_t = float(json.load(f)['time'])
                if total_t > 5.0 and total_t < 100.0: # skip empty scene and torch.compile time
                    scene_metrics['scene']['T-S'] = total_t
                    all_mean['T-S'].append(total_t)
            eval_metrics[room_id] = scene_metrics
            print (scene_metrics)

    for k in all_mean.keys():
        all_mean[k] = np.mean(all_mean[k])

    with open(f"{save_root}/mean.json", 'w') as f:
        json.dump(all_mean, f)
    with open(f"{save_root}/eval_metrics.json", 'w') as f:
        json.dump(eval_metrics, f)