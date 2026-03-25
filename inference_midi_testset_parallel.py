# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['ATTN_BACKEND'] = 'xformers'
import cv2
import json
import torch
import random
torch.backends.cuda.matmul.allow_tf32 = True
from threeDFixer.moge.model.v2 import MoGeModel
from threeDFixer.utils import render_utils, postprocessing_utils
from threeDFixer.pipelines import ThreeDFixerPipeline
from threeDFixer.datasets.utils import (
    transform_vertices,
    edge_mask_morph_gradient,
    process_scene_image,
    process_instance_image,
    project2ply
)
import imageio
from PIL import Image
import numpy as np
from einops import repeat
import utils3d
import trimesh
from glob import glob
import argparse
from easydict import EasyDict as edict
import time
from sklearn.neighbors import KNeighborsRegressor

CROP_SIZE = 518

def get_est_depth(est_depth_path, device):
    npz = np.load(est_depth_path)
    est_depth = npz['depth']
    intrinsics = npz['intrinsics']
    if 'mask' in npz:
        est_depth_mask = npz['mask']
    else:
        est_depth_mask = np.ones_like(est_depth) > 0.0
    if 'lstsq_scale' in npz:
        est_depth *= npz['lstsq_scale']
    est_depth = torch.from_numpy(est_depth).to(dtype=torch.float32, device=device)
    ivalid_mask = torch.logical_or(torch.isnan(est_depth), torch.isinf(est_depth))
    est_depth_mask = np.logical_and(est_depth_mask, ~ivalid_mask.detach().cpu().numpy())
    est_depth = torch.where(ivalid_mask, 0.0, est_depth)
    return est_depth, est_depth_mask, intrinsics

def align_depth(relative_depth, metric_depth, mask=None):
    regressor = KNeighborsRegressor()
    if mask is not None:
        regressor.fit(relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1))
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(relative_depth.shape)
    return depth

def save_image(img, save_path):
    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    imageio.v3.imwrite(save_path, img)

def get_midi_gt_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = depth[..., 0].astype(np.float32)
    mask = np.ones_like(depth)
    mask[depth > 1000.0] = 0.0  # depth = 65535 is the invalid value
    depth[~(mask > 0.5)] = 0.0
    return depth, mask

R1 = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
R2 = np.array([[0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

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

save_projected_colored_pcd = lambda pts, pts_color, fpath: trimesh.PointCloud(pts.reshape(-1, 3), pts_color.reshape(-1, 3)).export(fpath)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the generated results.')
    parser.add_argument('--testset_dir', type=str, required=True,
                        help='Directory to load the testset scenes.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory to load the model ckpts.')
    parser.add_argument('--valid_ratio_threshold', type=float, default=0.005,
                        help='Minimum value of valid pixel ratio.')
    parser.add_argument('--chunk_size', type=int, default=2,
                        help='Number of generated 3D assets in parallel.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    set_random_seed(opt.seed)

    ############## model information
    pipeline = ThreeDFixerPipeline.from_pretrained(
        opt.model_dir
    )
    pipeline.cuda()
    moge_v2 = pipeline.models['scene_cond_model']
    ############## model information

    midi_test_src = opt.testset_dir
    src_root = f'{midi_test_src}/3D-FRONT-TEST-RENDER'
    metadata_list = sorted(glob(os.path.join(src_root, '*/*/meta.json')))
    metadata_list = metadata_list[int(opt.rank)::int(opt.world_size)]
    src_surf_root = f'{midi_test_src}/3D-FRONT-TEST-SCENE'

    save_root = opt.output_dir
    os.makedirs(save_root, exist_ok=True)

    chunk_size = opt.chunk_size

    for meta_path in metadata_list:
        
        scene_name_1, scene_name_2 = meta_path.split('/')[-3:-1]

        rgb_prefix = 'render'
        sample_frame = 0
        scene_root_dir = os.path.join(src_root, scene_name_1, scene_name_2)
        save_dir  = os.path.join(save_root, f'{scene_name_1}/{scene_name_2}/{rgb_prefix}_{sample_frame:04d}')
        os.makedirs(save_dir, exist_ok=True)

        save_mask_root = os.path.join(save_dir, 'masks')
        os.makedirs(save_mask_root, exist_ok=True)
        save_instance_image_root = os.path.join(save_dir, 'instances')
        os.makedirs(save_instance_image_root, exist_ok=True)
        save_pcd_root = os.path.join(save_dir, 'points')
        os.makedirs(save_pcd_root, exist_ok=True)
        save_mesh_root = os.path.join(save_dir, 'objects')
        os.makedirs(save_mesh_root, exist_ok=True)
        save_gt_pcd_root = os.path.join(save_dir, 'gt_pcd')
        os.makedirs(save_gt_pcd_root, exist_ok=True)

        with open(os.path.join(scene_root_dir, 'meta.json'), 'r') as f:
            metadata = json.load(f)

        gt_depth, gt_depth_mask = get_midi_gt_depth(os.path.join(scene_root_dir, f'depth_{sample_frame:04d}.exr'))
        H, W = gt_depth.shape
        transform_matrix = list(filter(lambda x: x['index'] == f'{sample_frame:04d}', metadata['locations']))[0]['transform_matrix']
        pos_desc = list(filter(lambda x: x['index'] == f'{sample_frame:04d}', metadata['locations']))[0]['position']
        c2w = np.array(transform_matrix)
        focal_length = 0.5 * W / np.tan(0.5 * metadata['camera_angle_x'])
        K = np.array([
            [focal_length, 0.0, 0.5 * W],
            [0.0, focal_length, 0.5 * H],
            [0.0, 0.0, 1.0],
        ])
        c2w = R2 @ c2w

        K = torch.from_numpy(K)
        c2w = torch.from_numpy(c2w)

        R_T = R1 @ np.linalg.inv(R2)
        save_projected_colored_pcd = lambda pts, pts_color, fpath: trimesh.PointCloud(pts.reshape(-1, 3), pts_color.reshape(-1, 3)).export(fpath)
        
        ### get GT masks
        mask_pack, foreground_mask = get_midi_mask(os.path.join(scene_root_dir, f'semantic_{sample_frame:04d}.png'))
        if not foreground_mask.any():
            continue
        ### get GT masks

        input_image = imageio.v3.imread(os.path.join(scene_root_dir, f'{rgb_prefix}_{sample_frame:04d}.webp'))
        if input_image.shape[-1] == 4:
            input_image = np.where(repeat(input_image[..., -1:], '... c -> ... (n c)', n=3) > 0, input_image[..., :3], 255)
        imageio.v3.imwrite(os.path.join(save_dir, 'input_image.png'), input_image)

        # infer depth
        dpt_input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        if dpt_input_image.shape[0] == 4:
            dpt_input_image = dpt_input_image[:3]
        est_t = time.time()
        output = moge_v2.infer(dpt_input_image)
        total_t = time.time() - est_t
        depth = output['depth']
        intrinsics = output['intrinsics']
        np.savez_compressed(f"{save_dir}/depth.npz", depth=depth.detach().cpu().numpy(), intrinsics=intrinsics.detach().cpu().numpy())
        del output
        del intrinsics
        del depth
        torch.cuda.empty_cache()

        # load info for 3d gen
        image = Image.open(os.path.join(save_dir, 'input_image.png')).convert('RGB')
        H, W = image.size
        est_depth, est_depth_mask, intrinsics = get_est_depth(f"{save_dir}/depth.npz", device)

        # align pred depth with GT for metrics-computing
        gt_depth = torch.from_numpy(gt_depth).to(est_depth)
        depth_mask = np.logical_and(est_depth_mask, gt_depth_mask)
        est_depth = align_depth(est_depth.detach().cpu().numpy(), 
                            gt_depth.detach().cpu().numpy(), 
                            np.logical_and(depth_mask, foreground_mask))
        est_depth = torch.from_numpy(est_depth).to(gt_depth)

        imageio.v3.imwrite(os.path.join(save_mask_root, f'foreground_mask.png'), foreground_mask)
        imageio.v3.imwrite(os.path.join(save_mask_root, f'depth_mask.png'), depth_mask)

        erode_kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))

        image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.).to(est_depth).permute(2, 0, 1)
        if not os.path.exists(os.path.join(save_pcd_root, 'scene.ply')):
            scene_est_depth_pts, scene_est_depth_pts_colors = \
                    project2ply(torch.from_numpy(depth_mask).to(device=est_depth.device), 
                    est_depth, 
                    image_pt,
                    K.to(est_depth), 
                    c2w.to(est_depth),)
            save_projected_colored_pcd(scene_est_depth_pts, scene_est_depth_pts_colors,
                                        f"{save_pcd_root}/scene.ply")
        if not os.path.exists(os.path.join(save_pcd_root, 'gt_depth_scene.ply')):
            scene_gt_depth_pts, scene_gt_depth_pts_colors = \
                    project2ply(torch.from_numpy(depth_mask).to(device=est_depth.device), 
                    gt_depth, 
                    image_pt,
                    K.to(est_depth), 
                    c2w.to(est_depth),)
            save_projected_colored_pcd(scene_gt_depth_pts, scene_gt_depth_pts_colors,
                                        f"{save_pcd_root}/gt_depth_scene.ply")
            
        # camera information
        extrinsics = c2w.float()
        extrinsics[:3, 1:3] *= -1
        fov = float(metadata['camera_angle_x'])
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        # camera information

        instance_image_masked_pack = []
        scene_image_masked_pack = []
        pcd_points_pack = []
        pcd_colors_pack = []
        instance_name_pack = []

        for object_id, object_mask in enumerate(mask_pack):
            if object_mask is None:
                continue
            instance_name = f'{object_id+1}'
            try:

                image_pt = torch.from_numpy(np.array(image).astype(np.float32) / 255.).to(est_depth).permute(2, 0, 1)

                object_mask = object_mask > 0.0
                instance_mask = np.logical_and(object_mask, depth_mask).astype(np.uint8)
                valid_ratio = np.sum((instance_mask > 0).astype(np.float32)) / (H * W)
                print (f'valid ratio of {instance_name}: {valid_ratio:.4f}')
                if valid_ratio < opt.valid_ratio_threshold:
                    continue
                
                edge_mask = edge_mask_morph_gradient(instance_mask, kernel, 2)
                fg_mask = (instance_mask > edge_mask).astype(np.uint8)
                color_mask = fg_mask.astype(np.float32) + edge_mask.astype(np.float32) * 0.5

                image = Image.open(os.path.join(save_dir, 'input_image.png'))
                scene_image, scene_image_masked = process_scene_image(image, instance_mask, CROP_SIZE)
                instance_image, instance_mask, instance_rays_o, instance_rays_d, instance_rays_c, \
                    instance_rays_t = process_instance_image(image, instance_mask, color_mask, est_depth, K, c2w, CROP_SIZE)
                
                save_image(scene_image, os.path.join(save_instance_image_root, f'input_scene_image_{instance_name}.png'))
                save_image(scene_image_masked, os.path.join(save_instance_image_root, f'input_scene_image_masked_{instance_name}.png'))
                save_image(instance_image, os.path.join(save_instance_image_root, f'input_instance_image_{instance_name}.png'))
                save_image(torch.cat([instance_image, instance_mask]), os.path.join(save_instance_image_root, f'input_instance_image_masked_{instance_name}.png'))

                pcd_points = (instance_rays_o.to(device) + instance_rays_d.to(device) * instance_rays_t[..., None].to(device)).detach().cpu().numpy() # pt2np
                pcd_colors = instance_rays_c

                save_projected_colored_pcd(pcd_points, repeat(pcd_colors, 'n -> n c', c=3),
                                            f"{save_pcd_root}/instance_est_depth_{instance_name}.ply")

                instance_image_masked_pack.append(
                    torch.cat([instance_image, instance_mask]).to(device)
                )
                scene_image_masked_pack.append(
                    scene_image_masked.to(device)
                )
                pcd_points_pack.append(pcd_points)
                pcd_colors_pack.append(pcd_colors)
                instance_name_pack.append(instance_name)
                
            except Exception as e:
                print (instance_name, e)

        try:
            for sub_idx in range(0, len(instance_image_masked_pack), chunk_size):
                
                sub_t = time.time()
                ### generate 3D assets
                outputs, coarse_trans, coarse_scale, fine_trans, fine_scale = pipeline.run_parallel(
                    torch.stack(instance_image_masked_pack[sub_idx:sub_idx + chunk_size]),
                    torch.stack(scene_image_masked_pack[sub_idx:sub_idx + chunk_size]),
                    seed=opt.seed,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    points=pcd_points_pack[sub_idx:sub_idx + chunk_size],
                    points_mask=pcd_colors_pack[sub_idx:sub_idx + chunk_size],
                    slat_sampler_params={
                        "steps": 15,
                        "cfg_interval": [0.8, 1.0]
                    },
                )
                total_t += (time.time() - sub_t)

                for chunk_idx, instance_name in enumerate(instance_name_pack[sub_idx:sub_idx + chunk_size]):
                    video = render_utils.render_video(outputs['gaussian'][chunk_idx])['color']
                    imageio.mimsave(os.path.join(save_instance_image_root, f'instance_gs_fine_{instance_name}.mp4'), video, fps=30)

                    # GLB files can be extracted from the outputs
                    glb = postprocessing_utils.to_glb(
                        outputs['gaussian'][chunk_idx],
                        outputs['mesh'][chunk_idx],
                        # Optional parameters
                        simplify=0.95,          # Ratio of triangles to remove in the simplification process
                        texture_size=1024,      # Size of the texture used for the GLB
                        transform_fn=lambda x: transform_vertices(x, ops=['scale', 'translation', 'scale', 'translation'], 
                                                                params=[fine_scale[chunk_idx], fine_trans[chunk_idx][None], 
                                                                        coarse_scale[chunk_idx], coarse_trans[chunk_idx][None]])
                    )
                    glb.apply_transform(R_T)
                    glb.export(os.path.join(save_mesh_root, f'{instance_name}.glb'))
                ### generate 3D assets
        except Exception as e:
            print (instance_name, e)

        if total_t > 0.0:
            with open(os.path.join(save_mesh_root, 'time.txt'), 'w') as f:
                json.dump({'time': str(total_t)}, f) 