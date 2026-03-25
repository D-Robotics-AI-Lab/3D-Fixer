# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import json
import torch
import random
from threeDFixer.moge.model.v2 import MoGeModel
from threeDFixer.utils import render_utils, postprocessing_utils
from threeDFixer.pipelines import ThreeDFixerPipeline
from threeDFixer.datasets.utils import (
    transform_vertices,
    get_gt_depth,
    get_cam_poses,
    edge_mask_morph_gradient,
    process_scene_image,
    process_instance_image,
    project2ply
)
import imageio
from PIL import Image
import numpy as np
from einops import repeat
import argparse
from easydict import EasyDict as edict
import fsspec
import utils3d
import trimesh
import shutil
from sklearn.linear_model import RANSACRegressor, LinearRegression

CROP_SIZE = 518

def get_topN_instant_mask(instance_mask_path, n=5):
    assert n > 0
    index_mask = imageio.v3.imread(instance_mask_path)
    h, w = index_mask.shape[:2]
    index_mask = np.rint(index_mask.astype(np.float32) / 65535 * 100.0) # hand coded, max obj nums = 100
    instance_list = np.unique(index_mask).astype(np.uint8)
    instance_list = instance_list[instance_list != 0]
    instance_list = instance_list[instance_list != 1]
    instance_list = instance_list[instance_list != 2]

    mask_pack = []
    mask_ratio = []
    for inst_idx in instance_list:
        mask_pack.append(
            index_mask == inst_idx
        )
        mask_ratio.append(
            mask_pack[-1].astype(np.float32).sum() / (h * w)
        )

    mask_ratio = np.array(mask_ratio)
    mask_ratio_ind = np.argsort(mask_ratio)[-n:]

    selected_mask_packs = []
    selected_mask_idxs = []
    for idx in mask_ratio_ind:
        selected_mask_packs.append(mask_pack[idx])
        selected_mask_idxs.append(instance_list[idx])

    return selected_mask_packs, selected_mask_idxs

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

def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=True), min_samples=min_samples)
    if mask is not None:
        regressor.fit(relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1))
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(relative_depth.shape)
    return depth

def save_image(img, save_path):
    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    imageio.v3.imwrite(save_path, img)

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
    parser.add_argument('--top_instance_mask', type=int, default=5,
                        help='The top N instances according to valid pixel ratio.')
    parser.add_argument('--valid_ratio_threshold', type=float, default=0.005,
                        help='Minimum value of valid pixel ratio.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    set_random_seed(opt.seed)

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)

    testset_dir = opt.testset_dir
    if os.path.exists(os.path.join(testset_dir, 'transforms_list.json')):
        with open(os.path.join(testset_dir, 'transforms_list.json'), 'r') as f:
            transforms_list = json.load(f)
    else:
        fs, path = fsspec.core.url_to_fs(testset_dir)
        transforms_list = sorted(fs.glob(
            testset_dir + f"/*/transforms.json"
        ))
        with open(os.path.join(testset_dir, 'transforms_list.json'), 'w') as f:
            json.dump(transforms_list, f)
    transforms_list = sorted(transforms_list)
    transforms_list = transforms_list[int(opt.rank)::int(opt.world_size)]

    ############## model information
    pipeline = ThreeDFixerPipeline.from_pretrained(
        opt.model_dir
    )
    pipeline.cuda()
    moge_v2 = pipeline.models['scene_cond_model']
    ############## model information

    for transforms_path in transforms_list:

        with open(transforms_path, 'r') as f:
            js = json.load(f)

        selected_frame = js['frames'][0]
        scene_sha256 = transforms_path.split('/')[-2]
        scene_root_dir = os.path.dirname(transforms_path)
        instances_gt = js['instance']

        save_dir = os.path.join(output_dir, scene_sha256)
        os.makedirs(save_dir, exist_ok=True)
        
        save_mask_root = os.path.join(save_dir, 'masks')
        os.makedirs(save_mask_root, exist_ok=True)
        save_instance_image_root = os.path.join(save_dir, 'instances')
        os.makedirs(save_instance_image_root, exist_ok=True)
        save_pcd_root = os.path.join(save_dir, 'points')
        os.makedirs(save_pcd_root, exist_ok=True)
        save_mesh_root = os.path.join(save_dir, 'objects')
        os.makedirs(save_mesh_root, exist_ok=True)

        gt_depth = get_gt_depth(os.path.join(scene_root_dir, selected_frame['file_path'].replace('.png', '_depth.png')), selected_frame)
        gt_depth = gt_depth.detach().cpu().numpy()
        gt_depth_mask = gt_depth > 0.0
        H, W = gt_depth.shape
        K, c2w = get_cam_poses(selected_frame, H, W)
        instance_mask_path = os.path.join(scene_root_dir, selected_frame['file_path'].replace('.png', '_index.png'))
        mask_pack, mask_pack_inst_idx = get_topN_instant_mask(instance_mask_path, n=opt.top_instance_mask)

        shutil.copy(
            os.path.join(scene_root_dir, selected_frame['file_path']),
            os.path.join(save_dir, 'input_image.png')
        )

        # infer depth
        input_image = imageio.v3.imread(os.path.join(scene_root_dir, selected_frame['file_path'])).astype(np.float32)[..., :3]
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        if input_image.shape[0] == 4:
            input_image = input_image[:3] 
        output = moge_v2.infer(input_image)
        depth = output['depth']
        intrinsics = output['intrinsics']
        np.savez_compressed(f"{save_dir}/depth.npz", depth=depth.detach().cpu().numpy(), intrinsics=intrinsics.detach().cpu().numpy())
        del output
        del intrinsics
        del depth
        # infer depth

        # load info for 3d gen
        image = Image.open(os.path.join(save_dir, 'input_image.png')).convert('RGB')
        H, W = image.size
        est_depth, est_depth_mask, intrinsics = get_est_depth(f"{save_dir}/depth.npz", device)

        # align pred depth with GT for metrics-computing
        gt_depth = torch.from_numpy(gt_depth).to(est_depth)
        depth_mask = np.logical_and(est_depth_mask, gt_depth_mask)
        foreground_mask = np.stack(mask_pack).any(axis=0)
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
        extrinsics = torch.from_numpy(np.array(selected_frame['transform_matrix'])).float()
        extrinsics[:3, 1:3] *= -1
        fov = float(selected_frame['camera_angle_x'])
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        # camera information

        for object_id, object_mask in zip(mask_pack_inst_idx, mask_pack):

            try:
                instance_name = f'{object_id}'

                object_mask = object_mask > 0
                instance_mask = np.logical_and(object_mask, depth_mask).astype(np.uint8)
                valid_ratio = np.sum((instance_mask > 0).astype(np.float32)) / (H * W)
                print (f'valid ratio of {instance_name}: {valid_ratio:.4f}')
                if valid_ratio < opt.valid_ratio_threshold:
                    continue
                
                ### process condition information
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
                ### process condition information
                
                ### generate 3D assets
                outputs, coarse_trans, coarse_scale, fine_trans, fine_scale = pipeline.run(
                    torch.cat([instance_image, instance_mask]).to(device),
                    scene_image_masked = scene_image_masked.to(device),
                    seed=opt.seed,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    points=pcd_points,
                    points_mask=pcd_colors,
                )

                video = render_utils.render_video(outputs['gaussian'][0])['color']
                imageio.mimsave(os.path.join(save_instance_image_root, f'instance_gs_fine_{instance_name}.mp4'), video, fps=30)

                # GLB files can be extracted from the outputs
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    # Optional parameters
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=1024,      # Size of the texture used for the GLB
                    transform_fn=lambda x: transform_vertices(x, ops=['scale', 'translation', 'scale', 'translation'], 
                                                            params=[fine_scale, fine_trans[None], coarse_scale, coarse_trans[None]])
                )
                glb.export(os.path.join(save_mesh_root, f'{instance_name}.glb'))
                ### generate 3D assets

            except Exception as e:
                print (instance_name, e)