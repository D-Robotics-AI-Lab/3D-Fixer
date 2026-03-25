# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import json
import cv2
import torch
from PIL import Image
import imageio
import numpy as np
import open3d as o3d
from einops import rearrange

def voxelize_mesh(points, faces, clip_range_first=False, return_mask=True, resolution=64):
    if clip_range_first:
        points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    if isinstance(faces, o3d.cuda.pybind.utility.Vector3iVector):
        mesh.triangles = faces
    else:
        mesh.triangles = o3d.cuda.pybind.utility.Vector3iVector(faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    coords = ((torch.tensor(vertices) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    if return_mask:
        ss_mask = rearrange(ss, 'c (x n1) (y n2) (z n3) -> (n1 n2 n3 c) x y z', n1=4, n2=4, n3=4).float()
        return ss , ss_mask
    else:
        return ss
    
def transform_vertices(vertices, ops, params):
    for op, param in zip(ops, params):
        if op == 'scale':
            vertices = vertices * param
        elif op == 'translation':
            vertices = vertices + param
        else:
            raise NotImplementedError
    return vertices

def normalize_vertices(vertices, scale_factor=1.0):
    min_pos, max_pos = np.min(vertices, axis=0), np.max(vertices, axis=0)
    trans_pos = (min_pos + max_pos)[None] / 2.0
    scale_pos = np.max(max_pos - min_pos) * scale_factor # 1: [-0.5, 0.5], 2.0: [-0.25, 0.25]
    
    vertices = transform_vertices(vertices, ops=['translation', 'scale'],
                                        params=[-trans_pos, 1.0 / (scale_pos + 1e-6)])
    return vertices, trans_pos, scale_pos

def renormalize_vertices(vertices, val_range=0.5, scale_factor=1.25):
    min_pos, max_pos = np.min(vertices, axis=0), np.max(vertices, axis=0)
    if (min_pos < -val_range).any() or (max_pos > val_range).any():
        trans_pos = (min_pos + max_pos)[None] / 2.0
        scale_pos = np.max(max_pos - min_pos) * scale_factor # 1: [-0.5, 0.5], 2.0: [-0.25, 0.25]
        vertices = transform_vertices(vertices, ops=['translation', 'scale'],
                                            params=[-trans_pos, 1.0 / (scale_pos + 1e-6)])
    return vertices

def rot_vertices(vertices, rot_angles, axis_list=['z']):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    for ang, axis in zip(rot_angles, axis_list):
        if axis == 'x':
            R = pcd.get_rotation_matrix_from_xyz((ang, 0, 0))
            pcd.rotate(R, center=(0., 0., 0.))
            del R
        elif axis == 'y':
            R = pcd.get_rotation_matrix_from_xyz((0, ang, 0))
            pcd.rotate(R, center=(0., 0., 0.))
            del R
        elif axis == 'z':
            R = pcd.get_rotation_matrix_from_xyz((0, 0, ang))
            pcd.rotate(R, center=(0., 0., 0.))
            del R
        else:
            raise NotImplementedError
    rot_vertices = np.array(pcd.points)
    del pcd
    return rot_vertices

def _rotmat_x(a: torch.Tensor) -> torch.Tensor:
    # a: scalar tensor
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.stack([
        torch.stack([torch.ones_like(a), torch.zeros_like(a), torch.zeros_like(a)]),
        torch.stack([torch.zeros_like(a), ca, -sa]),
        torch.stack([torch.zeros_like(a), sa, ca]),
    ])
    return R  # [3,3]

def _rotmat_y(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.stack([
        torch.stack([ca, torch.zeros_like(a), sa]),
        torch.stack([torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)]),
        torch.stack([-sa, torch.zeros_like(a), ca]),
    ])
    return R

def _rotmat_z(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.stack([
        torch.stack([ca, -sa, torch.zeros_like(a)]),
        torch.stack([sa, ca, torch.zeros_like(a)]),
        torch.stack([torch.zeros_like(a), torch.zeros_like(a), torch.ones_like(a)]),
    ])
    return R

def rot_vertices_torch(vertices, rot_angles, axis_list=('z',), center=(0.0, 0.0, 0.0)):
    """
    vertices: (N,3) numpy or torch
    rot_angles: iterable of angles (radians), length matches axis_list
    axis_list: iterable like ['x','y','z'] (applied in order)
    center: rotation center, default origin (0,0,0), same as your Open3D code

    return: torch.Tensor (N,3)
    """
    v = torch.as_tensor(vertices)
    device, dtype = v.device, v.dtype

    c = torch.tensor(center, device=device, dtype=dtype).view(1, 3)
    v = v - c  # translate to center

    # Compose rotations in the same order as your for-loop:
    # Open3D effectively does v <- v @ R^T (for row-vector points).
    for ang, axis in zip(rot_angles, axis_list):
        a = torch.as_tensor(ang, device=device, dtype=dtype)
        if axis == 'x':
            R = _rotmat_x(a)
        elif axis == 'y':
            R = _rotmat_y(a)
        elif axis == 'z':
            R = _rotmat_z(a)
        else:
            raise NotImplementedError(f"Unknown axis {axis}")

        v = v @ R.T  # match Open3D row-vector convention

    v = v + c
    return v

def get_instance_mask(instance_mask_path):
    index_mask = imageio.v3.imread(instance_mask_path)
    index_mask = np.rint(index_mask.astype(np.float32) / 65535 * 100.0) # hand coded, max obj nums = 100
    instance_list = np.unique(index_mask).astype(np.uint8)
    return index_mask, instance_list

def get_gt_depth(gt_depth_path, metadata):
    gt_depth = imageio.v3.imread(gt_depth_path).astype(np.float32) / 65535.
    depth_min, depth_max = metadata['depth']['min'], metadata['depth']['max']
    gt_depth = gt_depth * (depth_max - depth_min) + depth_min
    return torch.from_numpy(gt_depth).to(dtype=torch.float32)

def get_est_depth(est_depth_path):
    npz = np.load(est_depth_path)
    est_depth = npz['depth']
    est_depth_mask = npz['mask']
    est_depth = torch.from_numpy(est_depth).to(dtype=torch.float32)
    ivalid_mask = torch.logical_or(torch.isnan(est_depth), torch.isinf(est_depth))
    est_depth_mask = np.logical_and(est_depth_mask, ~ivalid_mask.detach().cpu().numpy())
    est_depth = torch.where(ivalid_mask, 0.0, est_depth)
    return est_depth, est_depth_mask

def get_mix_est_depth(est_depth_path, image_size):
    if 'MoGe' in est_depth_path:
        npz = np.load(est_depth_path)
        est_depth = npz['depth']
        est_depth_mask = npz['mask']
        est_depth = torch.from_numpy(est_depth).to(dtype=torch.float32)
        ivalid_mask = torch.logical_or(torch.isnan(est_depth), torch.isinf(est_depth))
        est_depth_mask = np.logical_and(est_depth_mask, ~ivalid_mask.detach().cpu().numpy())
        est_depth = torch.where(ivalid_mask, 0.0, est_depth)
        return est_depth, est_depth_mask
    elif 'DAv2_' in est_depth_path or 'ml-depth-pro' in est_depth_path:
        npz = np.load(est_depth_path)
        est_depth = npz['depth']
        est_depth_mask = np.logical_not(np.logical_or(
            np.isnan(est_depth),
            np.isinf(est_depth),
        ))
        est_depth = torch.from_numpy(est_depth).to(dtype=torch.float32)
        ivalid_mask = torch.logical_or(torch.isnan(est_depth), torch.isinf(est_depth))
        est_depth_mask = np.logical_and(est_depth_mask, ~ivalid_mask.detach().cpu().numpy())
        est_depth = torch.where(ivalid_mask, 0.0, est_depth)
        return est_depth, est_depth_mask
    elif 'VGGT_1B' in est_depth_path:
        npz = np.load(est_depth_path)
        est_depth = npz['depth']
        est_depth_mask = npz['depth_conf'] > 2.0
        valid_depth_mask = np.logical_not(np.logical_or(
            np.isnan(est_depth),
            np.isinf(est_depth),
        ))
        est_depth_mask = np.logical_and(
            est_depth_mask,
            valid_depth_mask
        )
        est_depth = np.where(valid_depth_mask, est_depth, 0.0)

        depth_min, depth_max = np.min(est_depth), np.max(est_depth)
        est_depth = (est_depth - depth_min) / (depth_max - depth_min + 1e-6)
        est_depth = Image.fromarray(est_depth)
        est_depth = est_depth.resize((image_size, image_size), Image.Resampling.NEAREST)
        est_depth = torch.tensor(np.array(est_depth)).to(dtype=torch.float32)
        est_depth = est_depth * (depth_max - depth_min) + depth_min

        est_depth_mask = Image.fromarray(est_depth_mask.astype(np.float32))
        est_depth_mask = est_depth_mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        est_depth_mask = np.array(est_depth_mask) > 0.5

        ivalid_mask = torch.logical_or(torch.isnan(est_depth), torch.isinf(est_depth))
        est_depth_mask = np.logical_and(est_depth_mask, ~ivalid_mask.detach().cpu().numpy())
        est_depth = torch.where(ivalid_mask, 0.0, est_depth)
        return est_depth, est_depth_mask

def lstsq_align_depth(est_depth, gt_depth, mask):
    valid_coords = torch.nonzero(mask)
    if valid_coords.shape[0] > 0:
        valid_gt_depth  = gt_depth[valid_coords[:, 0], valid_coords[:, 1]]
        valid_est_depth = est_depth[valid_coords[:, 0], valid_coords[:, 1]]
        X = torch.linalg.lstsq(valid_est_depth[None, :, None], valid_gt_depth[None, :, None]).solution
        lstsq_scale = X.item()
    else:
        lstsq_scale = 1.0
    return est_depth * lstsq_scale

def get_cam_poses(frame_info, H, W):
    camera_angle_x = float(frame_info['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    K = torch.from_numpy(K).float()
    c2w = torch.from_numpy(np.array(frame_info['transform_matrix'])).float()
    return K, c2w

def edge_mask_morph_gradient(mask, kernel, iterations=1):
    """
    mask: HxW, bool/uint8
    ksize: 3/5/7... 越大边缘越厚
    return: edge_mask uint8 {0,1}
    """
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)

    dil = cv2.dilate(m, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=0.0)
    ero = cv2.erode(m, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=0.0)

    edge = (dil - ero)  # 0/1/2
    edge = (edge > 0).astype(np.uint8)
    return edge

def process_scene_image(image: Image.Image, instance_mask: np.ndarray, image_size: int,
                        resize_perturb: bool = False, resize_perturb_ratio: float = 0.0):
    image_rgba = image
    try:
        alpha = np.array(image_rgba.getchannel("A")) > 0
    except ValueError:
        alpha = np.ones_like(np.array(image_rgba.getchannel(0))) > 0
    alpha = np.logical_and(alpha, instance_mask).astype(np.uint8) * 255

    image_resized = image_rgba.resize((image_size, image_size), Image.Resampling.LANCZOS).convert("RGB")
    alpha_resized = Image.fromarray(alpha, mode="L").resize((image_size, image_size), Image.Resampling.NEAREST)

    if resize_perturb and np.random.rand() < resize_perturb_ratio:
        rand_reso = np.random.randint(32, image_size)

        image_resized = image_resized.resize((rand_reso, rand_reso), Image.Resampling.LANCZOS)
        image_resized = image_resized.resize((image_size, image_size), Image.Resampling.LANCZOS)

        alpha_resized = alpha_resized.resize((rand_reso, rand_reso), Image.Resampling.NEAREST)
        alpha_resized = alpha_resized.resize((image_size, image_size), Image.Resampling.NEAREST)

    img_np = np.array(image_resized, dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    a_np = np.array(alpha_resized, dtype=np.uint8) 
    a_t = torch.from_numpy(a_np).unsqueeze(0).float() / 255.0
    img4 = torch.cat([img_t, a_t], dim=0)  # (4,S,S)
    return img_t, img4

def get_rays(i, j, K, c2w):
    i = i.float() + 0.5
    j = j.float() + 0.5
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_fast(u: torch.Tensor, v: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor):
    """
    u, v: 1D tensor (pixel coords), dtype long/int64 or int32
    K: (3,3) or (4,4) but used as 3x3; on same device as output
    c2w: (4,4) or (3,4), uses [:3,:3] and [:3,3]
    return:
      rays_o: (N,3)
      rays_d: (N,3)
    """
    # 确保 float 并加 0.5 取像素中心
    u = u.to(dtype=torch.float32) + 0.5
    v = v.to(dtype=torch.float32) + 0.5

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # dirs in camera frame (N,3)
    dirs = torch.stack([(u - cx) / fx,
                        -(v - cy) / fy,
                        -torch.ones_like(u)], dim=-1)

    # 旋转到世界坐标：dirs @ R^T (更常见/更快)
    R = c2w[:3, :3]            # (3,3)
    rays_d = dirs @ R.T        # (N,3)

    # 原点：相机中心 (3,) 扩展到 (N,3)
    t = c2w[:3, 3]
    rays_o = t.expand_as(rays_d)
    return rays_o, rays_d

def process_instance_image(image: Image.Image, instance_mask: np.ndarray, color_mask: np.ndarray, depth_map: torch.Tensor, 
                           K: torch.Tensor, c2w: torch.Tensor, image_size: int):
    image_rgba = image
    try:
        alpha = np.asarray(image_rgba.getchannel("A")) > 0
    except ValueError:
        alpha = np.ones_like(np.array(image_rgba.getchannel(0))) > 0
    alpha = np.logical_and(alpha, instance_mask).astype(np.uint8) * 255
    valid_mask = np.array(alpha).nonzero()

    bbox = [valid_mask[1].min(), valid_mask[0].min(), valid_mask[1].max(), valid_mask[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]

    i, j = torch.from_numpy(valid_mask[1]), torch.from_numpy(valid_mask[0])
    rays_o, rays_d = get_rays(i, j, K, c2w)
    rays_color = color_mask[valid_mask[0], valid_mask[1]].astype(np.float32)
    rays_t = depth_map[valid_mask[0], valid_mask[1]]

    image_resized = image_rgba.crop(aug_bbox).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha_resized = Image.fromarray(alpha, mode="L").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)

    img_np = np.asarray(image_resized, dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    a_np = np.asarray(alpha_resized, dtype=np.uint8) 
    a_t = torch.from_numpy(a_np).unsqueeze(0).float() / 255.0
    return img_t, a_t, rays_o, rays_d, rays_color, rays_t

def get_crop_area_rays(image: Image.Image, instance_mask: np.ndarray, K: torch.Tensor, c2w: torch.Tensor, image_size):

    alpha = np.asarray(image.getchannel("A")) > 0
    if instance_mask is not None:
        alpha = np.logical_and(alpha, instance_mask).astype(np.float32) # * 255
    else:
        alpha = alpha.astype(np.float32)
    valid_mask = np.array(alpha).nonzero()

    bbox = [valid_mask[1].min(), valid_mask[0].min(), valid_mask[1].max(), valid_mask[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]

    i, j = torch.meshgrid(
        torch.linspace(aug_bbox[0], aug_bbox[2]-1, steps=image_size), 
        torch.linspace(aug_bbox[1], aug_bbox[3]-1, steps=image_size)
    )
    rays_o, rays_d = get_rays(i, j, K, c2w)
    return rays_o, rays_d

def process_instance_image_crop(image: Image.Image, instance_mask: np.ndarray, color_mask: np.ndarray, 
                                depth_map: torch.Tensor, 
                                gt_depth_map: torch.Tensor, 
                           K: torch.Tensor, c2w: torch.Tensor, image_size: int,
                           edge_mask_morph_gradient_fn):
    image_rgba = image
    alpha = np.asarray(image_rgba.getchannel("A")) > 0
    alpha = np.logical_and(alpha, instance_mask).astype(np.float32) # * 255
    valid_mask = np.array(alpha).nonzero()

    bbox = [valid_mask[1].min(), valid_mask[0].min(), valid_mask[1].max(), valid_mask[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]

    i, j = torch.meshgrid(
        torch.linspace(aug_bbox[0], aug_bbox[2]-1, steps=image_size), 
        torch.linspace(aug_bbox[1], aug_bbox[3]-1, steps=image_size)
    )
    rays_o, rays_d = get_rays(i, j, K, c2w)

    image_resized = image_rgba.crop(aug_bbox).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha_resized = Image.fromarray(alpha, mode="F").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)
    depth_map_resized = Image.fromarray(depth_map.detach().cpu().numpy(), mode="F").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)
    gt_depth_map_resized = Image.fromarray(gt_depth_map.detach().cpu().numpy(), mode="F").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)
    color_mask_resized = Image.fromarray(color_mask.astype(np.float32), mode="F").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)

    img_np = np.asarray(image_resized, dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    a_np = np.asarray(alpha_resized, dtype=np.float32).astype(dtype=np.uint8)

    edge_mask = edge_mask_morph_gradient_fn((a_np > 0).astype(np.uint8))
    fg_mask = (a_np > edge_mask).astype(np.uint8)
    rays_color = fg_mask.astype(np.float32) + edge_mask.astype(np.float32) * 0.5

    valid_mask = fg_mask.nonzero()
    rays_t = torch.from_numpy(np.asarray(depth_map_resized).astype(np.float32))

    a_t = torch.from_numpy(a_np).unsqueeze(0).float() # / 255.0
    return img_t, a_t, fg_mask, rays_o, rays_d, rays_color, rays_t, valid_mask, depth_map_resized, gt_depth_map_resized, color_mask_resized

def process_instance_image_only(image: Image.Image, instance_mask: np.ndarray, image_size: int):
    image_rgba = image
    alpha = np.asarray(image_rgba.getchannel("A")) > 0
    alpha = np.logical_and(alpha, instance_mask).astype(np.uint8) * 255
    valid_mask = np.array(alpha).nonzero()

    bbox = [valid_mask[1].min(), valid_mask[0].min(), valid_mask[1].max(), valid_mask[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]

    image_resized = image_rgba.crop(aug_bbox).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha_resized = Image.fromarray(alpha, mode="L").crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)

    img_np = np.asarray(image_resized, dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    a_np = np.asarray(alpha_resized, dtype=np.uint8) 
    a_t = torch.from_numpy(a_np).unsqueeze(0).float() / 255.0
    return img_t, a_t

def crop_depth_image(depth_image, aug_bbox, image_size):
    d = depth_image.cpu()
    d_np = d.numpy().astype(np.float32)
    img = Image.fromarray(d_np, mode="F")
    img = img.crop(aug_bbox)
    img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    out = torch.from_numpy(np.asarray(img, dtype=np.float32))
    return out
    
def proj_depth2pcd(mask, depth, image, rays_o, rays_d):
    mask = torch.nonzero(mask)
    
    ### 
    mask = [mask[:, 0].detach().cpu().numpy(), mask[:, 1].detach().cpu().numpy()]
    pixel_depth = depth[mask[0], mask[1]]
    pixel_color = image.detach().permute(1, 2, 0)[mask[0], mask[1]]

    pixel_points = rays_o[mask[0], mask[1]] + rays_d[mask[0], mask[1]] * pixel_depth[:, None] # pt
    return pixel_points.detach().cpu().numpy(), pixel_color.detach().cpu().numpy()

def vox2pts(ss, resolution = 64):
    coords = torch.nonzero(ss[0] > 0, as_tuple=False)
    position = (coords.float() + 0.5) / resolution - 0.5
    position = position.detach().cpu().numpy()
    return position

def voxelize_pcd(points, points_color=None, clip_range_first=False, return_mask=True, resolution=64):
    if clip_range_first:
        points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=1/resolution, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < resolution), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / resolution - 0.5
    coords = ((torch.tensor(vertices) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    if points_color is not None:
        points_t = torch.from_numpy(points).to(torch.float32)
        colors_t = torch.from_numpy(points_color).to(torch.float32)

        coords = torch.floor((points_t + 0.5) * resolution).to(torch.long)
        coords = torch.clamp(coords, 0, resolution - 1)
        ix, iy, iz = coords[:, 0], coords[:, 1], coords[:, 2]
        lin = ix * (resolution * resolution) + iy * resolution + iz # linear index in [0, R^3)

        sum_color = torch.zeros((resolution * resolution * resolution), dtype=torch.float32)
        sum_color.index_add_(0, lin, colors_t)
        count = torch.zeros((resolution * resolution * resolution,), dtype=torch.long)
        ones = torch.ones_like(lin, dtype=torch.long)
        count.index_add_(0, lin, ones)

        count_f = count.to(torch.float32)
        mean_color = sum_color / torch.clamp(count_f, min=1.0)  # empty -> divide by 1 (still 0)
        color_mean = mean_color.view(resolution, resolution, resolution, 1).permute(3, 0, 1, 2).contiguous()
    if return_mask:
        ss_mask = rearrange(ss if points_color is None else color_mean, 'c (x n1) (y n2) (z n3) -> (n1 n2 n3 c) x y z', n1=4, n2=4, n3=4).float()
        return ss , ss_mask
    else:
        return ss
    
def voxelize_pcd_pt(points, points_color=None, clip_range_first=False, return_mask=True, resolution=64):
    points = torch.nan_to_num(points)
    points_color = torch.nan_to_num(points_color) if isinstance(points_color, torch.Tensor) else points_color
    device = points.device
    if clip_range_first:
        points = torch.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=1/resolution, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < resolution), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / resolution - 0.5
    coords = ((torch.tensor(vertices, device=device) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long, device=device)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    if points_color is not None:
        points_t = points.to(torch.float32)
        colors_t = points_color.to(torch.float32)

        coords = torch.floor((points_t + 0.5) * resolution).to(torch.long)
        coords = torch.clamp(coords, 0, resolution - 1)
        ix, iy, iz = coords[:, 0], coords[:, 1], coords[:, 2]
        lin = ix * (resolution * resolution) + iy * resolution + iz # linear index in [0, R^3)

        sum_color = torch.zeros((resolution * resolution * resolution), dtype=torch.float32, device=device)
        sum_color.index_add_(0, lin, colors_t)
        count = torch.zeros((resolution * resolution * resolution,), dtype=torch.long, device=device)
        ones = torch.ones_like(lin, dtype=torch.long)
        count.index_add_(0, lin, ones)

        count_f = count.to(torch.float32)
        mean_color = sum_color / torch.clamp(count_f, min=1.0)  # empty -> divide by 1 (still 0)
        color_mean = mean_color.view(resolution, resolution, resolution, 1).permute(3, 0, 1, 2).contiguous()
    if return_mask:
        ss_mask = rearrange(ss if points_color is None else color_mean, 'c (x n1) (y n2) (z n3) -> (n1 n2 n3 c) x y z', n1=4, n2=4, n3=4).float()
        return ss , ss_mask
    else:
        return ss
        
def get_std_cond(root, instance, crop_size, return_mask=False):
    image_root = os.path.join(root, 'renders_cond', instance)
    if os.path.exists(os.path.join(image_root, 'transforms.json')):
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

    image = image.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    alpha = image.getchannel(3)
    image = image.convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    alpha = torch.tensor(np.array(alpha)).float() / 255.0
    image = image * alpha.unsqueeze(0)
    if return_mask:
        return image, alpha.unsqueeze(0)
    else:
        return image

def map_rotated_slat2canonical_pose(vertices, rot_slat_info):
    vertices_scale = rot_slat_info['scale']
    vertices_trans = np.array(rot_slat_info['translation'])
    rand_rot = rot_slat_info['rotate']
    pcd = o3d.geometry.PointCloud()
    vertices = vertices * vertices_scale
    vertices = vertices + vertices_trans
    pcd.points = o3d.utility.Vector3dVector(vertices)
    R1 = pcd.get_rotation_matrix_from_xyz((-rand_rot[0], 0, 0))
    R2 = pcd.get_rotation_matrix_from_xyz((0, -rand_rot[1], 0))
    R3 = pcd.get_rotation_matrix_from_xyz((0, 0, -rand_rot[2]))
    pcd.rotate(R3, center=(0., 0., 0.))
    pcd.rotate(R2, center=(0., 0., 0.))
    pcd.rotate(R1, center=(0., 0., 0.))
    vertices = np.asarray(pcd.points)

    return vertices

def project2ply(mask, depth, image, K, c2w):
    mask = torch.nonzero(mask)

    rays_o, rays_d = get_rays(mask[:, 1], mask[:, 0], K, c2w)
    
    ### 
    mask = [mask[:, 0].detach().cpu().numpy(), mask[:, 1].detach().cpu().numpy()]
    pixel_depth = depth[mask[0], mask[1]]
    pixel_color = image.detach().permute(1, 2, 0).cpu().numpy()[mask[0], mask[1]]

    pixel_points = rays_o + rays_d * pixel_depth[:, None]
    pixel_points = pixel_points.detach().cpu().numpy()
    return pixel_points, pixel_color