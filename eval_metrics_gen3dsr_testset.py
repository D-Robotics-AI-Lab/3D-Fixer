import os
import argparse
import numpy as np
import random
import trimesh
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interpn
import json

def interpolate_array(input_array, coordinates):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/util.py#L394
    Interpolate values from the input array at specified coordinates using bilinear interpolation.

    Parameters:
    - input_array: 2D NumPy array (HxW)
    - coordinates: 2D NumPy array (nx2) containing coordinates (row, column)

    Returns:
    - sampled_values: 1D NumPy array containing sampled values
    """
    # Generate grid coordinates
    H, W = input_array.shape[:2]
    grid_x, grid_y = np.arange(W), np.arange(H)
    # Interpolate values at specified coordinates
    sampled_values = interpn((grid_y, grid_x), input_array, coordinates,
                             method='linear', bounds_error=False, fill_value=0)
    return sampled_values

def depth_to_points(depth, K=None, R=None, t=None):
    """
    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # from reference to target viewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[:, :, :, :3, 0][0]

def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

def compute_frustum_planes(K, h, w, near=0.1, far=5):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/util.py#L209
    """
    coord = np.array([
        [0, 0], [w, 0], [0, h], [w, h],
        [0, 0], [w, 0], [0, h], [w, h]
    ])
    coord = np.concatenate((coord, np.ones_like(coord)[..., [0]]), -1)
    coord = coord.astype(np.float32)
    depths = np.ones_like(coord[:, :1])
    depths[:4] = near
    depths[4:] = far
    corners = depths * (np.linalg.inv(K) @ coord.T).T
    planes = np.array([
        [0, 2, 4, 6],
        [1, 5, 3, 7],
        [0, 4, 1, 5],
        [2, 3, 6, 7]
    ])
    normals = np.cross(
        corners[planes[:, 0]] - corners[planes[:, 3]],
        corners[planes[:, 1]] - corners[planes[:, 2]]
    )
    return np.concatenate([corners[planes[:, 0]], normals], axis=-1)

def project_from_camera_to_image(points_3d, K):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L11
    Project 3D points to the image plane using camera intrinsics matrix K.

    Parameters:
    - points_3d: numpy array of shape (n, 3) representing 3D points
    - K: numpy array of shape (3, 3) representing camera intrinsics matrix

    Returns:
    - points_2d: numpy array of shape (n, 2) representing projected points in the image plane
    """
    # Project 3D points to 2D using camera intrinsics matrix K
    points_2d_homogeneous = np.dot(K, points_3d.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return points_2d


def load_all_objects(rec_path, frustum_planes):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L29
    """
    object_paths = rec_path.glob('*.glb')
    scene = trimesh.Trimesh()
    for object_path in object_paths:
        if object_path.stem != 'full_scene':
            mesh = trimesh.load(object_path)
            scene += mesh.geometry[list(mesh.geometry)[0]]

    for point_normal in frustum_planes:
        scene = scene.slice_plane(point_normal[:3], point_normal[3:])
    return scene


def mask_mesh_projection(mesh: trimesh.Trimesh, mask, K):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L42
    """
    vertices = mesh.vertices
    vertices2d = project_from_camera_to_image(vertices, K)
    valid = np.isclose(interpolate_array(mask, vertices2d[:, [1, 0]]), 1)
    new_vertices = vertices[valid]
    new_indices = np.empty(len(vertices))
    new_indices[valid] = np.arange(valid.sum())
    new_faces = new_indices[mesh.faces[np.all(valid[mesh.faces], 1)]]
    return trimesh.Trimesh(new_vertices, new_faces)


def load_bg_meshes(gt_path, rec_path, K):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L53
    """
    gt_depth = np.load(gt_path)
    depth_mask = gt_depth < 100
    points = depth_to_points(gt_depth[None], K)
    triangles = create_triangles(gt_depth.shape[0], gt_depth.shape[1])
    gt_bg_mesh = mask_mesh_projection(trimesh.Trimesh(points.reshape(-1, 3), triangles), depth_mask, K)
    if rec_path.is_file():
        rec_bg_mesh = mask_mesh_projection(trimesh.load(rec_path), depth_mask, K)
    else:
        rec_bg_mesh = trimesh.Trimesh()
    return gt_bg_mesh, rec_bg_mesh


def compute_metrics_meshes(gt, pred, num_points=1000000, thresholds=[0.1, 0.01, 0.001], eps=1e-6):
    """
    Reference: https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py#L66
    """
    metrics = {}
    gt_points = gt.as_open3d.sample_points_uniformly(num_points)
    pred_points = pred.as_open3d.sample_points_uniformly(num_points)

    dist_gt_pred = np.array(gt_points.compute_point_cloud_distance(pred_points))
    dist_pred_gt = np.array(pred_points.compute_point_cloud_distance(gt_points))
    metrics["Chamfer"] = (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2

    for t in thresholds:
        precision = 100.0 * (dist_pred_gt < t).mean()
        recall = 100.0 * (dist_gt_pred < t).mean()
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics[f"Precision@{t}"] = precision
        metrics[f"Recall@{t}"] = recall
        metrics[f"F1@{t}"] = f1
    return metrics

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Path to the location of the Front3D data",
                        default="../imgs/FRONT3D", required=True)
    parser.add_argument("--rec_path", help="Path to the directory with the reconstructed scenes",
                        default="", required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    rec_path = Path(args.rec_path)

    set_random_seed(int(args.seed))

    metrics = defaultdict(list)

    with open(data_root / "scene_ids") as f:
        scene_ids = f.read().split('\n')

    for scene_id in tqdm(scene_ids):
        if os.path.exists(rec_path / f"rec_{scene_id}"):
            with open(data_root / "annotation" / f"annotation_00{scene_id}.json") as f:
                annotation = json.load(f)
            K = np.array(annotation['camera_intrinsics'])
            frustum_planes = compute_frustum_planes(K, 968, 1296)
            
            gt = trimesh.load(data_root/ "sceneobjgt" / f"sceneobjgt_00{scene_id}.ply")
            rec = load_all_objects(rec_path / f"rec_{scene_id}" / "objects", frustum_planes)

            if len(rec.vertices) == 0:
                continue
            current_metrics = compute_metrics_meshes(gt, rec)
            print(f"scene: {scene_id} Chamfer: {current_metrics['Chamfer']} F0.1: {current_metrics['F1@0.1']}")
            for m in current_metrics:
                metrics[m].append(current_metrics[m])

    print("All scenes: ")
    avg_metrics = {}
    for m in metrics.keys():
        avg_metrics[f"{m}_mean"] = np.mean(metrics[m])
    print(avg_metrics)
    metrics.update(avg_metrics)

    with open(rec_path / f'gen3dsr_metrics.json', 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


