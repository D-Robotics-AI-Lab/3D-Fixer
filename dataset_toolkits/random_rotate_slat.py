# SPDX-FileCopyrightText: 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import copy
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image
import open3d as o3d
import copy
import threeDFixer.models as models
import threeDFixer.modules.sparse as sp

torch.set_grad_enabled(False)

def get_data(frames, sha256):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            try:
                image_path = os.path.join(opt.output_dir, 'renders', sha256, view['file_path'])
                image = Image.open(image_path)
                image = image.resize((518, 518), Image.Resampling.LANCZOS)
                image = np.array(image).astype(np.float32) / 255
                image = image[:, :, :3] * image[:, :, 3:]
                image = torch.from_numpy(image).permute(2, 0, 1).float()

                c2w = torch.tensor(view['transform_matrix'])
                c2w[:3, 1:3] *= -1
                extrinsics = torch.inverse(c2w)
                fov = view['camera_angle_x']
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

                return {
                    'image': image,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics
                }
            except Exception as e:
                print(f"Error loading image for {sha256}: {e}", flush=True)
                return None
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--slat_encoder', type=str, default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='SLAT encoder model')
    parser.add_argument('--rand_rot_times', type=int, default=16,
                        help='Maximal rotation numbers for each instance.')
    parser.add_argument('--max_num_voxels', type=int, default=32768)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    slat_encoder = models.from_pretrained(opt.slat_encoder).eval().cuda()
    slat_encoder_name = os.path.basename(opt.slat_encoder)

    feature_name = f'{opt.model}_{slat_encoder_name}'

    # get file list
    try:
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'my_metadata/metadata.csv'))
    except Exception:
        if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
            metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
        else:
            raise ValueError('metadata.csv not found')
        
    if opt.filter_low_aesthetic_score is not None:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
    metadata = metadata[metadata['rendered'] == True]
    metadata = metadata[metadata['num_voxels'] <= opt.max_num_voxels]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'latents_rot', feature_name, f'{sha256}/transforms.json')):
            try:
                with open(os.path.join(opt.output_dir, 'latents_rot', feature_name, f'{sha256}/transforms.json'), 'r') as f:
                    js = json.load(f)
                sha256s.remove(sha256)
            except Exception:
                continue

    # extract features
    load_queue = Queue(maxsize=16)
    try:
        with ThreadPoolExecutor(max_workers=16) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(sha256):
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_data(frames, sha256):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)
                    mesh = o3d.io.read_triangle_mesh(os.path.join(opt.output_dir, 'renders', sha256, 'mesh.ply'))
                    load_queue.put((sha256, data, mesh))
                except Exception as e:
                    load_queue.put((None, None, None))
                    print(f"Error loading data for {sha256}: {e}", flush=True)

            loader_executor.map(loader, sha256s)

            def saver(sha256, save_pack_list):
                try:
                    instance_path = os.path.join(opt.output_dir, 'latents_rot', feature_name, f'{sha256}')
                    os.makedirs(instance_path, exist_ok=True)
                    for save_pack in save_pack_list:
                        pack = save_pack.pop('pack')
                        np.savez_compressed(os.path.join(instance_path, save_pack['file_path']), **pack)
                    with open(os.path.join(instance_path, "transforms.json"), 'w') as f:
                        json.dump(save_pack_list, f)

                except Exception as e:
                    print(f"Error saving data for {sha256}: {e}", flush=True)

            def random_rotate_mesh(input_mesh):
                rand_rot = (np.random.rand(3) * 2.0 - 1.0) * np.pi
                mesh = copy.deepcopy(input_mesh)
                
                vertices = np.clip(np.nan_to_num(np.asarray(mesh.vertices)), -0.5 + 1e-6, 0.5 - 1e-6)
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                
                R1 = mesh.get_rotation_matrix_from_xyz((rand_rot[0], 0, 0))
                R2 = mesh.get_rotation_matrix_from_xyz((0, rand_rot[1], 0))
                R3 = mesh.get_rotation_matrix_from_xyz((0, 0, rand_rot[2]))
                mesh.rotate(R1, center=(0., 0., 0.))
                mesh.rotate(R2, center=(0., 0., 0.))
                mesh.rotate(R3, center=(0., 0., 0.))

                vertices = np.asarray(mesh.vertices)
                min_vertices, max_vertices = np.min(vertices, axis=0), np.max(vertices, axis=0)
                vertices_trans = (min_vertices + max_vertices) / 2.0
                vertices_scale = max(max_vertices - min_vertices) + 1e-6
                vertices = vertices - vertices_trans
                vertices = vertices / vertices_scale
                vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
                vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
                assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
                vertices = (vertices + 0.5) / 64 - 0.5

                positions = torch.from_numpy(vertices).float().cuda()
                indices = ((positions + 0.5) * 64).long()
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"

                # map coords back to canonical pose to integrate features
                pcd = o3d.geometry.PointCloud()
                vertices = vertices * vertices_scale
                vertices = vertices + vertices_trans
                pcd.points = o3d.utility.Vector3dVector(vertices)
                R1 = mesh.get_rotation_matrix_from_xyz((-rand_rot[0], 0, 0))
                R2 = mesh.get_rotation_matrix_from_xyz((0, -rand_rot[1], 0))
                R3 = mesh.get_rotation_matrix_from_xyz((0, 0, -rand_rot[2]))
                pcd.rotate(R3, center=(0., 0., 0.))
                pcd.rotate(R2, center=(0., 0., 0.))
                pcd.rotate(R1, center=(0., 0., 0.))
                vertices = np.asarray(pcd.points)
                # map coords back to canonical pose to integrate features

                positions = torch.from_numpy(vertices).float().cuda()
                return indices, positions, rand_rot, vertices_trans, vertices_scale

            for _ in tqdm(range(len(sha256s)), desc="Randomly rotating mesh & Extracting features & Encoding SLAT ..."):
                sha256, data, mesh = load_queue.get()
                if sha256 is None:
                    continue

                save_pack = []
                patchtokens_lst = []
                try:
                    n_views = len(data)
                    for i in range(0, n_views, opt.batch_size):
                        batch_data = data[i:i+opt.batch_size]
                        bs = len(batch_data)
                        batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                        features = dinov2_model(batch_images, is_training=True)
                        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                        patchtokens_lst.append(patchtokens)
                    patchtokens_lst = torch.cat(patchtokens_lst, dim=0)
                except Exception as e:
                    print(f"Error extracting features for {sha256}: {e}", flush=True)
                    continue

                for rot_num in range(opt.rand_rot_times):
                    try:
                        indices, positions, rand_rot, vertices_trans, vertices_scale = random_rotate_mesh(mesh)
                    except Exception:
                        continue
                    try:
                        n_views = len(data)
                        N = positions.shape[0]

                        pack = {
                            'indices': indices.cpu().numpy().astype(np.uint8),
                        }

                        uv_lst = []
                        for i in range(0, n_views, opt.batch_size):
                            batch_data = data[i:i+opt.batch_size]
                            bs = len(batch_data)
                            batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                            batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                            uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                            uv_lst.append(uv)
                        patchtokens = patchtokens_lst 
                        uv = torch.cat(uv_lst, dim=0)

                        patchtokens = F.grid_sample(
                            patchtokens, # [B, C, H, W]
                            uv.unsqueeze(1), # [B, 1, L, 2]
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(2).permute(0, 2, 1) 
                        patchtokens = torch.mean(patchtokens.float(), dim=0)

                        feats = sp.SparseTensor(
                            feats = patchtokens, 
                            coords = torch.cat([
                                torch.zeros(patchtokens.shape[0], 1).int().cuda(),
                                indices.int().cuda(),
                            ], dim=1),
                        ).cuda()

                        latent = slat_encoder(feats, sample_posterior=False)
                        if not torch.isfinite(latent.feats).all():
                            print ("Finite latent encountered.", flush=True)
                            continue
                        pack = {
                            'feats': latent.feats.cpu().numpy().astype(np.float32),
                            'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                        }
                        pack['patchtokens'] = patchtokens.detach().cpu().numpy().astype(np.float16)

                        save_pack.append({
                            'file_path': f'./{rot_num:03d}.npz',
                            'rotate': [rand_rot[0], rand_rot[1], rand_rot[2]],
                            'translation': [vertices_trans[0], vertices_trans[1], vertices_trans[2]],
                            'scale': vertices_scale,
                            'pack': pack
                        })
                    except Exception as e:
                        print(f"Error processing slat for {sha256}: {e}", flush=True)
                        continue

                # save features
                if len(save_pack) > 0:
                    print (f'{len(save_pack)} rotated SLATs will be saved.', flush=True)
                    saver_executor.submit(saver, sha256, save_pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")