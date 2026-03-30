# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics.

import os
import json
import copy
import sys
import time
import shutil
import imageio
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import subprocess
import numpy as np
from utils import sphere_hammersley_sequence

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
os.environ['WORKING_PATH'] = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, latents_name, num_views, check_view_stride):
    output_folder_root = os.path.join(output_dir, 'renders_with_rotated_slat', sha256)

    with open(os.path.join(output_dir, 'latents_rot', latents_name, sha256, 'transforms.json'), 'r') as f:
        js = json.load(f)

    for trans in js:
        
        output_folder = os.path.join(output_folder_root, trans['file_path'].replace(".npz", "").replace("./", ""))
        os.makedirs(output_folder, exist_ok=True)

        transforms = {
            'angles': trans['rotate'],
            'vertices_trans': trans['translation'],
            'vertices_scale': trans['scale'],
        }

        if not os.path.exists(os.path.join(output_folder, 'transforms.json')):
        
            # Build camera {yaw, pitch, radius, fov}
            yaws = []
            pitchs = []
            offset = (np.random.rand(), np.random.rand())
            for i in range(num_views):
                y, p = sphere_hammersley_sequence(i, num_views, offset)
                yaws.append(y)
                pitchs.append(p)
            radius = [2] * num_views
            fov = [40 / 180 * np.pi] * num_views
            views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
            
            args = [
                BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_with_rots.py'),
                '--',
                '--views', json.dumps(views),
                '--transforms', json.dumps(transforms),
                '--object', os.path.expanduser(file_path),
                '--resolution', '512',
                '--output_folder', output_folder,
                '--engine', 'CYCLES'
            ]
            if file_path.endswith('.blend'):
                args.insert(1, file_path)
            
            call(args, stdout=DEVNULL, stderr=DEVNULL)

        if os.path.exists(os.path.join(output_folder, 'transforms.json')):
            # check alpha mask PSNR to filter transparent objects
            tmp_dir = os.path.join(os.environ['WORKING_PATH'], 'place_to_hold_tmp_rends', str(time.time_ns()))
            os.makedirs(tmp_dir, exist_ok=True)

            with open(os.path.join(output_folder, 'transforms.json'), 'r') as f:
                js = json.load(f)
            check_frames_info = list(sorted(js['frames'], key=lambda x: x['file_path']))[::check_view_stride]

            mesh_path = os.path.join(output_dir, "renders", sha256, "mesh.ply")
            args = [
                BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'random_SLAT_mask_check.py'),
                '--',
                '--views', json.dumps(check_frames_info),
                '--transforms', json.dumps(transforms),
                '--object', os.path.expanduser(mesh_path),
                '--resolution', '512',
                '--output_folder', tmp_dir,
                '--engine', 'CYCLES'
            ]
            
            call(args, stdout=DEVNULL, stderr=DEVNULL)

            render_images = list(filter(lambda x: x.endswith('.png'), os.listdir(tmp_dir)))
            if len(render_images) > 0:
                psnr_list = {}
                alpha_val = 0.0
                for img_name in render_images:
                    src = imageio.v3.imread(os.path.join(tmp_dir, img_name)).astype(np.float32)[..., -1:] / 255.
                    tgt = imageio.v3.imread(os.path.join(output_folder, img_name)).astype(np.float32)[..., -1:] / 255.
                    src = np.nan_to_num(src)
                    tgt = np.nan_to_num(tgt)
                    mse_val = np.mean((src - tgt) ** 2)
                    if mse_val == 0:
                        psnr_val = 100.0
                    else:
                        max_pixel = 1.0
                        psnr_val = 10 * np.log10((max_pixel ** 2) / mse_val)
                    psnr_list[img_name] = psnr_val
                    alpha_val += psnr_val
                psnr_list['alpha_psnr_mean'] = alpha_val / len(render_images)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                with open(os.path.join(output_folder, 'check_psnrs.json'), 'w') as f:
                    json.dump(psnr_list, f)


if __name__ == '__main__':
    module_name = sys.argv[1]
    module_name = 'ObjaverseXL' if module_name in ['ObjaverseXL_sketchfab', 'ObjaverseXL_github'] else module_name
    dataset_utils = importlib.import_module(f'datasets.{module_name}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--latents_name', type=str, default='dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16',
                        help='SLATs latent name')
    parser.add_argument('--num_views', type=int, default=12,
                        help='Number of views to render')
    parser.add_argument('--check_view_stride', type=int, default=3,
                        help='View stride to check mask PSNR')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=4)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders_with_rotated_slat'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'my_metadata/metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        metadata = metadata[~metadata['local_path'].str.contains('.fbx')]
        metadata = metadata[~metadata['local_path'].str.contains('.FBX')]
        if 'num_rotations' in metadata.columns:
            metadata = metadata[metadata['num_rotations'] > 1]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'renders_with_rotated_slat' in metadata.columns:
            if "ObjaverseXL_github" in opt.output_dir:
                metadata = metadata[metadata['renders_with_rotated_slat'] == False]
            else:
                metadata = metadata[metadata['renders_with_rotated_slat'].isna()]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, latents_name=opt.latents_name, check_view_stride=opt.check_view_stride)
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects with random rotation')

