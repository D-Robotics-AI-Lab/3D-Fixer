# SPDX-FileCopyrightText: 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import time
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import subprocess
import numpy as np
import fsspec
import hashlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
TEMP_GLB_DIRS = f"{BLENDER_INSTALLATION_PATH}/place_to_hold_tmp_assets"
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.3-linux-x64.tar.xz -C {os.environ['WORKING_PATH']}")

def _render_scene(all_metadata: dict[str, pd.DataFrame], wall_mats_list, floor_mats_list, hdr_list, num_views,
                  num_min_objects, num_max_objects, output_dir, root_dir):
    chosen_walls = np.random.choice(wall_mats_list, 4)
    chosen_floor = np.random.choice(floor_mats_list, 2)
    chosen_hdr = np.random.choice(hdr_list, 2)

    expected_num = np.random.randint(num_min_objects, num_max_objects+1)
    chosen_objs = []
    composing_order = ['3D-FUTURE', 'ABO', 'HSSD']
    np.random.shuffle(composing_order)
    for source in composing_order:
        if source in all_metadata.keys() and len(all_metadata[source]) >= 1 and np.random.rand() < 0.5:
            sample = all_metadata[source].sample(n=1)
            sample['from'] = source
            chosen_objs.append(sample)

    left_num = expected_num - len(chosen_objs)
    while left_num > 0:
        left_usable_assets = 0
        for k, v in all_metadata.items():
            if k in composing_order:
                continue
            left_usable_assets += len(v)
        if left_usable_assets <= 0:
            break
        for k, v in all_metadata.items():
            if k in composing_order:
                continue
            if len(v) <= 0:
                continue
            sample_n = np.random.randint(0, left_num+1)
            sample = v.sample(n=sample_n)
            sample['from'] = k
            chosen_objs.append(sample)
            left_num -= sample_n
            if left_num <= 0:
                break

    chosen_metadata = pd.concat(chosen_objs).to_records()
    tmp_zip_dirs = []
    tmp_assets_dirs = []

    temp_glb_dirs = os.path.join(TEMP_GLB_DIRS, f"{time.time_ns()}")
    os.makedirs(temp_glb_dirs, exist_ok=True)

    chosed_obj_dir_dict = {}
    for rec in chosen_metadata:
        try:
            if rec['from'] in ["3D-FUTURE", "ABO", "HSSD", "ObjaverseXL_sketchfab"]:
                src_asset_path = os.path.join(root_dir, rec['from'], rec['local_path'])
                obj_name = os.path.basename(src_asset_path)
                tgt_asset_path = os.path.join(temp_glb_dirs, f"{time.time_ns()}_" + obj_name)
                shutil.copy(
                    src_asset_path, tgt_asset_path
                )
                tmp_assets_dirs.append(tgt_asset_path)
                chosed_obj_dir_dict[rec['sha256']] = tgt_asset_path
            elif rec['from'] == "ObjaverseXL_github":
                path_parts = rec['local_path'].split('/')
                file_name = os.path.join(*path_parts[5:])
                zip_file = os.path.join(root_dir, rec['from'], *path_parts[:5])
                tmp_dir = os.path.join(temp_glb_dirs, rec['sha256'] + f"_zip_{time.time_ns()}")
                os.makedirs(tmp_dir, exist_ok=True)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                asset_path = os.path.join(tmp_dir, file_name)
                chosed_obj_dir_dict[rec['sha256']] = asset_path
                tmp_zip_dirs.append(tmp_dir)
        except Exception as e:
            print (f'exceptions during preparing assets: {e}\n', flush=True)
        try:
            all_metadata[rec['from']].drop(index=str(rec['sha256']), inplace=True)
        except Exception as e:
            print (f'exceptions during dropping assets: {e}\n', flush=True)

    hash_object = hashlib.sha256()
    filename_str = ','.join([os.path.basename(fpath) for fpath in (list(chosen_walls) + list(chosen_floor) + list(chosen_hdr) + [v for k, v in chosed_obj_dir_dict.items()])])
    hash_object.update(
        bytes(filename_str, encoding='utf8')
    )
    hash_digest = hash_object.hexdigest()

    save_dir = os.path.join(output_dir, str(hash_digest))

    # convert .blend to .glb for convenience
    chosed_obj_dir_dict_wo_blend = {}
    for k, obj_path in chosed_obj_dir_dict.items():
        if obj_path.endswith('.blend'):
            obj_name = os.path.basename(obj_path)
            tmp_glb_path = os.path.join(temp_glb_dirs, f"{time.time_ns()}_" + obj_name.replace('.blend', '.glb'))
            args = [
                BLENDER_PATH, obj_path, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'convert_blend_to_glb.py'),
                '--',
                '--object', os.path.expanduser(obj_path),
                '--output_path', tmp_glb_path
            ]
            call(args, stdout=DEVNULL, stderr=DEVNULL)
            
            chosed_obj_dir_dict_wo_blend[k] = tmp_glb_path
            tmp_assets_dirs.append(tmp_glb_path)
        else:
            chosed_obj_dir_dict_wo_blend[k] = obj_path
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'create_new_scene.py'),
        '--',
        '--num_views', f'{num_views}',
        '--object', json.dumps(chosed_obj_dir_dict_wo_blend),
        '--wall', json.dumps(list(chosen_walls)),
        '--floor', json.dumps(list(chosen_floor)),
        '--hdrs', json.dumps(list(chosen_hdr)),
        '--resolution', '1024',
        '--output_folder', save_dir,
        '--engine', 'CYCLES',
        '--save_mesh',
        '--save_depth',
        '--save_index',
        '--save_normal',
    ]

    call(args, stdout=DEVNULL, stderr=DEVNULL)

    for asset in tmp_assets_dirs:
        print (f'after rendering, cleaning tmp_assets_dirs {asset}', flush=True)
        os.remove(asset)
    for asset in tmp_zip_dirs:
        print (f'after rendering, cleaning tmp_zip_dirs {asset}', flush=True)
        shutil.rmtree(asset, ignore_errors=True)

    shutil.rmtree(temp_glb_dirs, ignore_errors=True)
    
    out_line = f'after rendering scene {str(hash_digest)}, following assets are left:\n'
    for k, v in all_metadata.items():
        out_line += f'{k}: {len(v)}\n'
    print (out_line, flush=True)



def get_metadata(asset_metadata_dir):
    try:
        metadata = pd.read_csv(os.path.join(asset_metadata_dir, 'my_metadata/metadata_rendered.csv'))
    except FileNotFoundError:
        metadata = pd.read_csv(os.path.join(asset_metadata_dir, 'metadata.csv'))
    if 'uid' in metadata.columns:
        metadata.drop('uid', axis=1, inplace=True)
    metadata = metadata[~metadata['local_path'].isna()]
    return metadata

def get_material_files(material_dir, data_format='blend'):
    fs, path = fsspec.core.url_to_fs(material_dir)
    material_paths = fs.glob(
        material_dir + f"/*/*.{data_format}"
    )
    return material_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Directory to assets')
    parser.add_argument('--obj_asset_dir', type=str, required=True,
                        help='Directory to assets')
    parser.add_argument('--hdr_asset_dir', type=str, required=True,
                        help='Directory to assets')
    parser.add_argument('--floor_mat_dir', type=str, required=True,
                        help='Directory to assets')
    parser.add_argument('--wall_mat_dir', type=str, required=True,
                        help='Directory to assets')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    parser.add_argument('--num_max_objects', type=int, default=20,
                        help='Number of views to render')
    parser.add_argument('--num_min_objects', type=int, default=5,
                        help='Number of views to render')
    parser.add_argument('--num_scenes', type=int, default=10000,
                        help='Number of scenes to construct')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=4)
    opt = parser.parse_args(sys.argv[1:])
    opt = edict(vars(opt))
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    os.makedirs(opt.output_dir, exist_ok=True)

    # get asset list
    asset_sources = opt.obj_asset_dir.split(',')
    all_asset_metadata = {}
    for asset_dir in asset_sources:
        assert opt.root_dir in asset_dir
        dataset_name = os.path.basename(asset_dir.strip('/'))
        asset_metadata = get_metadata(asset_dir)
        all_asset_metadata[dataset_name] = asset_metadata

    # get wall material list
    wall_material_files = get_material_files(opt.wall_mat_dir, data_format='blend')
    # get floor material list
    floor_material_files = get_material_files(opt.floor_mat_dir, data_format='blend')
    # get hdr
    hdr_files = get_material_files(opt.hdr_asset_dir, data_format='exr')

    # all assets
    lines = '3D assets: \n'
    total_3d_asset_num = 0
    for k, v in all_asset_metadata.items():
        lines += f'- {k}: {len(v)}\n'
        total_3d_asset_num += len(v)
    lines += f'- total 3D assets: {total_3d_asset_num}\n'
    lines += f'- wall mats: {len(wall_material_files)}\n'
    lines += f'- floor mats: {len(floor_material_files)}\n'
    lines += f'- HDRs: {len(hdr_files)}\n'
    print (lines)

    start = opt.num_scenes * opt.rank // opt.world_size
    end = opt.num_scenes * (opt.rank + 1) // opt.world_size
    num_scenes = end - start

    # process objects
    func = partial(_render_scene, root_dir=opt.root_dir, output_dir=opt.output_dir,
                   num_max_objects=opt.num_max_objects, num_min_objects=opt.num_min_objects,
                   num_views=opt.num_views, hdr_list=hdr_files, floor_mats_list=floor_material_files,
                   wall_mats_list=wall_material_files, all_metadata=all_asset_metadata)
    
    try:
        with ThreadPoolExecutor(max_workers=opt.max_workers) as executor, \
            tqdm(total=num_scenes, desc="Composing and rendering scene images ...") as pbar:
            def worker(scene_num):
                try:
                    print (f"\nmakeing scene num: {scene_num} ...\n", flush=True)
                    func()
                    pbar.update()
                except Exception as e:
                    pbar.update()
                    print(f"Error processing scenes {scene_num} : {e}")

            executor.map(worker, list(range(num_scenes)))
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
