# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import utils3d
import fsspec
import zipfile
import objaverse.xl as oxl
from objaverse.xl.sketchfab import SketchfabDownloader
import json
from easydict import EasyDict as edict
from utils import get_file_hash
import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(),
                        help='Directory to save the metadata')
    parser.add_argument('--image_feature_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--ss_latent_model', type=str, default='ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--slat_latent_model', type=str, default='slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    def check_mv_worker(rec: np.record, pbar, records):
        try:
            instace_dir = os.path.join(opt.output_dir, f'renders/' + rec['sha256'])
            if os.path.exists(os.path.join(instace_dir, 'transforms.json')):
                with open(os.path.join(instace_dir, 'transforms.json'), 'r') as f:
                    js = json.load(f)
                records.append({
                    'sha256': rec['sha256'],
                    'rendered': True
                })
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking rendered mv images ' + rec['sha256'], flush=True)

    def check_rot_mv_worker(rec: np.record, pbar, records):
        try:
            instace_dir = os.path.join(opt.output_dir, f'renders_with_rotated_slat/' + rec['sha256'])
            rots_idx = []
            if os.path.exists(instace_dir):
                rot_list = os.listdir(instace_dir)
                for rot in rot_list:
                    if os.path.exists(os.path.join(instace_dir, rot, 'transforms.json')) and \
                        os.path.exists(os.path.join(instace_dir, rot, 'check_psnrs.json')):
                        with open(os.path.join(instace_dir, rot, 'check_psnrs.json'), 'r') as f:
                            check_psnrs = json.load(f)
                            check_mean = float(check_psnrs['alpha_psnr_mean'])
                        if check_mean >= 40:
                            rots_idx.append(rot)
            if len(rots_idx) > 0:
                records.append({
                    'sha256': rec['sha256'],
                    'rotated_mv_num': len(rots_idx),
                    'valid_rots': ','.join(rots_idx)
                })
            else:
                records.append({
                    'sha256': rec['sha256'],
                    'rotated_mv_num': 0,
                    'valid_rots': ''
                })
            pbar.update()
        except Exception as e:
            pbar.update()
            records.append({
                'sha256': rec['sha256'],
                'rotated_mv_num': 0,
                'valid_rots': ''
            })
            print (f'Caught exception {e} during checking rotated rendered mv images ' + rec['sha256'], flush=True)

    def check_saved_ply_worker(rec: np.record, pbar, records):
        size_limit_mb = 50
        try:
            instace_dir = os.path.join(opt.output_dir, f'renders/' + rec['sha256'])
            if os.path.exists(os.path.join(instace_dir, 'mesh.ply')):
                mesh_path = os.path.join(instace_dir, 'mesh.ply')
                size_mb = os.path.getsize(mesh_path) / (1024 ** 2)
                if size_mb < size_limit_mb:
                    try:
                        mesh = o3d.io.read_triangle_mesh(mesh_path)
                        if mesh.is_empty():
                            raise ValueError("empty mesh")
                        records.append({
                            'sha256': rec['sha256'],
                            'mesh_size': size_mb,
                            'readable': True
                        })
                        # pbar.update()
                    except Exception as e:
                        # pbar.update()
                        records.append({
                            'sha256': rec['sha256'],
                            'mesh_size': size_mb,
                            'readable': False
                        })
                        print (f'Caught exception {e} during checking saved ply mesh ' + rec['sha256'], flush=True)
                else:
                    records.append({
                        'sha256': rec['sha256'],
                        'mesh_size': size_mb,
                        'readable': False
                    })
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking saved ply mesh ' + rec['sha256'], flush=True)

    def check_mv_normal_worker(rec: np.record, pbar, records):
        try:
            instace_dir = os.path.join(opt.output_dir, f'renders/' + rec['sha256'], 'normal_finish')
            if os.path.exists(instace_dir):
                records.append({
                    'sha256': rec['sha256'],
                    'normal_rendered': True
                })
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking rendered mv normals ' + rec['sha256'], flush=True)

    def check_render_cond_worker(rec: np.record, pbar, records):
        try:
            instace_dir = os.path.join(opt.output_dir, f'renders_cond/' + rec['sha256'])
            if os.path.exists(os.path.join(instace_dir, 'transforms.json')):
                with open(os.path.join(instace_dir, 'transforms.json'), 'r') as f:
                    js = json.load(f)
                records.append({
                    'sha256': rec['sha256'],
                    'cond_rendered': True
                })
            pbar.update()
        except Exception as e:
            records.append({
                'sha256': rec['sha256'],
                'cond_rendered': False
            })
            pbar.update()
            print (f'Caught exception {e} during checking rendered cond images ' + rec['sha256'], flush=True)

    def check_vox_worker(rec: np.record, pbar, records):
        try:
            instace_fpath = os.path.join(opt.output_dir, f'voxels/' + rec['sha256'] + '.ply')
            if os.path.exists(instace_fpath):
                pts = utils3d.io.read_ply(instace_fpath)[0]
                records.append({'sha256': rec['sha256'], 'voxelized': True, 'num_voxels': len(pts)})        
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking voxelization ' + rec['sha256'], flush=True)

    def check_image_feats_worker(rec: np.record, pbar, records):
        try:
            instace_fpath = os.path.join(opt.output_dir, f'features/{opt.image_feature_model}/' + rec['sha256'] + '.npz')
            if os.path.exists(instace_fpath):
                records.append({'sha256': rec['sha256'], f'feature_{opt.image_feature_model}': True})        
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking image features ' + rec['sha256'], flush=True)

    def check_ss_latents_worker(rec: np.record, pbar, records):
        try:
            instace_fpath = os.path.join(opt.output_dir, f'ss_latents/{opt.ss_latent_model}/' + rec['sha256'] + '.npz')
            if os.path.exists(instace_fpath):
                records.append({'sha256': rec['sha256'], f'ss_latent_{opt.ss_latent_model}': True})        
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking ss latents ' + rec['sha256'], flush=True)

    def check_slat_latents_worker(rec: np.record, pbar, records):
        slat_name = f'{opt.image_feature_model}_{opt.slat_latent_model}'
        try:
            instace_fpath = os.path.join(opt.output_dir, f'latents/{slat_name}/' + rec['sha256'] + '.npz')
            if os.path.exists(instace_fpath):
                records.append({'sha256': rec['sha256'], f'latent_{slat_name}': True})        
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking slat latents ' + rec['sha256'], flush=True)

    def check_normal_slat_latents_worker(rec: np.record, pbar, records):
        slat_name = f'{opt.image_feature_model}_{opt.slat_latent_model}'
        try:
            instace_fpath = os.path.join(opt.output_dir, f'normal_latents/{slat_name}/' + rec['sha256'] + '.npz')
            if os.path.exists(instace_fpath):
                records.append({'sha256': rec['sha256'], f'normal_latent_{slat_name}': True})        
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking normal slat latents ' + rec['sha256'], flush=True)

    def check_rotated_slat_latents_worker(rec: np.record, pbar, records):
        slat_name = f'{opt.image_feature_model}_{opt.slat_latent_model}'
        try:
            instace_fpath = os.path.join(opt.output_dir, f'latents_rot/{slat_name}/' + rec['sha256'] + '/transforms.json')
            if os.path.exists(instace_fpath):
                with open(instace_fpath, 'r') as f:
                    js = json.load(f)
                    records.append({'sha256': rec['sha256'], f'rotated_latent_{slat_name}': True, 'num_rotations': len(js)})       
            pbar.update()
        except Exception as e:
            pbar.update()
            print (f'Caught exception {e} during checking rotated slat latents ' + rec['sha256'], flush=True)
    
    dataset_name = os.path.basename(opt.output_dir.strip('/'))
    my_metadata_dir = os.path.join(opt.output_dir, "my_metadata")
    os.makedirs(my_metadata_dir, exist_ok=True)
    origin_metadata = pd.read_csv(os.path.join(opt.output_dir, f"{dataset_name}.csv"))
    origin_metadata.set_index('sha256', inplace=True)
    origin_metadata = origin_metadata[~origin_metadata.index.duplicated(keep='first')]
    max_workers = opt.max_workers

    statistics = 'Statistics:\n'
    
    if dataset_name == 'ObjaverseXL_github':
        # 3. check download
        save_repo_format = 'zip'
        base_download_dir = os.path.join(opt.output_dir, "raw/github")
        fs, path = fsspec.core.url_to_fs(base_download_dir)
        downloaded_repo_dirs = fs.glob(
            base_download_dir + f"/repos/*/*.{save_repo_format}"
        )
        downloaded_repo_ids = set()
        for x in downloaded_repo_dirs:
            org, repo = x.split("/")[-2:]
            repo = repo[: -len(f".{save_repo_format}")]
            repo_id = f"{org}/{repo}"
            downloaded_repo_ids.add(repo_id)
        
        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(downloaded_repo_ids), desc="checking downloaded objaversexl github repos ...") as pbar:
            def check_download_worker_git(repo_id):
                try:
                    zip_repo = os.path.join(base_download_dir, f'repos/{repo_id}.{save_repo_format}')
                    with zipfile.ZipFile(zip_repo, "r") as zipf:
                        with zipf.open(".objaverse-file-hashes.json") as f:
                            js = json.load(f)
                    for rec in js:
                        path_in_repo = '/'.join(rec['fileIdentifier'].split('/')[7:])
                        records.append({
                            'sha256': rec['sha256'],
                            'local_path': os.path.join(f'raw/github/repos/{repo_id}.{save_repo_format}', path_in_repo)
                        })
                    pbar.update()
                except Exception as e:
                    pbar.update()
                    print (f'Caught exception {e} during checking downloaded {repo_id}', flush=True)
            executor.map(check_download_worker_git, downloaded_repo_ids)
            executor.shutdown(wait=True)
        metadata_w_local_path = pd.DataFrame.from_records(records)
        metadata_w_local_path.set_index('sha256', inplace=True)
        metadata_w_local_path = metadata_w_local_path[~metadata_w_local_path.index.duplicated(keep='first')]
        origin_metadata = origin_metadata.join(metadata_w_local_path, on='sha256', how='left')
        statistics += f'  - Number of assets: {len(origin_metadata)}\n'
        statistics += f'  - Number of assets downloaded: {len(metadata_w_local_path)}\n'
        origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
        del metadata_w_local_path
        with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
            f.write(statistics)
    
    elif dataset_name == 'ObjaverseXL_sketchfab':
        origin_metadata = origin_metadata.reset_index()
        annotations = oxl.get_annotations()
        annotations = annotations[annotations['sha256'].isin(origin_metadata['sha256'].values)]
        annotations[annotations["source"] == "sketchfab"]

        versioned_dirname = os.path.join(opt.output_dir + "/raw", "hf-objaverse-v1")
        fs, path = fsspec.core.url_to_fs(versioned_dirname)
        existing_file_paths = fs.glob(
            os.path.join(path, "glbs", "*", "*.glb"), refresh=True
        )
        existing_uids = {
            file.split("/")[-1].split(".")[0]
            for file in existing_file_paths
            if file.endswith(".glb")  # note partial files end with .glb.tmp
        }

        annotations["uid"] = annotations.apply(SketchfabDownloader._get_uid, axis=1)
        uids_to_sha256 = dict(zip(annotations["uid"], annotations["sha256"]))
        uids_set = set(uids_to_sha256.keys())
        already_downloaded_uids = uids_set.intersection(existing_uids)

        hf_object_paths = SketchfabDownloader._get_object_paths(
            download_dir=opt.output_dir + "/raw" if opt.output_dir is not None else "~/.objaverse"
        )
        out = {}
        for uid in already_downloaded_uids:
            hf_object_path = hf_object_paths[uid]
            fs_abs_object_path = os.path.join(versioned_dirname, hf_object_path)
            out[SketchfabDownloader.uid_to_file_identifier(uid)] = fs_abs_object_path
        origin_metadata = origin_metadata.set_index("file_identifier")
        records = []
        for k, v in out.items():
            sha256 = origin_metadata.loc[k, "sha256"]
            records.append({
                "sha256": sha256,
                "local_path": os.path.relpath(v, opt.output_dir)
            })
        metadata_w_local_path = pd.DataFrame.from_records(records)
        metadata_w_local_path.set_index('sha256', inplace=True)
        metadata_w_local_path = metadata_w_local_path[~metadata_w_local_path.index.duplicated(keep='first')]
        origin_metadata = origin_metadata.reset_index()
        origin_metadata.set_index('sha256', inplace=True)
        origin_metadata = origin_metadata.join(metadata_w_local_path, on='sha256', how='left')
        statistics += f'  - Number of assets: {len(origin_metadata)}\n'
        statistics += f'  - Number of assets downloaded: {len(metadata_w_local_path)}\n'
        origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
        del metadata_w_local_path
        with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
            f.write(statistics)

    elif dataset_name == '3D-FUTURE':
        origin_metadata = origin_metadata.reset_index()
        origin_metadata = origin_metadata.set_index("file_identifier")
        fs, path = fsspec.core.url_to_fs(os.path.join(opt.output_dir, 'raw'))
        existing_file_paths = [os.path.relpath(fpath, os.path.join(opt.output_dir, 'raw')) for fpath in fs.glob(
            os.path.join(path, "*/*"), refresh=True
        )]
        instances = list(filter(lambda x: x in origin_metadata.index, existing_file_paths))
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(instances), desc=f"Computing sha256s for {dataset_name}") as pbar:
            def extract_3d_future_sha256s_worker(instance: str) -> str:
                try:
                    sha256 = get_file_hash(os.path.join(opt.output_dir, 'raw', f"{instance}/image.jpg"))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            sha256s = executor.map(extract_3d_future_sha256s_worker, instances)
            executor.shutdown(wait=True)

        downloaded = {}
        for k, sha256 in zip(instances, sha256s):
            if sha256 is not None:
                if sha256 == origin_metadata.loc[k, "sha256"]:
                    downloaded[sha256] = os.path.join("raw", f"{k}/raw_model.obj")
                else:
                    print(f"Error downloading {k}: sha256s do not match")

        metadata_w_local_path = pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])
        metadata_w_local_path.set_index('sha256', inplace=True)
        metadata_w_local_path = metadata_w_local_path[~metadata_w_local_path.index.duplicated(keep='first')]
        origin_metadata = origin_metadata.reset_index()
        origin_metadata.set_index('sha256', inplace=True)
        origin_metadata = origin_metadata.join(metadata_w_local_path, on='sha256', how='left')
        statistics += f'  - Number of assets: {len(origin_metadata)}\n'
        statistics += f'  - Number of assets downloaded: {len(metadata_w_local_path)}\n'
        origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
        del metadata_w_local_path
        with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
            f.write(statistics)

    elif dataset_name == 'ABO':
        origin_metadata = origin_metadata.reset_index()
        origin_metadata = origin_metadata.set_index("file_identifier")

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(origin_metadata), desc=f"Computing sha256s for {dataset_name}") as pbar:
            def extract_abo_sha256s_worker(instance: str) -> str:
                try:
                    sha256 = get_file_hash(os.path.join(opt.output_dir, 'raw/3dmodels/original', instance))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            sha256s = executor.map(extract_abo_sha256s_worker, origin_metadata.index)
            executor.shutdown(wait=True)

        downloaded = {}
        for k, sha256 in zip(origin_metadata.index, sha256s):
            if sha256 is not None:
                if sha256 == origin_metadata.loc[k, "sha256"]:
                    downloaded[sha256] = os.path.join('raw/3dmodels/original', k)
                else:
                    print(f"Error downloading {k}: sha256s do not match")

        metadata_w_local_path = pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])
        metadata_w_local_path.set_index('sha256', inplace=True)
        metadata_w_local_path = metadata_w_local_path[~metadata_w_local_path.index.duplicated(keep='first')]
        origin_metadata = origin_metadata.reset_index()
        origin_metadata.set_index('sha256', inplace=True)
        origin_metadata = origin_metadata.join(metadata_w_local_path, on='sha256', how='left')
        statistics += f'  - Number of assets: {len(origin_metadata)}\n'
        statistics += f'  - Number of assets downloaded: {len(metadata_w_local_path)}\n'
        origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
        del metadata_w_local_path
        with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
            f.write(statistics)

    elif dataset_name == 'HSSD':
        origin_metadata = origin_metadata.reset_index()
        origin_metadata = origin_metadata.set_index("file_identifier")

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(origin_metadata), desc=f"Computing sha256s for {dataset_name}") as pbar:
            def extract_hssd_sha256s_worker(instance: str) -> str:
                try:
                    sha256 = get_file_hash(os.path.join(opt.output_dir, 'raw', instance))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            sha256s = executor.map(extract_hssd_sha256s_worker, origin_metadata.index)
            executor.shutdown(wait=True)

        downloaded = {}
        for k, sha256 in zip(origin_metadata.index, sha256s):
            if sha256 is not None:
                if sha256 == origin_metadata.loc[k, "sha256"]:
                    downloaded[sha256] = os.path.join('raw', k)
                else:
                    print(f"Error downloading {k}: sha256s do not match")

        metadata_w_local_path = pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])
        metadata_w_local_path.set_index('sha256', inplace=True)
        metadata_w_local_path = metadata_w_local_path[~metadata_w_local_path.index.duplicated(keep='first')]
        origin_metadata = origin_metadata.reset_index()
        origin_metadata.set_index('sha256', inplace=True)
        origin_metadata = origin_metadata.join(metadata_w_local_path, on='sha256', how='left')
        statistics += f'  - Number of assets: {len(origin_metadata)}\n'
        statistics += f'  - Number of assets downloaded: {len(metadata_w_local_path)}\n'
        origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
        del metadata_w_local_path
        with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
            f.write(statistics)

    def check_attrs(origin_metadata, metadata_filter_func, func, desc, statistics_desc, statistics):
        try:
            origin_metadata = origin_metadata.reset_index()
            metadata_recs = metadata_filter_func(origin_metadata)
            records = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor, \
                tqdm(total=len(metadata_recs), desc=f"checking {desc} from {dataset_name} ...") as pbar:
                executor.map(lambda x:func(x, pbar, records), metadata_recs)
                executor.shutdown(wait=True)
            metadata = pd.DataFrame.from_records(records)
            metadata.set_index('sha256', inplace=True)
            origin_metadata.set_index('sha256', inplace=True)
            origin_metadata = origin_metadata.join(metadata, on='sha256', how='left')
            statistics += f'  - Number of {statistics_desc}: {len(metadata)}\n'
            origin_metadata.to_csv(os.path.join(my_metadata_dir, "metadata.csv"))
            del metadata
            del metadata_recs
            with open(os.path.join(my_metadata_dir, 'statistics.txt'), 'w') as f:
                f.write(statistics)
            return origin_metadata, statistics
        except Exception as e:
            print (f'Errors when checking {desc}: {e}', flush=True)
            return origin_metadata, statistics
    
    # 4. check render mv
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[~df['local_path'].isna()].to_records(),
        check_mv_worker,
        'rendered multi-view images',
        'assets with multi-view images',
        statistics
    )

    # 5. check vox
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['rendered'] == True].to_records(),
        check_vox_worker,
        'voxels',
        'assets with voxels',
        statistics
    )

    # 6. check image feats
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['rendered'] == True].to_records(),
        check_image_feats_worker,
        'image features',
        f'assets with multi-view image features:\n      - {opt.image_feature_model}',
        statistics
    )

    # 7. check ss latents
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['voxelized'] == True].to_records(),
        check_ss_latents_worker,
        'ss latents',
        f'assets with ss latents:\n      - {opt.ss_latent_model}',
        statistics
    )

    # 8. check slat latents
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df[f'feature_{opt.image_feature_model}'] == True].to_records(),
        check_slat_latents_worker,
        'slat latents',
        f'assets with slat latents:\n      - {opt.slat_latent_model}',
        statistics
    )

    # 9. check rendered cond
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[~df['local_path'].isna()].to_records(),
        check_render_cond_worker,
        'rendered condition images',
        'assets with condition images',
        statistics
    )

    # 10. check saved .ply mesh
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['rendered'] == True].to_records(),
        check_saved_ply_worker,
        'saved ply mesh',
        'assets with readable mesh',
        statistics
    )

    # 11. check rotated SLATs
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['voxelized'] == True].to_records(),
        check_rotated_slat_latents_worker,
        'rotated slat latents',
        f'assets with rotated slat latents:\n      - {opt.slat_latent_model}',
        statistics
    )
    
    # 12. check rotated MV
    origin_metadata, statistics = check_attrs(
        origin_metadata,
        lambda df: df[df['num_rotations'] > 0].to_records(),
        check_rot_mv_worker,
        'rotated rendered multi-view images',
        'assets with rotated multi-view images',
        statistics
    )
