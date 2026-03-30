# SPDX-FileCopyrightText: 2026 Ze-Xin Yin and Robot labs of Horizon Robotics
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file in the project root for full license information.

import os
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_temp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
import uuid
from typing import Any, List, Optional, Union

import cv2
import torch
import numpy as np
from PIL import Image
import trimesh
import random
import imageio
from einops import repeat

from gradio_image_prompter import ImagePrompter
import gradio as gr

from threeDFixer.pipelines import ThreeDFixerPipeline
from threeDFixer.datasets.utils import (
    edge_mask_morph_gradient,
    process_scene_image,
    process_instance_image,
    transform_vertices,
    normalize_vertices,
    project2ply
)
from threeDFixer.utils import render_utils, postprocessing_utils
from scripts.grounding_sam2 import plot_segmentation, segment
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import copy

MARKDOWN = """
## Image to 3D Scene with [3D-Fixer](https://zx-yin.github.io/3dfixer/)
1. Upload an image, and draw bounding boxes for each instance by holding and dragging the mouse. Then click "Run Segmentation" to generate the segmentation result.
2. If you find the generated 3D scene satisfactory, download it by clicking the "Download scene GLB" button, and you can also download each islolated 3D instance.
3. In this implementation, we generate each instances one by one, and update the scene results at the "Generated GLB" area, besides, we display isolated instances below.
4. it may take a while for the first time inference due to the usage of ```torch.compile```.
"""
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/example_data")
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_RATIO_THRESHOLD = 0.005
CROP_SIZE = 518
work_space = None
dpt_pack = None
generated_object_map = {}

# Prepare models
## Grounding SAM
sam2_checkpoint = "./checkpoints/sam2-hiera-large/sam2_hiera_large.pt"
sam2_model_cfg = "configs/sam2/sam2_hiera_l.yaml"
sam2_predictor = SAM2ImagePredictor(
    build_sam2(sam2_model_cfg, sam2_checkpoint),
)

############## 3D-Fixer model
model_dir = 'HorizonRobotics/3D-Fixer'
pipeline = ThreeDFixerPipeline.from_pretrained(
    model_dir, compile=True
)
pipeline.cuda()
############## 3D-Fixer model

rot = np.array([
    [-1.0,  0.0,  0.0, 0.0],
    [ 0.0,  0.0,  1.0, 0.0],
    [ 0.0,  1.0,  0.0, 0.0],
    [ 0.0,  0.0,  0.0, 1.0],
], dtype=np.float32)

c2w = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=torch.float32, device=DEVICE)

save_projected_colored_pcd = lambda pts, pts_color, fpath: trimesh.PointCloud(pts.reshape(-1, 3), pts_color.reshape(-1, 3)).export(fpath)

EXAMPLES = [
    [
        {
            "image": "assets/example_data/scene1/rgb.png",
        },
        "assets/example_data/scene1/seg.png",
        1024,
        False,
        25, 5.5, 0.8, 1.0, 5.0
        # num_inference_steps, guidance_scale, cfg_interval_start, cfg_interval_end, t_rescale
    ],
    [
        {
            "image": "assets/example_data/scene2/rgb.png",
        },
        "assets/example_data/scene2/seg.png",
        1,
        False,
        25, 5.0, 0.8, 1.0, 5.0
    ],
    [
        {
            "image": "assets/example_data/scene3/rgb.png",
        },
        "assets/example_data/scene3/seg.png",
        1,
        False,
        25, 5.0, 0.8, 1.0, 5.0
    ],
    [
        {
            "image": "assets/example_data/scene4/rgb.png",
        },
        "assets/example_data/scene4/seg.png",
        42,
        False,
        25, 5.0, 0.8, 1.0, 5.0
    ],
    [
        {
            "image": "assets/example_data/scene5/rgb.png",
        },
        "assets/example_data/scene5/seg.png",
        1,
        False,
        25, 5.0, 0.8, 1.0, 5.0
    ],
    [
        {
            "image": "assets/example_data/scene6/rgb.png",
        },
        "assets/example_data/scene6/seg.png",
        1,
        False,
        25, 5.0, 0.8, 1.0, 5.0
    ]
]

@torch.no_grad()
def run_segmentation(
    image_prompts: Any,
    polygon_refinement: bool = True,
) -> Image.Image:
    rgb_image = image_prompts["image"].convert("RGB")

    global work_space

    # pre-process the layers and get the xyxy boxes of each layer
    if len(image_prompts["points"]) == 0:
        gr.Error("No points provided for segmentation. Please add points to the image.")
        return None
    
    boxes = [
        [
            [int(box[0]), int(box[1]), int(box[3]), int(box[4])]
            for box in image_prompts["points"]
        ]
    ]

    detections = segment(
        sam2_predictor,
        rgb_image,
        boxes=[boxes],
        polygon_refinement=polygon_refinement,
    )
    seg_map_pil = plot_segmentation(rgb_image, detections)

    torch.cuda.empty_cache()

    work_space = os.path.join(TMP_DIR, f"work_space_{uuid.uuid4()}")
    os.makedirs(work_space, exist_ok=True)
    seg_map_pil.save(os.path.join(work_space, 'mask.png'))

    return seg_map_pil

@torch.no_grad()
def run_depth_estimation(
    image_prompts: Any,
    seg_image: Union[str, Image.Image],
) -> Image.Image:
    rgb_image = image_prompts["image"].convert("RGB")

    rgb_image = rgb_image.resize((1024, 1024), Image.Resampling.LANCZOS)

    global dpt_pack
    global work_space
    if work_space is None:
        work_space = os.path.join(TMP_DIR, f"work_space_{uuid.uuid4()}")
        os.makedirs(work_space, exist_ok=True)
    global generated_object_map

    generated_object_map = {}

    origin_W, origin_H = rgb_image.size
    if max(origin_H, origin_W) > 1024:
        factor = max(origin_H, origin_W) / 1024
        H = int(origin_H // factor)
        W = int(origin_W // factor)
        rgb_image = rgb_image.resize((W, H), Image.Resampling.LANCZOS)
    W, H = rgb_image.size

    input_image = np.array(rgb_image).astype(np.float32)
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

    output = pipeline.models['scene_cond_model'].infer(input_image)
    depth = output['depth']
    intrinsics = output['intrinsics']

    invalid_mask = torch.logical_or(torch.isnan(depth), torch.isinf(depth))
    depth_mask = ~invalid_mask

    depth = torch.where(invalid_mask, 0.0, depth)
    K = torch.from_numpy(
        np.array([
            [intrinsics[0, 0].item() * W, 0, 0.5*W],
            [0, intrinsics[1, 1].item() * H, 0.5*H],
            [0, 0, 1]
        ])
    ).to(dtype=torch.float32, device=DEVICE)

    dpt_pack = {
        'c2w': c2w,
        'K': K,
        'depth_mask': depth_mask,
        'depth': depth
    }

    instance_labels = np.unique(np.array(seg_image).reshape(-1, 3), axis=0)
    seg_image = seg_image.resize((W, H), Image.Resampling.LANCZOS)
    seg_image = np.array(seg_image)

    mask_pack = []
    for instance_label in instance_labels:
        if (instance_label == np.array([0, 0, 0])).all():
            continue
        else:
            instance_mask = (seg_image.reshape(-1, 3) == instance_label).all(axis=-1).reshape(H, W)
            mask_pack.append(instance_mask)
    fg_mask = torch.from_numpy(np.stack(mask_pack).any(axis=0)).to(DEVICE)

    scene_est_depth_pts, scene_est_depth_pts_colors = \
        project2ply(depth_mask, depth, input_image, K, c2w)
    save_ply_path = os.path.join(work_space, "scene_pcd.glb")

    fg_depth_pts, _ = \
        project2ply(fg_mask, depth, input_image, K, c2w)
    _, trans, scale = normalize_vertices(fg_depth_pts)

    if trans.shape[0] == 1:
        trans = trans[0]

    dpt_pack.update(
        {
            "trans": trans,
            "scale": scale,
        }
    )
    
    trimesh.PointCloud(scene_est_depth_pts.reshape(-1, 3), scene_est_depth_pts_colors.reshape(-1, 3)).\
        apply_translation(-trans).apply_scale(1. / (scale + 1e-6)).\
        apply_transform(rot).export(save_ply_path)
    
    torch.cuda.empty_cache()

    return save_ply_path


def save_image(img, save_path):
    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    imageio.v3.imwrite(save_path, img)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def export_single_glb_from_outputs(
    outputs,
    fine_scale,
    fine_trans,
    coarse_scale,
    coarse_trans,
    trans,
    scale,
    rot,
    work_space,
    instance_name,
    run_id
):

    with torch.enable_grad():
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
            transform_fn=lambda x: transform_vertices(
                x,
                ops=["scale", "translation", "scale", "translation"],
                params=[fine_scale, fine_trans[None], coarse_scale, coarse_trans[None]],
            ),
            verbose=False
        )

    instance_glb_path = os.path.abspath(
        os.path.join(work_space, f"{run_id}_{instance_name}.glb")
    )

    glb.apply_translation(-trans) \
       .apply_scale(1.0 / (scale + 1e-6)) \
       .apply_transform(rot) \
       .export(instance_glb_path)

    return instance_glb_path, glb


def export_scene_glb(trimeshes, work_space, scene_name):
    scene_path = os.path.abspath(os.path.join(work_space, scene_name))
    trimesh.Scene(trimeshes).export(scene_path)

    return scene_path

@torch.no_grad()
def run_generation(
    rgb_image: Any,
    seg_image: Union[str, Image.Image],
    seed: int,
    randomize_seed: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    cfg_interval_start: float = 0.5,
    cfg_interval_end: float = 1.0,
    t_rescale: float = 3.0,
):
    global dpt_pack
    global work_space
    global generated_object_map
    generated_object_map = {}
    run_id = str(uuid.uuid4())
    
    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    instance_labels = np.unique(np.array(seg_image).reshape(-1, 3), axis=0)
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    set_random_seed(seed)

    H, W = dpt_pack['depth_mask'].shape
    rgb_image = rgb_image.resize((W, H), Image.Resampling.LANCZOS)
    seg_image = seg_image.resize((W, H), Image.Resampling.LANCZOS)

    depth_mask = dpt_pack['depth_mask'].detach().cpu().numpy() > 0

    seg_image = np.array(seg_image)

    mask_pack = []
    for instance_label in instance_labels:
        if (instance_label == np.array([0, 0, 0])).all():
            continue
        else:
            instance_mask = (seg_image.reshape(-1, 3) == instance_label).all(axis=-1).reshape(H, W)
            mask_pack.append(instance_mask)

    erode_kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    results = []
    trimeshes = []

    trans = dpt_pack['trans']
    scale = dpt_pack['scale']

    def build_stream_html(status_text: str):
        cards_html = "".join([
            f"""
            <div style="
                width: 220px;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 8px;
                background: white;
                box-sizing: border-box;
            ">
                <div style="font-weight: 600; margin-bottom: 6px;">
                    {item["name"]}
                </div>

                <video
                    autoplay 
                    muted 
                    loop 
                    playsinline 
                    preload="metadata"
                    poster="/file={item['poster_path']}?v={run_id}"
                    style="
                        width: 100%;
                        border-radius: 8px;
                        display: block;
                        background: #f5f5f5;
                    "
                >
                    <source src="/file={item['mp4_path']}?v={run_id}" type="video/mp4">
                </video>

                <div style="
                    margin-top: 6px;
                    font-size: 12px;
                    color: #666;
                ">
                    Status: {item.get("status_text", "Unknown")}
                </div>

                <div style="
                    margin-top: 4px;
                    font-size: 13px;
                    color: #444;
                    word-break: break-all;
                ">
                    {os.path.basename(item["glb_path"]) if item["glb_path"] is not None else "GLB not ready yet"}
                </div>
            </div>
            """
            for item in results
        ])

        return f"""
        <div style="padding: 8px 0;">
            <div style="font-weight: 700; margin-bottom: 8px;">Status: {status_text}</div>
            <div style="font-weight: 700; margin-bottom: 12px;">Generated objects: {len(results)}</div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start;">
                {cards_html}
            </div>
        </div>
        """

    def build_selector_and_download_updates(default_latest: bool = True):
        object_choices = [item["name"] for item in results if item["glb_path"] is not None]

        if len(object_choices) == 0:
            return (
                gr.update(choices=[], value=None),
                gr.update(value=None, interactive=False),
            )

        selected_value = object_choices[-1] if default_latest else object_choices[0]
        selected_path = generated_object_map[selected_value]

        return (
            gr.update(choices=object_choices, value=selected_value),
            gr.update(value=selected_path, interactive=True),
        )

    # 初始空状态
    yield (
        None,
        build_stream_html("Generating..."),
        gr.update(value=None, interactive=False),
        gr.update(choices=[], value=None),
        gr.update(value=None, interactive=False),
    )

    current_scene_path = None

    for instance_name, object_mask in enumerate(mask_pack):
        try:
            est_depth = dpt_pack['depth'].to('cpu')
            c2w = dpt_pack['c2w'].to('cpu')
            K = dpt_pack['K'].to('cpu')

            intrinsics = dpt_pack['K'].float().to(DEVICE)
            extrinsics = copy.deepcopy(dpt_pack['c2w']).float().to(DEVICE)
            extrinsics[:3, 1:3] *= -1

            object_mask = object_mask > 0
            instance_mask = np.logical_and(object_mask, depth_mask).astype(np.uint8)
            valid_ratio = np.sum((instance_mask > 0).astype(np.float32)) / (H * W)
            print(f'valid ratio of {instance_name}: {valid_ratio:.4f}')
            if valid_ratio < VALID_RATIO_THRESHOLD:
                continue

            ### process condition information
            edge_mask = edge_mask_morph_gradient(instance_mask, kernel, 3)
            fg_mask = (instance_mask > edge_mask).astype(np.uint8)
            color_mask = fg_mask.astype(np.float32) + edge_mask.astype(np.float32) * 0.5

            image = rgb_image
            scene_image, scene_image_masked = process_scene_image(image, instance_mask, CROP_SIZE)
            instance_image, instance_mask, instance_rays_o, instance_rays_d, instance_rays_c, \
                instance_rays_t = process_instance_image(image, instance_mask, color_mask, est_depth, K, c2w, CROP_SIZE)

            save_image(scene_image, os.path.join(work_space, f'input_scene_image_{instance_name}.png'))
            save_image(scene_image_masked, os.path.join(work_space, f'input_scene_image_masked_{instance_name}.png'))
            save_image(instance_image, os.path.join(work_space, f'input_instance_image_{instance_name}.png'))
            save_image(
                torch.cat([instance_image, instance_mask]),
                os.path.join(work_space, f'input_instance_image_masked_{instance_name}.png')
            )

            pcd_points = (
                instance_rays_o.to(DEVICE) +
                instance_rays_d.to(DEVICE) * instance_rays_t[..., None].to(DEVICE)
            ).detach().cpu().numpy()
            pcd_colors = instance_rays_c

            save_projected_colored_pcd(
                pcd_points,
                repeat(pcd_colors, 'n -> n c', c=3),
                f"{work_space}/instance_est_depth_{instance_name}.ply"
            )
            ### process condition information

            ### generate 3D assets
            outputs, coarse_trans, coarse_scale, fine_trans, fine_scale = pipeline.run(
                torch.cat([instance_image, instance_mask]).to(DEVICE),
                scene_image_masked=scene_image_masked.to(DEVICE),
                seed=seed,
                extrinsics=extrinsics.to(DEVICE),
                intrinsics=intrinsics.to(DEVICE),
                points=pcd_points,
                points_mask=pcd_colors,
                sparse_structure_sampler_params={
                    "steps": num_inference_steps,
                    "cfg_strength": guidance_scale,
                    "cfg_interval": [cfg_interval_start, cfg_interval_end],
                    "rescale_t": t_rescale
                },
                slat_sampler_params={
                    "steps": num_inference_steps,
                    "cfg_strength": guidance_scale,
                    "cfg_interval": [cfg_interval_start, cfg_interval_end],
                    "rescale_t": t_rescale
                }
            )

            mp4_path = os.path.abspath(
                os.path.join(work_space, f"{run_id}_instance_gs_fine_{instance_name}.mp4")
            )
            poster_path = os.path.abspath(
                os.path.join(work_space, f"{run_id}_instance_gs_fine_{instance_name}.png")
            )

            video = render_utils.render_video(
                outputs["gaussian"][0],
                bg_color=(1.0, 1.0, 1.0)
            )["color"]
            imageio.mimsave(mp4_path, video, fps=30)
            imageio.imwrite(poster_path, video[0])

            object_label = f"Object {len(results) + 1}"
            result_index = len(results)

            results.append({
                "name": object_label,
                "mp4_path": mp4_path,
                "poster_path": poster_path,
                "glb_path": None,
                "instance_index": instance_name,
                "status_text": "Exporting GLB...",
            })

            # 第一次更新：视频先出来，3D 场景保持当前不变
            selector_update, single_download_update = build_selector_and_download_updates(default_latest=True)
            yield (
                current_scene_path,
                build_stream_html("Generating..."),
                gr.update(value=current_scene_path, interactive=(current_scene_path is not None)),
                selector_update,
                single_download_update,
            )

            # 同步导出 GLB
            instance_glb_path, glb = export_single_glb_from_outputs(
                outputs=outputs,
                fine_scale=fine_scale,
                fine_trans=fine_trans,
                coarse_scale=coarse_scale,
                coarse_trans=coarse_trans,
                trans=trans,
                scale=scale,
                rot=rot,
                work_space=work_space,
                instance_name=instance_name,
                run_id=run_id
            )

            results[result_index]["glb_path"] = instance_glb_path
            results[result_index]["status_text"] = "GLB ready"
            generated_object_map[object_label] = instance_glb_path

            trimeshes.append(glb)
            current_scene_path = export_scene_glb(
                trimeshes=trimeshes,
                work_space=work_space,
                scene_name=f"{run_id}_scene_step_{len(trimeshes)}.glb",
            )

            # 第二次更新：GLB 完成后再更新 3D / 下载 / selector
            selector_update, single_download_update = build_selector_and_download_updates(default_latest=True)
            yield (
                current_scene_path,
                build_stream_html("Generating..."),
                gr.update(value=current_scene_path, interactive=True),
                selector_update,
                single_download_update,
            )

            # del instance_image, instance_mask
            # del instance_rays_o, instance_rays_d, instance_rays_c, instance_rays_t
            # del outputs, coarse_trans, coarse_scale, fine_trans, fine_scale
            # del video
            # gc.collect()
            # torch.cuda.empty_cache()

        except Exception as e:
            print(e)

    # final
    ready_items = [item for item in results if item["glb_path"] is not None]
    if len(ready_items) > 0:
        final_scene_path = export_scene_glb(
            trimeshes=trimeshes,
            work_space=work_space,
            scene_name=f"{run_id}_scene_final.glb",
        )

        selector_update, single_download_update = build_selector_and_download_updates(default_latest=True)

        yield (
            final_scene_path,
            build_stream_html("Finished"),
            gr.update(value=final_scene_path, interactive=True),
            selector_update,
            single_download_update,
        )
    else:
        # 初始空状态
        yield (
            None,
            "<div style='padding: 8px 0;'><b>Status:</b> No valid object generated.</div>",
            gr.update(value=None, interactive=False),
            gr.update(choices=[], value=None),
            gr.update(value=None, interactive=False),
        )

def update_single_download(selected_name):
    global generated_object_map

    if selected_name is None or selected_name not in generated_object_map:
        return gr.update(value=None, interactive=False)

    return gr.update(value=generated_object_map[selected_name], interactive=True)

# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    
    with gr.Column():
        with gr.Row():
            image_prompts = ImagePrompter(label="Input Image", type="pil")
            seg_image = gr.Image(
                label="Segmentation Result", type="pil", format="png"
            )
            with gr.Column():
                with gr.Accordion("Segmentation Settings", open=True):
                    polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=False)
                seg_button = gr.Button("Run Segmentation (step 1)")
                dpt_button = gr.Button("Run Depth estimation (step 2)", variant="primary")
        with gr.Row():
            dpt_model_output = gr.Model3D(label="Estimated depth map", interactive=False)                
            model_output = gr.Model3D(label="Generated GLB", interactive=False)
            with gr.Column():
                with gr.Accordion("Generation Settings", open=True):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25,
                    )
                    with gr.Row():
                        cfg_interval_start = gr.Slider(
                            label="CFG interval start",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.8,
                        )
                        cfg_interval_end = gr.Slider(
                            label="CFG interval end",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                        )
                        t_rescale = gr.Slider(
                            label="t rescale factor",
                            minimum=1.0,
                            maximum=5.0,
                            step=0.1,
                            value=5.0,
                        )
                    guidance_scale = gr.Slider(
                        label="CFG scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=5.0,
                    )
                gen_button = gr.Button("Run Generation (step 3)", variant="primary", interactive=False)
                download_glb = gr.DownloadButton(label="Download scene GLB", interactive=False)
                with gr.Row():
                    object_selector = gr.Dropdown(label="Choose instance: ")
                    download_single_glb = gr.DownloadButton(label="Download single GLB", interactive=False)

        stream_output = gr.HTML(label="Generated Objects Stream")
        with gr.Row():
            gr.Examples(
                examples=EXAMPLES,
                fn=run_generation,
                inputs=[image_prompts, seg_image, seed, randomize_seed, num_inference_steps, guidance_scale, cfg_interval_start, cfg_interval_end, t_rescale],
                outputs=[model_output, download_glb, seed],
                cache_examples=False,
            )

    seg_button.click(
        run_segmentation,
        inputs=[
            image_prompts,
            polygon_refinement,
        ],
        outputs=[seg_image],
    ).then(lambda: gr.Button(interactive=True), outputs=[dpt_button])

    dpt_button.click(
        run_depth_estimation,
        inputs=[
            image_prompts,
            seg_image
        ],
        outputs=[dpt_model_output],
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_button])

    gen_button.click(
        run_generation,
        inputs=[
            image_prompts,
            seg_image,
            seed,
            randomize_seed,
            num_inference_steps,
            guidance_scale,
            cfg_interval_start,
            cfg_interval_end,
            t_rescale
        ],
        outputs=[model_output, 
                 stream_output, 
                 download_glb, 
                 object_selector,
                 download_single_glb],
    )

    object_selector.change(
        update_single_download,
        inputs=[object_selector],
        outputs=[download_single_glb],
    )

demo.launch(allowed_paths=[TMP_DIR, EXAMPLE_DIR])
