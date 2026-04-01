# ARSG-110K

ARSG-110K is a dataset of 110K+ scene-level dataset curated from [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md).
This dataset serves for 3D scene generation tasks.

## Dataset Location

The dataset is hosted on Hugging Face Datasets. You can preview the dataset at

[https://huggingface.co/datasets/HorizonRobotics/ARSG-110K](https://huggingface.co/datasets/HorizonRobotics/ARSG-110K)

There is no need to download the csv files manually. We provide toolkits to load and prepare the dataset.

## Dataset Toolkits (Object-level processing)

We provide [toolkits](dataset_toolkits) for data preparation.

**We first prepare object-level dataset following the pipline from [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md), as the following step 1-9.**

### Step 1: Install Dependencies

```
. ./dataset_toolkits/setup.sh
```

### Step 2: Load Metadata

First, we need to load the metadata of the dataset.

```
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <OUTPUT_DIR> [--source <SOURCE>]
```

- `SUBSET`: The subset of the dataset to load. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `SOURCE`: Required if `SUBSET` is `ObjaverseXL`. Options are `sketchfab` and `github`.

For example, to load the metadata of the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

Next, we need to download the 3D assets.

```
python dataset_toolkits/download.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to download. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.

You can also specify the `RANK` and `WORLD_SIZE` of the current process if you are using multiple nodes for data preparation.

For example, to download the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run: 

***NOTE: The example command below sets a large `WORLD_SIZE` for demonstration purposes. Only a small portion of the dataset will be downloaded.***

```
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 160000
```

Some datasets may require interactive login to Hugging Face or manual downloading. Please follow the instructions given by the toolkits.

After downloading, update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 4: Render Multiview Images (& Calculate Aesthetic Scores)

Multiview images can be rendered with:

```
python dataset_toolkits/render.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 150.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

(Optional) If you want to calculate the aesthetic scores of your own rendered datasets, you can use the following command:

```
python dataset_toolkits/calculate_aesthetic_scores.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```
- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

Don't forget to update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 5: Voxelize 3D Models

We can voxelize the 3D models with:

```
python dataset_toolkits/voxelize.py <SUBSET> --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to voxelize. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to voxelize the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:
```
python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 6: Extract DINO Features

To prepare the training data for SLat VAE, we need to extract DINO features from multiview images and aggregate them into sparse voxel grids.

```
python dataset_toolkits/extract_features.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.


For example, to extract DINO features from the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 7: Encode Sparse Structures

Encoding the sparse structures into latents to train the first stage generator:

```
python dataset_toolkits/encode_ss_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode the sparse structures into latents for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 8: Encode SLat

Encoding SLat for second stage generator training:

```
python dataset_toolkits/encode_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode SLat for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 9: Render Image Conditions

To train the image conditioned generator, we need to render image conditions with augmented views.

```
python dataset_toolkits/render_cond.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 24.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render image conditions for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

**Please Note: we use ```construct_my_metadata.py``` to generate the metadata for our process pipeline. After running the above steps, please update our metadata with:**

```
python dataset_toolkits/construct_my_metadata.py --output_dir datasets/ObjaverseXL_sketchfab
```

## Dataset Toolkits (Scene-level processing)

### Step 10: Generate randomly rotated SLATs

To fine-tune the SLAT decoders and the second stage generator to enrich the priors, we need to generate randomly rotated SLATs.

```
python dataset_toolkits/random_rotate_slat.py --output_dir <OUTPUT_DIR> [--rand_rot_times <RAND_ROT_TIMES>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RAND_ROT_TIMES`: Maximal rotation numbers for each instance.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to generate randomly rotated SLATs for the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/random_rotate_slat.py --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/construct_my_metadata.py --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 11: Render Multiview Images corresponding to the rotated SLATs

To fine-tune the 3DGS decoder, we rotate 3D assets according to the above step, as the training of this model requires image-based rendering loss.

```
python dataset_toolkits/render_with_rotated_slats.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to render. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 150.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render the ObjaverseXL (sketchfab) subset and save it to `datasets/ObjaverseXL_sketchfab`, we can run:

```
python dataset_toolkits/render_with_rotated_slats.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

Then update the metadata file with:

```
python dataset_toolkits/construct_my_metadata.py --output_dir datasets/ObjaverseXL_sketchfab
```

### Step 12: Render scene images

Download the scene information ```ARSG-110K.zip```, and the auxiliary assets including HDR maps (```hdrs.zip```) and texture maps (```materials_floor.zip``` and ```materials_wall.zip```). Unzip and put the auxiliary assets in a folder, and unzip the ```ARSG-110K.zip```.

```
python dataset_toolkits/render_scene.py 
    --obj_root_dir <OBJ_ROOT_DIR> \
    --obj_asset_dir <OBJ_ASSET_DIR> \
    --scene_data_dir <SCENE_DATA_DIR> \
    --auxiliary_assets_dir <AUXILIARY_ASSETS_DIR> \
    [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OBJ_ROOT_DIR`: The parent directory of all assets.
- `OBJ_ASSET_DIR`: The directory to object assets, for example, '<ROOT_DIR>/3D-FUTURE,<ROOT_DIR>/ABO/...'.
- `SCENE_DATA_DIR`: The directory to the unziped ```ARSG-110K.zip```.
- `AUXILIARY_ASSETS_DIR`: The directory put HDR maps and texture maps.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

After rendering scene data, run the following command to group the training data records.

```
python dataset_toolkits/group_valid_scene_renderings.py \
    --obj_asset_dir <OBJ_ASSET_DIR> \
    --obj_root_dir <OBJ_ROOT_DIR> \
    --scene_data_dir <SCENE_DATA_DIR>
```

### Step 13: Estimate depth maps

Finally, you can use the following commands to estimate depth maps using MoGe v2, Depth-pro, Depth-Anything v2, and VGGT.

#### MoGe v2:

```
python dataset_toolkits/estimate_moge_v2_depth.py \
    --dpt_model_dir <DPT_MODEL_DIR> \
    --scene_root_dir <SCENE_DATA_DIR> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `DPT_MODEL_DIR`: The path to load MoGe v2 ckpts.
- `SCENE_DATA_DIR`: The directory to the rendered scenes.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

#### Depth-pro:

Please follow the instruction from [depth-pro](https://github.com/apple/ml-depth-pro) to install the module, download the checkpoints, and put the checkpoints in ```./checkpoints```.

```
python dataset_toolkits/estimate_dpt_pro_depth.py \
    --scene_root_dir <SCENE_DATA_DIR> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SCENE_DATA_DIR`: The directory to the rendered scenes.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

#### Depth-Anything v2:

Use the following commands to prepare depth anything v2 code:

```
git clone https://github.com/DepthAnything/Depth-Anything-V2.git /tmp/Depth-Anything-V2
cp -r /tmp/Depth-Anything-V2/depth_anything_v2 ./dataset_toolkits
```

Then download the ckpts and use the commands:

```
python dataset_toolkits/estimate_dpt_pro_depth.py \
    --dpt_model_dir <DPT_MODEL_DIR> \
    --dpt_encoder <DPT_ENCODER> \
    --scene_root_dir <SCENE_DATA_DIR> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `DPT_MODEL_DIR`: The path to load MoGe v2 ckpts.
- `DPT_ENCODER`: The Encoder version for depth-anything v2.
- `SCENE_DATA_DIR`: The directory to the rendered scenes.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

#### VGGT:

Use the following commands to prepare VGGT code:

```
git clone https://github.com/facebookresearch/vggt.git /tmp/vggt
cp -r /tmp/vggt/vggt ./dataset_toolkits
```

Then download the ckpts and use the commands:

```
python dataset_toolkits/estimate_vggt_depth.py \
    --vggt_model_dir <VGGT_MODEL_DIR> \
    --scene_root_dir <SCENE_DATA_DIR> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `VGGT_MODEL_DIR`: The path to load VGGT ckpts.
- `SCENE_DATA_DIR`: The directory to the rendered scenes.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

### Step 14 (Optional): Create more new scenes

If you wish to create more new scenes, you can use the following commands:

```
python dataset_toolkits/create_new_scene.py 
    --obj_root_dir <OBJ_ROOT_DIR> \
    --obj_asset_dir <OBJ_ASSET_DIR> \
    --scene_data_dir <SCENE_DATA_DIR> \
    --auxiliary_assets_dir <AUXILIARY_ASSETS_DIR> \
    --num_scenes <NUM_SCENES> \
    [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OBJ_ROOT_DIR`: The parent directory of all assets.
- `OBJ_ASSET_DIR`: The directory to object assets, for example, '<ROOT_DIR>/3D-FUTURE,<ROOT_DIR>/ABO/...'.
- `SCENE_DATA_DIR`: The directory to save newly created scenes.
- `AUXILIARY_ASSETS_DIR`: The directory put HDR maps and texture maps.
- `NUM_SCENES`: Number of scenes you wish to create.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.
