# This file is modified from TRELLIS:
# https://github.com/microsoft/TRELLIS
# Original license: MIT
# Copyright (c) the TRELLIS authors
# Modifications Copyright (c) 2026 Ze-Xin Yin, Robot labs of Horizon Robotics, and D-Robotics.

import argparse, sys, os, math, re, glob
from typing import *
import bpy
import random
from mathutils import Vector, Matrix, Euler
import numpy as np
import json
import glob
import copy


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.wm.usd_import,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.wm.stl_import,
    "usda": bpy.ops.wm.usd_import,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

FLOOR_SIZE = 5.0
BBOX_LEN = 2.0
BBOX_LINE = BBOX_LEN**2 * 3
DEPTH_BBOX_LINE = (FLOOR_SIZE * 1.5)**2 * 3
INDEX_OFFSET = 2 # 0 for non, 1 for wall, and 2 for floor
MARGINS = 0.5

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    # 
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
    # 
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False,
               save_index=False):
    if not any([save_depth, save_normal, save_albedo, save_mist, save_index]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    # 
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['ViewLayer'].use_pass_mist = save_mist
    bpy.context.scene.view_layers['ViewLayer'].use_pass_object_index = save_index
    # 
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    # 
    render_layers = nodes.new('CompositorNodeRLayers')
    # 
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 20 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        # 
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        # 
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    # 
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        # 
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        # 
        outputs['normal'] = normal_file_output
    # 
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        # 
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        # 
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        # 
        outputs['albedo'] = albedo_file_output
    # 
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        # 
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        # 
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        # 
        outputs['mist'] = mist_file_output
    if save_index:
        index_file_output = nodes.new('CompositorNodeOutputFile')
        index_file_output.base_path = ''
        index_file_output.file_slots[0].use_node_format = True
        index_file_output.format.file_format = 'PNG'
        index_file_output.format.color_mode = 'BW'
        index_file_output.format.color_depth = '16'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 100.0 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        # 
        links.new(render_layers.outputs['IndexOB'], map.inputs[0])
        links.new(map.outputs[0], index_file_output.inputs[0])
        # 
        outputs['index'] = index_file_output
    # 
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.
    # 
    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    # 
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # 
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # 
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    # 
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (2, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    # 
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, FLOOR_SIZE)
    top_light.scale = (100, 100, 100)
    # 
    # # create bottom light
    # bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    # bpy.context.collection.objects.link(bottom_light)
    # bottom_light.data.energy = 1000
    # bottom_light.location = (0, 0, -10)
    # bottom_light.rotation_euler = (0, 0, 0)
    # # 
    return {
        "default_light": default_light,
        "top_light": top_light,
        # "bottom_light": bottom_light
    }


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.
    # 
    Args:
        object_path (str): Path to the model file.
    # 
    Raises:
        ValueError: If the file extension is not supported.
    # 
    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")
    # 
    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz
        # 
        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None
    # 
    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]
    # 
    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)

def delete_objects(parent_root) -> None:
    for o in (parent_root.children_recursive+[parent_root]):
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.ops.object.delete()

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.
    # 
    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()
    # 
    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.
    # 
    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.
    # 
    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.
    # 
    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.
    # 
    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)
    # 
    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if (isinstance(obj.data, bpy.types.Mesh) and not obj.hide_render)]
    print (scene_meshes)
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.
    # 
    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).
    # 
    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        # 
        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]
    # 
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale
    # 
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    # 
    return scale, offset

def set_obj_index(parent_root, index):
    children_recursive = parent_root.children_recursive + [parent_root]
    scene_meshes = [obj for obj in children_recursive if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        obj.pass_index = index

def normalize_obj(loaded_obj_list) -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.
    # 
    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).
    # 
    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    scene_root_objects = list(filter(lambda x: x not in loaded_obj_list, scene_root_objects))
    print (scene_root_objects)
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        # 
        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]
    # 
    bbox_min, bbox_max = obj_bbox(scene)
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale
    # 
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = obj_bbox(scene)
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")
    # 
    return scene, scale, offset

def aabb_intersect(a_min, a_max, b_min, b_max):
    """检查两个 AABB 是否相交"""
    for i in range(3):
        if a_max[i] < b_min[i] or a_min[i] > b_max[i]:
            return False
    return True

def aabb_inside(a_min, a_max, b_min, b_max):
    """检查A是否在B范围内"""
    for i in range(3):
        if a_min[i] < b_min[i] or a_max[i] > b_max[i]:
            return False
    return True

def attempt_randomize_obj(obj, loaded_obj_list, floor_size=FLOOR_SIZE, default_size=1.0):
    z_rot = np.random.rand() * 2.0 * np.pi
    org_rot_mode = obj.rotation_mode
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = Euler((0.0, 0.0, z_rot))
    obj.rotation_mode = org_rot_mode
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # random scale
    rand_scaling_factor = random.uniform(0.5, 2.0)
    obj.scale *= rand_scaling_factor
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # random position
    a_min, a_max = obj_bbox(obj)
    x_min = -(floor_size / 2.0 - abs(a_min[0]))
    x_max = floor_size / 2.0 - abs(a_max[0])
    y_min = -(floor_size / 2.0 - abs(a_min[1]))
    y_max = floor_size / 2.0 - abs(a_max[1])
    x, y = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
    z = obj.location[2]
    z_move = - default_size / 2.0 - a_min[2]
    target_location = Vector((x, y, z + z_move))
    trans = obj.location - target_location
    obj.location = obj.location - trans
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # 
    a_min, a_max = obj_bbox(obj)
    collision = False
    for other in loaded_obj_list:
        b_min, b_max = obj_bbox(other)
        if aabb_intersect(a_min, a_max, b_min, b_max):
            collision = True
            break
    scale = rand_scaling_factor
    trans = (trans[0], trans[1], trans[2])
    rot = (0.0, 0.0, z_rot)
    return not collision, obj, scale, trans, rot
    # if not collision:
    #     scale = rand_scaling_factor
    #     trans = (trans[0], trans[1], trans[2])
    #     rot = (0.0, 0.0, z_rot)
    #     return True, obj, scale, trans, rot
    # return False, obj, None, None, None

def derandomize_obj(obj, scale, trans, rot):
    # 
    trans_vec = Vector((trans[0], trans[1], trans[2]))
    obj.location = obj.location + trans_vec
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # 
    obj.scale /= scale
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # 
    org_rot_mode = obj.rotation_mode
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = Euler((0.0, 0.0, -rot[2]))
    obj.rotation_mode = org_rot_mode
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def randomize_obj(obj, loaded_obj_list, floor_size=FLOOR_SIZE, default_size=1.0, max_attempts = 100):
    obj_origin_scale = copy.deepcopy(obj.scale)
    obj_origin_dim = copy.deepcopy(obj.dimensions)
    obj_origin_rot = copy.deepcopy(obj.rotation_quaternion)
    obj_origin_location = copy.deepcopy(obj.location)
    for attempt in range(max_attempts):
        # random rotation
        # z_rot = random.uniform(-np.pi / 2.0, np.pi / 2.0)
        z_rot = np.random.rand() * 2.0 * np.pi
        org_rot_mode = obj.rotation_mode
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = Euler((0.0, 0.0, z_rot))
        obj.rotation_mode = org_rot_mode
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # random scale
        rand_scaling_factor = random.uniform(0.5, 2.0)
        obj.scale *= rand_scaling_factor
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # random position
        a_min, a_max = obj_bbox(obj)
        x_min = -(floor_size / 2.0 - abs(a_min[0]))
        x_max = floor_size / 2.0 - abs(a_max[0])
        y_min = -(floor_size / 2.0 - abs(a_min[1]))
        y_max = floor_size / 2.0 - abs(a_max[1])
        x, y = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
        z = obj.location[2]
        z_move = - default_size / 2.0 - a_min[2]
        target_location = Vector((x, y, z + z_move))
        trans = obj.location - target_location
        obj.location = obj.location - trans
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # 
        a_min, a_max = obj_bbox(obj)
        collision = False
        for other in loaded_obj_list:
            b_min, b_max = obj_bbox(other)
            if aabb_intersect(a_min, a_max, b_min, b_max):
                collision = True
                break
            # if not aabb_inside(a_min, a_max, (-floor_size/2.0, -floor_size/2.0, -default_size/2.0), (floor_size/2.0, floor_size/2.0, floor_size/2.0-default_size/2.0)):
            #     collision = True
            #     break
        if not collision:
            scale = rand_scaling_factor
            trans = (trans[0], trans[1], trans[2])
            rot = (0.0, 0.0, z_rot)
            # bpy.ops.object.select_all(action="DESELECT")
            # obj.select_set(True)
            # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # bpy.ops.object.select_all(action="DESELECT")
            return True, obj, scale, trans, rot
        else:
            obj.location = obj.location + trans
            bpy.context.view_layer.update()
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            obj.scale /= rand_scaling_factor
            bpy.context.view_layer.update()
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            org_rot_mode = obj.rotation_mode
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = Euler((0.0, 0.0, z_rot))
            obj.rotation_mode = org_rot_mode
            bpy.context.view_layer.update()
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # obj.dimensions = obj_origin_dim
            # obj.rotation_quaternion = obj_origin_rot
            
            
            # bpy.ops.object.select_all(action="DESELECT")
            # obj.select_set(True)
            # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # bpy.ops.object.select_all(action="DESELECT")
    return False, obj, None, None, None


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def obj_bbox(parent_root) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.
    # 
    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)
    # 
    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    children_recursive = parent_root.children_recursive + [parent_root]
    scene_meshes = [obj for obj in children_recursive if isinstance(obj.data, bpy.types.Mesh)]
    print (scene_meshes)
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_blend_files(folder):
    """返回指定文件夹下所有.blend文件路径"""
    return [f for f in glob.glob(os.path.join(folder, '*/*.blend'))]

def load_random_material(blend_files):
    """随机选择一个.blend文件并加载其中的一个材质对象"""
    # 
    blend_file = random.choice(blend_files)
    with bpy.data.libraries.load(blend_file, link=False) as (data_from, data_to):
        if not data_from.materials:
            return None, blend_file
        mat_name = random.choice(data_from.materials)
        bpy.ops.wm.append(
            filepath=os.path.join(blend_file, "Material", mat_name),
            directory=os.path.join(blend_file, "Material"),
            filename=mat_name
        )
        material = bpy.data.materials.get(mat_name)  # 获取真正材质对象
        return material, blend_file

def set_envs(floor_mat_list, wall_mat_list, hdr_list, floor_size=FLOOR_SIZE*1.5):
    # 
    hdr_file = random.choice(hdr_list)
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    env_texture = nodes.new(type='ShaderNodeTexEnvironment')
    try:
        env_texture.image = bpy.data.images.load(hdr_file)
        background = nodes['Background']
        world.node_tree.links.new(env_texture.outputs['Color'], background.inputs['Color'])
        bpy.context.scene.render.film_transparent = False
    except Exception as e:
        hdr_file = ''
        print (f'error loading hdr: {e}')

    # 
    # 添加地板
    bpy.ops.mesh.primitive_plane_add(size=floor_size, enter_editmode=False, 
                                     align='WORLD', location=(0, 0, -0.5), scale=(1, 1, 1))
    floor = bpy.context.active_object
    floor.name = "MyManuallyAddedFloor"
    floor_mat, floor_mat_file = load_random_material(floor_mat_list)
    bpy.data.objects['MyManuallyAddedFloor'].pass_index = 1
    if floor_mat:
        if floor.data.materials:
            floor.data.materials[0] = floor_mat
        else:
            floor.data.materials.append(floor_mat)
    # 
    # 添加墙体 (四个方向)
    wall_positions = [
        ((0, floor_size/2.0, floor_size/2.0-0.5), (np.deg2rad(90), 0, 0)),
        ((0, -floor_size/2.0, floor_size/2.0-0.5), (np.deg2rad(-90), 0, 0)),
        ((floor_size/2.0, 0, floor_size/2.0-0.5), (0, np.deg2rad(-90), 0)),
        ((-floor_size/2.0, 0, floor_size/2.0-0.5), (0, np.deg2rad(90), 0)),
    ]
    # 
    wall_mat_file_list = []
    wall_index = 1
    for loc, rot in wall_positions:
        bpy.ops.mesh.primitive_plane_add(size=floor_size, enter_editmode=False,
                                         align='WORLD', location=loc, rotation=rot, scale=(1, 1, 1))
        wall = bpy.context.active_object
        wall.name = f"MyManuallyAddedWall_{wall_index:03d}"
        wall_index += 1
        wall_mat, wall_mat_file = load_random_material(wall_mat_list)
        wall_mat_file_list.append(wall_mat_file)
        if wall_mat:
            if wall.data.materials:
                wall.data.materials[0] = wall_mat
            else:
                wall.data.materials.append(wall_mat)
    
    bpy.data.objects['MyManuallyAddedWall_001'].pass_index = 2
    bpy.data.objects['MyManuallyAddedWall_002'].pass_index = 2
    bpy.data.objects['MyManuallyAddedWall_003'].pass_index = 2
    bpy.data.objects['MyManuallyAddedWall_004'].pass_index = 2

    return floor_mat_file, wall_mat_file_list, hdr_file

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_upper_hemisphere(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    # 
    # 1. 应用抖动并归一化
    u = (u + offset[0] / num_samples) % 1.0
    v = (v + offset[1]) % 1.0
    # 
    # 2. 强制 u ∈ [0.5, 1.0) 确保上半球
    u = 0.5 + u / 2.0
    # 
    # 3. 适配上半球的非线性变换
    # if u < 0.125:
    #     u = 2 * u
    # else: ### TODO: check
    u = (2 / 3) * u + 1 / 6
    # 
    # 4. 球面映射（theta ≥ 0 恒成立）
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def generate_views(num_views = 10):
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_upper_hemisphere(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 20, 70
    radius_min = np.sqrt(BBOX_LINE) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(BBOX_LINE) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (num_views,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(BBOX_LINE) / 2 / r) for r in radius]
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views

def get_instance_pose(obj, cam):
    # 世界变换矩阵
    M_world = obj.matrix_world
    # 相机坐标系逆矩阵
    M_cam_inv = cam.matrix_world.inverted()
    # 物体相对相机的变换矩阵
    M_obj_cam = M_cam_inv @ M_world
    # 提取平移向量 (相机坐标系下的位置)
    translation = M_obj_cam.to_translation()
    # 提取旋转 (相机坐标系下的四元数)
    rotation_quat = M_obj_cam.to_quaternion()
    return translation, rotation_quat

def main(arg):
    # mv to 
    # os.makedirs(arg.output_folder, exist_ok=True)

    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_index=arg.save_index,
    )
    init_scene()

    loaded_obj_list = []
    instance_info = {}
    max_attempts = 100
    for sha256, fpath in arg.object.items():
        load_object(fpath)
        obj, scale, offset = normalize_obj(loaded_obj_list)
        for _ in range(max_attempts):
            success, placed_obj, rand_scale, rand_trans, rand_rot = attempt_randomize_obj(obj, loaded_obj_list)
            if success:
                break
            else:
                derandomize_obj(obj, rand_scale, rand_trans, rand_rot)
        if success:
            loaded_obj_list.append(obj)
            set_obj_index(obj, len(loaded_obj_list) + INDEX_OFFSET)
            instance_info[len(loaded_obj_list) + INDEX_OFFSET] = {
                'sha256': sha256,
                'scale': scale,
                "offset": [offset.x, offset.y, offset.z],
                'rand_scale': rand_scale,
                "rand_trans": [rand_trans[0], rand_trans[1], rand_trans[2]],
                "rand_rot": [rand_rot[0], rand_rot[1], rand_rot[2]],
                "transform_matrix": get_transform_matrix(placed_obj)
            }
        else:
            delete_objects(obj)

    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # set room
    floor_mat_file, wall_mat_files, hdr_file = set_envs(arg.floor, arg.wall, arg.hdrs)
    wall_001_visible = np.random.rand() < 0.9
    wall_002_visible = np.random.rand() < 0.9
    wall_003_visible = np.random.rand() < 0.9
    wall_004_visible = np.random.rand() < 0.9

    # Create a list of views
    to_export = {
        "aabb": [[-FLOOR_SIZE/2.0, -FLOOR_SIZE/2.0, -FLOOR_SIZE/2.0], [FLOOR_SIZE/2.0, FLOOR_SIZE/2.0, FLOOR_SIZE/2.0]],
        "instance": instance_info,
        "floor": floor_mat_file,
        "wall": wall_mat_files,
        "hdr": hdr_file,
        "wall_001_visible": wall_001_visible,
        "wall_002_visible": wall_002_visible,
        "wall_003_visible": wall_003_visible,
        "wall_004_visible": wall_004_visible,
        "frames": []
    }

    os.makedirs(arg.output_folder, exist_ok=True)

    views = generate_views(arg.num_views)
    for i, view in enumerate(views):
        bpy.data.objects['MyManuallyAddedWall_001'].hide_render=not wall_001_visible
        bpy.data.objects['MyManuallyAddedWall_002'].hide_render=not wall_002_visible
        bpy.data.objects['MyManuallyAddedWall_003'].hide_render=not wall_003_visible
        bpy.data.objects['MyManuallyAddedWall_004'].hide_render=not wall_004_visible
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        location_x = view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch'])
        location_y = view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch'])
        location_z = view['radius'] * np.sin(view['pitch'])
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        if location_y + MARGINS >= bpy.data.objects['MyManuallyAddedWall_001'].location[1]:
            bpy.data.objects['MyManuallyAddedWall_001'].hide_render=True
        if location_y - MARGINS <= bpy.data.objects['MyManuallyAddedWall_002'].location[1]:
            bpy.data.objects['MyManuallyAddedWall_002'].hide_render=True
        if location_x + MARGINS >= bpy.data.objects['MyManuallyAddedWall_003'].location[0]:
            bpy.data.objects['MyManuallyAddedWall_003'].hide_render=True
        if location_x - MARGINS <= bpy.data.objects['MyManuallyAddedWall_004'].location[0]:
            bpy.data.objects['MyManuallyAddedWall_004'].hide_render=True
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        # 
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(DEPTH_BBOX_LINE)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(DEPTH_BBOX_LINE)
        # 
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.base_path = arg.output_folder
            output.file_slots[0].path = f'{i:03d}_{name}'
            # output.file_slots[0].path = bpy.path.abspath(os.path.join(arg.output_folder, f'{i:03d}_{name}'))
            # print (name, os.path.join(arg.output_folder, f'{i:03d}_{name}'), print(os.path.isabs(os.path.join(arg.output_folder, f'{i:03d}_{name}'))))
        # 
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            print (f'{output.base_path}/{output.file_slots[0].path}')
            print (glob.glob(f'{output.base_path}/{output.file_slots[0].path}*.{ext}'))
            path = glob.glob(f'{output.base_path}/{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.base_path}/{output.file_slots[0].path}.{ext}')
        # 
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam),
            "instances_in_the_view": {},
        }
        for obj_idx, obj in enumerate(loaded_obj_list):
            translation, rotation_quat = get_instance_pose(obj, cam)
            metadata['instances_in_the_view'][obj_idx+1 + INDEX_OFFSET] = {
                'translation': [t for t in translation],
                'rotation': [rot for rot in rotation_quat]
            }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(DEPTH_BBOX_LINE),
                'max': view['radius'] + 0.5 * np.sqrt(DEPTH_BBOX_LINE)
            }
        # 
        to_export["frames"].append(metadata)
            
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)

    if arg.save_mesh:
        delete_objects(bpy.data.objects['MyManuallyAddedFloor'])
        delete_objects(bpy.data.objects['MyManuallyAddedWall_001'])
        delete_objects(bpy.data.objects['MyManuallyAddedWall_002'])
        delete_objects(bpy.data.objects['MyManuallyAddedWall_003'])
        delete_objects(bpy.data.objects['MyManuallyAddedWall_004'])
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        # export ply mesh
        bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, 'mesh.ply'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--num_views', type=int, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=json.loads, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--floor', type=json.loads, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--wall', type=json.loads, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--hdrs', type=json.loads, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--save_index', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    
