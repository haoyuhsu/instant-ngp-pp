import pickle
import numpy as np
import os
import sys
import bpy
import math
import shutil
import json
import time
from mathutils import Vector, Matrix
import argparse
import glob


context = bpy.context
scene = context.scene
render = scene.render

object_dict = dict()  # object path -> object_name

# TODO: handle the case when there are multiple objects in the scene
# existing_objects = set(scene.objects)
# all_objects = set(scene.objects)
# new_objects = all_objects - existing_objects


# Function to ensure collection is visible and renderable
def ensure_collection_visibility(collection_name):
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        collection.hide_viewport = False  # Ensure collection is visible in the viewport
        collection.hide_render = False    # Ensure collection is enabled for rendering
    else:
        print(f"Collection '{collection_name}' not found.")

ensure_collection_visibility("Collection") # Ensure default collection is visible and renderable


def enable_render_for_all_objects():
    for obj in bpy.data.objects:
        obj.hide_viewport = False # Ensure the object is visible in the viewport
        obj.hide_render = False  # Ensure the object is visible in the render

enable_render_for_all_objects() # Ensure all objects are visible in the render


# Stackoverflow: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


# get rotation matrix from aligning one vector to another vector
# link: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def get_rot_mat(v1, v2):
    """
    Calculate rotation matrix to align vector v1 to vector v2

    Args:
        v1: (3,) vector
        v2: (3,) vector

    Returns:
        R: (3, 3) rotation matrix
    """
    # if two numpy array are the same, return identity matrix
    if np.allclose(v1, v2):
        return np.eye(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)
    return R 


def rotate_obj(obj, R):
    """
    Apply rotation matrix to blender object

    Args:
        obj: blender object
        R: (3, 3) rotation matrix
    """
    R = Matrix(R)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = R.to_quaternion()


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        # if obj.type not in {"CAMERA", "LIGHT"}:
        #     bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else object_meshes(single_obj):
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def object_meshes(single_obj):
    for obj in [single_obj] + single_obj.children_recursive:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene(single_obj=None):
    bbox_min, bbox_max = scene_bbox(single_obj)
    scale = 1 / max(bbox_max - bbox_min)
    # for obj in scene_root_objects():
    #     obj.scale = obj.scale * scale
    single_obj.scale = single_obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox(single_obj)
    offset = -(bbox_min + bbox_max) / 2
    # for obj in scene_root_objects():
    #     obj.matrix_world.translation += offset
    single_obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def duplicate_hierarchy(obj, parent=None):
    """Recursively duplicate an object and all its children."""
    # Duplicate the object (without the data)
    new_obj = obj.copy()
    # Link the object data if it exists (for meshes, curves, etc.)
    if new_obj.data:
        new_obj.data = obj.data.copy()
    # If a parent is specified, set the duplicated object's parent
    if parent:
        new_obj.parent = parent
    # Link the new object to the collection
    bpy.context.collection.objects.link(new_obj)
    # Recursively duplicate children
    for child in obj.children:
        duplicate_hierarchy(child, new_obj)
    return new_obj


def create_linked_duplicate(object_name: str) -> None:
    """Creates n linked duplicate of the given object."""
    original_obj = bpy.data.objects.get(object_name)
    if original_obj:
        new_obj = duplicate_hierarchy(original_obj)
    else:
        new_obj = None
        print(f"Object '{object_name}' not found.")
    return new_obj


def load_object(object_path: str) -> bpy.types.Object:
    """Loads a glb model into the scene."""
    # check if the same object has been loaded before
    if object_path in object_dict:
        print("Object {} already loaded.".format(object_path))
        new_obj = create_linked_duplicate(object_dict[object_path])
        return new_obj
    # import the object
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    # Find the most recently added object
    new_obj = bpy.context.object
    object_name = new_obj.name
    # Store the object name in the dictionary
    object_dict[object_path] = object_name
    return new_obj


def setup_camera():
    # Find a camera in the scene
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            print("found camera")
            break
    # If no camera is found, create a new one
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    # Set the camera as the active camera for the scene
    bpy.context.scene.camera = cam
    return cam


class Camera():
    def __init__(self, im_height, im_width, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.w = im_width
        self.h = im_height
        self.camera = setup_camera()
        
    def set_camera(self, K, c2w):
        self.K = K       # (3, 3)
        self.c2w = c2w   # (3 or 4, 4), camera to world transform
        # original camera model: x: right, y: down, z: forward (OpenCV, COLMAP format)
        # Blender camera model:  x: right, y: up  , z: backward (OpenGL, NeRF format)
        
        self.camera.data.type = 'PERSP'
        self.camera.data.lens_unit = 'FOV'
        f = K[0, 0]
        rad = 2 * np.arctan(self.w/(2 * f))
        self.camera.data.angle = rad
        self.camera.data.sensor_fit = 'HORIZONTAL'
        
        self.pose = self.transform_pose(c2w)
        self.camera.matrix_world = Matrix(self.pose)
        
    def transform_pose(self, pose):
        '''
        Transform camera-to-world matrix
        Input:  (3 or 4, 4) x: right, y: down, z: forward
        Output: (4, 4)      x: right, y: up  , z: backward
        '''
        pose_bl = np.zeros((4, 4))
        pose_bl[3, 3] = 1
        # camera position remain the same
        pose_bl[:3, 3] = pose[:3, 3] 
        
        R_c2w = pose[:3, :3]
        transform = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ]) 
        R_c2w_bl = R_c2w @ transform
        pose_bl[:3, :3] = R_c2w_bl
        
        # More transform (align scene z-axis to blender z-axis)
        # blender_trans = np.array([
        #     [-1,  0,  0,  0],
        #     [0,  0,  -1,  0],
        #     [0,  -1, 0,  0],
        #     [0,  0,  0,  1]
        # ])
        # pose_bl = blender_trans @ pose_bl
        
        return pose_bl
        
    def render_path_rgb(self, cam_list, dir_name='rgb'):
        dir_path = os.path.join(self.out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        num = len(cam_list)
        for i in range(num):
            cam_info = cam_list[i]
            self.set_camera(cam_info['K'], cam_info['c2w'])
            img_path = os.path.join(dir_path, '{:0>3d}.png'.format(i))
            bpy.context.scene.render.filepath = img_path
            bpy.ops.render.render(use_viewport=True, write_still=True)
            
    def initialize_depth_extractor(self):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.view_layer.cycles.denoising_store_passes = True
        bpy.context.scene.use_nodes = True
        node_tree = bpy.data.scenes["Scene"].node_tree
        render_layers = node_tree.nodes['Render Layers']
        node_tree.nodes.new(type="CompositorNodeOutputFile")
        file_output = node_tree.nodes['File Output']
        file_output.format.file_format = 'OPEN_EXR'
        links = node_tree.links
        new_link = links.new(render_layers.outputs[2], file_output.inputs[0])
    
    def render_path_depth(self, cam_list, dir_name='depth'):
        self.initialize_depth_extractor()
        dir_path = os.path.join(self.out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        num = len(cam_list)
        for i in range(num):
            cam_info = cam_list[i]
            self.set_camera(cam_info['K'], cam_info['c2w'])
            bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path = os.path.join(dir_path, '{:0>3d}'.format(i))
            bpy.ops.render.render(use_viewport=True, write_still=True)


def setup_blender_env(img_width, img_height, env_map_path):

    reset_scene()

    # Set render engine and parameters
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = img_width
    render.resolution_y = img_height
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 32  # 32 for testing, 256 or higher for final
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    preferences = context.preferences
    preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    preferences.addons["cycles"].preferences.get_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    add_env_lighting(os.path.join('/home/max/Desktop/instant-ngp-pp/', env_map_path), 1.0)
    add_sun_lighting()

    # TODO: figure out why AttributeError: 'WorldLighting' object has no attribute 'use_ambient_occlusion'
    # scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
    # scene.world.light_settings.ao_factor = 0.2  # set it to 0.5


def add_env_lighting(env_map_path: str, strength: float = 1.0):
    """
    Add environment lighting to the scene with controllable strength.

    Args:
        env_map_path (str): Path to the environment map.
        strength (float): Strength of the environment map.
    """
    # Ensure that we are using nodes for the world's material
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create an environment texture node and load the image
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(env_map_path)

    # Create a Background node and set its strength
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength

    # Create an Output node
    out = nodes.new('ShaderNodeOutputWorld')

    # Link nodes together
    links = world.node_tree.links
    links.new(env.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], out.inputs['Surface'])


# This is used for auxiliary lighting (for debugging)
def add_sun_lighting() -> None:
    # Check and delete any existing light object
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
            bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="SUN")
    light2 = bpy.data.lights["Sun"]
    light2.energy = 3
    bpy.data.objects["Sun"].location[2] = 2.0
    bpy.data.objects["Sun"].scale[0] = 100
    bpy.data.objects["Sun"].scale[1] = 100
    bpy.data.objects["Sun"].scale[2] = 100


def create_camera_list(c2w, K):
    """
    Create a list of camera parameters

    Args:
        c2w: (N, 4, 4) camera to world transform
        K: (3, 3) or (N, 3, 3) camera intrinsic matrix
    """
    cam_list = []
    for i in range(len(c2w)):
        pose = c2w[i].reshape(-1, 4)
        if len(K.shape) == 3:
            cam_list.append({'c2w': pose, 'K': K[i]})
        else:
            cam_list.append({'c2w': pose, 'K': K})
    return cam_list


def transform_object_origin(obj):
    """
    Transform object to align with the scene, make the bottom point of the object to be the origin

    Args:
        obj: blender objectå
    """
    bbox_min, bbox_max = scene_bbox(obj)

    new_origin = np.zeros(3)
    new_origin[0] = (bbox_max[0] + bbox_min[0]) / 2.
    new_origin[1] = (bbox_max[1] + bbox_min[1]) / 2.
    new_origin[2] = bbox_min[2]

    all_object_nodes = [obj] + obj.children_recursive

    ## move the asset origin to the bottom point
    for obj_node in all_object_nodes:
        if obj_node.data:
            me = obj_node.data
            mw = obj_node.matrix_world
            matrix = obj_node.matrix_world
            o = Vector(new_origin)
            o = matrix.inverted() @ o
            me.transform(Matrix.Translation(-o))
            mw.translation = mw @ o

    ## move all transform to origin
    for obj_node in all_object_nodes:
        obj_node.matrix_world.translation = [0, 0, 0]
        obj_node.rotation_quaternion = [1, 0, 0, 0]


def insert_object(obj_path, pos, rot, scale=1.0):
    """
    Insert object into the scene

    Args:
        obj_path: path to the object
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale of the object

    Returns:
        inserted_obj: blender object
    """
    inserted_obj = load_object(obj_path)
    # inserted_obj = bpy.context.object  # get the last inserted object (if single object)
    normalize_scene(inserted_obj)
    transform_object_origin(inserted_obj)
    inserted_obj.location = pos
    inserted_obj.scale *= scale
    # bpy.ops.object.transform_apply(scale=True)
    rotate_obj(inserted_obj, rot)
    # bpy.context.view_layer.update()                 # Update the scene
    return inserted_obj


def add_shadow_catcher(pos, rot, scale=1.0, option='plane', results_dir=None):
    """
    Add shadow catcher to the scene

    Args:
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale of the object
        option: 'plane' or 'mesh', which type of shadow catcher to add
        results_dir: path to the results directory
    """
    if option == 'plane':
        # add a plane as shadow catcher
        bpy.ops.mesh.primitive_plane_add()
        plane = bpy.context.object
        plane.location = pos
        plane.scale *= scale * 2.0   # 2.0 to make the plane size larger
        rotate_obj(plane, rot)
        plane.is_shadow_catcher = True
        plane.visible_glossy = False
        plane.visible_diffuse = False
    elif option == 'mesh':
        # add meshes extracted from NeRF as shadow catcher
        mesh_path = os.path.join(results_dir, 'meshes.ply')
        if not os.path.exists(mesh_path):
            AssertionError('meshes.ply does not exist')
        bpy.ops.import_mesh.ply(filepath=mesh_path)
        mesh = bpy.context.object
        mesh.is_shadow_catcher = True
        mesh.visible_glossy = False
        mesh.visible_diffuse = False

config_path = '/home/max/Desktop/instant-ngp-pp/results/lerf/teatime/blender_cfg.json'
with open(config_path, 'r') as f:
    config = json.load(f)

results_dir = config['results_dir']
h, w = config['im_height'], config['im_width']
K = np.array(config['K'])
c2w = np.array(config['c2w'])
env_map_path = config['env_map_path']
scene_up_vector = np.array(config['up_vector'])
insert_object_info = config['insert_object_info']

output_dir = os.path.join(results_dir, 'blend_results')
os.makedirs(output_dir, exist_ok=True)

setup_blender_env(w, h, env_map_path)
# align_R = get_alignment_rot(np.array([0, 0, 1]), scene_up_vector)  # No need to align since current camera poses are already aligned

cam = Camera(h, w, output_dir)
cam_list = create_camera_list(c2w, K)

# insert objects
for obj_info in insert_object_info:
    obj_path = os.path.join('/home/max/Desktop/instant-ngp-pp/', obj_info['object_path'])
    pos = np.array(obj_info['pos'])
    rot = np.array(obj_info['rot'])
    # rot = align_R @ rot
    scale = obj_info['scale']
    _ = insert_object(obj_path, pos, rot, scale)

bpy.context.view_layer.update()     # Update the scene


# # render rgb and depth without a shadow catcher
# cam.render_path_rgb(cam_list, dir_name='rgb')
# cam.render_path_depth(cam_list, dir_name='depth')

# add shadow catcher
for obj_info in insert_object_info:
    pos = np.array(obj_info['pos'])
    rot = np.array(obj_info['rot'])
    # rot = align_R @ rot
    scale = obj_info['scale']
    add_shadow_catcher(pos, rot, scale, option='plane')

bpy.context.view_layer.update()     # Update the scene

# # render rgb and depth with a shadow catcher
# cam.render_path_rgb(cam_list, dir_name='rgb_shadow')
# cam.render_path_depth(cam_list, dir_name='depth_shadow')
