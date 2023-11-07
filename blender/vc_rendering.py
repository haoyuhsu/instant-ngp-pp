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

def apply_rot_mat_to_obj(obj, R):
    """
    Apply rotation matrix to blender object

    Args:
        obj: blender object
        R: (3, 3) rotation matrix
    """
    R = Matrix(R)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = R.to_quaternion()

def sort_key(x):
    if len(x) > 2 and x[-10] == "_":
        return x[-9:]
    return x

class Camera():
    def __init__(self, im_height, im_width, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.w = im_width
        self.h = im_height
        bpy.data.scenes['Scene'].render.resolution_x = self.w
        bpy.data.scenes['Scene'].render.resolution_y = self.h
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        bpy.context.scene.camera = cam
        self.camera = cam
        
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

def setup_blender_env():

    # delete everything within the scene
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        # bpy.context.view_layer.objects.active = obj
        bpy.ops.object.delete()

    # maybe default setting is 'RGBA'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # use blender cycles
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = True
    bpy.data.scenes["Scene"].cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    bpy.context.scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
    bpy.context.scene.world.light_settings.ao_factor = 0.2  # set it to 0.5

    # nodes
    # bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    # bpy.context.view_layer.cycles.use_denoising = True
    # bpy.context.view_layer.cycles.denoising_store_passes = True
    # bpy.context.scene.use_nodes = True
    # tree = bpy.context.scene.node_tree
    # nodes = tree.nodes

    # create render layers
    # node_rl = nodes['Render Layers']
    # node_out_img = nodes.new('CompositorNodeOutputFile')
    # node_out_img.base_path = output_folder
    # node_out_img.format.file_format = 'PNG'
    # node_out_img.format.compression = 100
    # tree.links.new(node_rl.outputs['Image'], node_out_img.inputs['Image'])

def add_env_lighting(env_map_path):
    """
    Add environment lighting to the scene

    Args:
        env_map_path: path to the environment map
    """
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    nodes.clear()
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(env_map_path)
    out = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(env.outputs['Color'], out.inputs['Surface'])

def get_alignment_rot(v1, v2):
    """
    Get rotation matrix to align v1 to v2

    Args:
        v1: (3,) asset up vector (default)
        v2: (3,) world scene up vector

    Returns:
        R: (3, 3) rotation matrix
    """
    R = get_rot_mat(v1, v2)
    return Matrix(R)

def create_camera_list(c2w, K):
    cam_list = []
    for i in range(len(c2w)):
        pose = c2w[i].reshape(-1, 4)
        if len(K.shape) == 3:
            cam_list.append({'c2w': pose, 'K': K[i]})
        else:
            cam_list.append({'c2w': pose, 'K': K})
    return cam_list

def transform_object_origin(obj, use_verts=True):
    """
    Transform object to align with the scene, make the bottom point of the object to be the origin

    Args:
        obj: blender object
        use_verts: whether to use vertices or bounding box to calculate the bottom point
    """
    all_object_nodes = [obj] + obj.children_recursive
    # get the bottom point of all components in an asset
    vert_list = []
    for obj_node in all_object_nodes:
        if obj_node.data:
            me = obj_node.data
            matrix = obj_node.matrix_world
            if use_verts:
                data = (v.co for v in me.vertices)
            else:
                data = (Vector(v) for v in obj_node.bound_box)
            coords = np.array([matrix @ v for v in data])
            vert_list.append(coords)

    all_vertices = np.concatenate(vert_list, axis=0)
    x = all_vertices.T[0]
    y = all_vertices.T[1]
    z = all_vertices.T[2]

    new_origin = np.zeros(3)
    new_origin[0] = (x.max() + x.min()) / 2.
    new_origin[1] = (y.max() + y.min()) / 2.
    new_origin[2] = z.min()

    # move the asset origin to the bottom point
    for obj_node in all_object_nodes:
        if obj_node.data:
            me = obj_node.data
            mw = obj_node.matrix_world
            matrix = obj_node.matrix_world
            o = Vector(new_origin)
            o = matrix.inverted() @ o
            me.transform(Matrix.Translation(-o))
            mw.translation = mw @ o

    # move all transform to origin
    for obj_node in all_object_nodes:
        obj_node.matrix_world.translation = [0, 0, 0]
        obj_node.rotation_quaternion = [1, 0, 0, 0]


def insert_object(obj_path, pos, rot, scale=0.03):
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
    bpy.ops.import_scene.gltf(filepath=obj_path)
    inserted_obj = bpy.context.object
    transform_object_origin(inserted_obj, use_verts=True)
    inserted_obj.location = pos
    inserted_obj.scale = scale * np.array([1, 1, 1])
    apply_rot_mat_to_obj(inserted_obj, rot)
    return inserted_obj

def add_shadow_catcher(pos, rot, scale=0.03, option='plane', results_dir=None):
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
        plane.scale = scale * 0.5 * np.array([1, 1, 1])
        apply_rot_mat_to_obj(plane, rot)
        plane.is_shadow_catcher = True
        plane.visible_glossy = False
        plane.visible_diffuse = False
    elif option == 'mesh':
        # add meshes extracted from NeRF as shadow catcher
        meshes_folder = os.path.join(results_dir, 'semantic_mesh_deva')
        meshes_filenames = [filename for filename in sorted(os.listdir(meshes_folder)) if filename.endswith('.ply')]
        for mesh_file in meshes_filenames:
            mesh_path = os.path.join(meshes_folder, mesh_file)
            bpy.ops.import_mesh.ply(filepath=mesh_path)
            mesh = bpy.context.object
            mesh.is_shadow_catcher = True
            mesh.visible_glossy = False
            mesh.visible_diffuse = False

def run_blender_render(config_path):
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

    setup_blender_env()
    add_env_lighting(env_map_path)
    align_R = get_alignment_rot(np.array([0, 0, 1]), scene_up_vector)

    # insert objects
    for obj_info in insert_object_info:
        obj_path = obj_info['object_path']
        pos = np.array(obj_info['pos'])
        rot = np.array(obj_info['rot'])
        rot = align_R @ rot
        scale = obj_info['scale']
        _ = insert_object(obj_path, pos, rot, scale)

    cam = Camera(h, w, output_dir)
    cam_list = create_camera_list(c2w, K)

    # render rgb and depth without a shadow catcher
    cam.render_path_rgb(cam_list, dir_name='rgb')
    cam.render_path_depth(cam_list, dir_name='depth')

    # add shadow catcher
    for obj_info in insert_object_info:
        pos = np.array(obj_info['pos'])
        rot = np.array(obj_info['rot'])
        rot = align_R @ rot
        scale = obj_info['scale']
        add_shadow_catcher(pos, rot, scale, option='plane')
        # add_shadow_catcher(pos, rot, scale, option='mesh', results_dir=results_dir)

    # render rgb and depth with a shadow catcher
    cam.render_path_rgb(cam_list, dir_name='rgb_shadow')
    cam.render_path_depth(cam_list, dir_name='depth_shadow')

def run_blender_render_terminal(blender_exec_path, config_path):
    os.system('export blender={} --background --python vc_rendering.py -- --input_config_path {}'.format(blender_exec_path, config_path))

if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--input_config_path', type=str, default='')
    args = parser.parse_args()
    run_blender_render(args.input_config_path)