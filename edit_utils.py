import os
import open3d as o3d
import numpy as np
import math
import random
from retrieval.wrapper_objaverse import retrieve_object_from_objaverse
from tracking.demo_with_text import run_deva
from extract_semantic_mesh import extract_semantic_meshes
import glob
from PIL import Image
from gpt.estimate_scale import estimate_object_scale

'''
Wrapper of the modular functions for GPT model to call
'''

def get_object_3d_location(scene_representation, object_name, N_SAMPLES=2000):
    print("Localizing object: {}".format(object_name))
    obj_mesh_path = None
    # find out if there exists extracted mesh
    if os.path.exists(os.path.join(scene_representation.semantic_mesh_dir, object_name + '.ply')):
        print("Found extracted mesh of {}.".format(object_name))
        obj_mesh_path = os.path.join(scene_representation.semantic_mesh_dir, object_name + '.ply')
    else:
        # otherwise, segment image and extract mesh
        print("Extracting mesh of {} from input views......".format(object_name))
        # integrate Tracking-with-DEVA & extract_semantic_mesh.py
        object_tracking_results_dir = os.path.join(scene_representation.tracking_results_dir, object_name)
        os.makedirs(object_tracking_results_dir, exist_ok=True)
        run_deva(scene_representation.dataset.imgs_dir, object_tracking_results_dir, object_name)
        obj_mesh_path = extract_semantic_meshes(scene_representation, object_tracking_results_dir)
        if obj_mesh_path is None:
            raise ValueError("Mesh extraction failed for object {}.".format(object_name))
        print("Extracted mesh of {} saved at {}.".format(object_name, obj_mesh_path))
    # load meshes
    mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    mesh.compute_vertex_normals()
    # randomly sample points on the mesh surface if the mesh is facing upwards
    upwards = scene_representation.up_vector
    vertices = np.asarray(mesh.vertices)[np.dot(np.array(mesh.vertex_normals), upwards) > math.cos(math.radians(30))]
    sampled_vertices = vertices[np.random.choice(vertices.shape[0], N_SAMPLES, replace=False)]
    return sampled_vertices

def get_3d_asset(object_name):
    obj_info = retrieve_object_from_objaverse(object_name)
    if obj_info is None:
        print("Making one with generative model.")
        ##### TODO: use Fantasia3D or project from Adobe to generate object #####
    else:
        print("Retrieved object {} from objaverse.".format(object_name))

    print("Rendering object {}......".format(obj_info['object_path']))

    # Render the object in Blender
    os.system('{} --background --python ./blender/asset_rendering.py -- --object_file={} --output_dir={}'.format( \
        '/snap/bin/blender', \
        obj_info['object_path'], \
        './blender/images'
    ))

    img_path_list = sorted(glob.glob(os.path.join('./blender/images/', obj_info['object_id'], '*.png')))
    img_path = np.random.choice(img_path_list)

    # Estimate the scale of the object by GPT4 API
    object_scale = estimate_object_scale(img_path, obj_info['object_name'])  # use both rendered image and object name
    # object_scale = estimate_object_scale(None, obj_info['object_name'])        # use only object name
    # object_scale = estimate_object_scale(img_path, None)                  # use only rendered image
    obj_info['object_scale'] = object_scale
    print("Estimated scale of {} is {} meters.".format(obj_info['object_name'], object_scale))

    return obj_info

def put_object_in_scene(scene_representation, object_info, object_locations):
    print("Inserting object: {}".format(object_info['object_name']))
    assert isinstance(object_info, dict)
    selected_positions = object_locations[random.randint(0, len(object_locations)-1)]
    # simply store the location and orientation of the object in the scene representation
    new_object_info = object_info.copy()
    new_object_info['pos'] = selected_positions
    new_object_info['rot'] = np.eye(3)
    new_object_info['scale'] = object_info['object_scale'] / scene_representation.scene_scale
    scene_representation.insert_object(new_object_info)
    print("Inserted object {} into scene at {}.".format(new_object_info['object_name'], selected_positions))

def change_object_texture(obj, texture_name):
    print("Texturing object {} into {}".format(obj, texture_name))
    return obj

if __name__ == '__main__':
    from scene_representation import SceneRepresentation
    from opt import get_opts
    hparams = get_opts()
    scene_representation = SceneRepresentation(hparams)
    loc = get_object_3d_location(scene_representation, 'sand')
    obj = get_3d_asset('apple')
    put_object_in_scene(scene_representation, obj, loc)
    # render result
    scene_representation.render_scene(skip_render_NeRF=True)