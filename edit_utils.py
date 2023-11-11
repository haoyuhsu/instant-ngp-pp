import os
import open3d as o3d
import numpy as np
import math
import random
from retrieval.wrapper_objaverse import retrieve_object_from_objaverse

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
        ##### TODO: integrate Tracking-with-DEVA & extract_semantic_mesh.py #####
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
    obj_id = list(obj_info.keys())[0]   # get the id of the obj file
    obj_path = obj_info[obj_id]         # get the path of the obj file
    new_obj_info = {}
    new_obj_info['object_name'] = object_name
    new_obj_info['object_id'] = obj_id
    new_obj_info['object_path'] = obj_path
    # new_obj_info['object_path'] = '/home/max/Desktop/Blender_Stuff/insertion/assets/apple.glb'  # for debugging
    return new_obj_info

def put_object_in_scene(scene_representation, object_info, object_locations):
    print("Inserting object: {}".format(object_info['object_name']))
    assert isinstance(object_info, dict)
    selected_positions = object_locations[random.randint(0, len(object_locations)-1)]
    # simply store the location and orientation of the object in the scene representation
    object_info['pos'] = selected_positions
    object_info['rot'] = np.eye(3)
    object_info['scale'] = 0.017
    scene_representation.insert_object(object_info)

def change_object_texture(obj, texture_name):
    print("Texturing object {} into {}".format(obj, texture_name))
    return obj