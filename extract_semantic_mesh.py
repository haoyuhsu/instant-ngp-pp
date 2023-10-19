import os
import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import plyfile
import skimage.measure
from models.networks import NGP
from opt import get_opts
from utils import load_ckpt
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from tqdm import trange
from models.custom_functions import RayAABBIntersector
import math
from einops import rearrange, repeat, reduce
import vren
import json

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01

###################################
##### Adapt from rendering.py #####
###################################
def volume_render(
    model, 
    rays_o,
    rays_d,
    hits_t,
    # Image properties to be updated
    opacity,
    depth,
    rgb,
    normal_pred,
    normal_raw,
    sem,
    # Other parameters
    **kwargs
):
    N_rays = len(rays_o)
    device = rays_o.device
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']

    classes = kwargs.get('num_classes', 7)
    samples = 0
    total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    xyzs_list = []
    sigmas_list = []
    deltas_list = []
    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t, alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        ## Shapes
        # xyzs: (N_alive*N_samples, 3)
        # dirs: (N_alive*N_samples, 3)
        # deltas: (N_alive, N_samples) intervals between samples (with previous ones)
        # ts: (N_alive, N_samples) ray length for each samples
        # N_eff_samples: (N_alive) #samples along each ray <= N_smaples

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        normals_pred = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        sems = torch.zeros(len(xyzs), classes, device=device)
       
        _sigmas, _rgbs, _normals_pred, _normals_raw, _sems = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)

        sigmas[valid_mask] = _sigmas.detach().float()
        rgbs[valid_mask] = _rgbs.detach().float()
        normals_pred[valid_mask] = _normals_pred.float()
        normals_raw[valid_mask] = _normals_raw.float()
        sems[valid_mask] = _sems.float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        # print(sigmas.shape)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        xyzs = rearrange(xyzs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        # print("================================")
        # print("sigma size:", sigmas.shape)
        # print("delta shape:", deltas.shape)
        # print("xyzs shape:", xyzs.shape)
        # print("alive_indices shape:", alive_indices.shape)
        # concat sigma and delta, perform alpha compositing later on
        sigmas_full = torch.zeros(N_rays, N_samples, device=device)
        deltas_full = torch.zeros(N_rays, N_samples, device=device)
        xyzs_full = torch.zeros(N_rays, N_samples, 3, device=device)
        sigmas_full[alive_indices] = sigmas
        deltas_full[alive_indices] = deltas
        xyzs_full[alive_indices] = xyzs
        sigmas_list.append(sigmas_full)
        deltas_list.append(deltas_full)
        xyzs_list.append(xyzs_full)

        vren.composite_test_fw(
            sigmas, rgbs, normals_pred, normals_raw, sems, deltas, ts,
            hits_t, alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, normal_raw, sem)
        alive_indices = alive_indices[alive_indices>=0]
        # print(len(alive_indices))

    raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)

    results = {}
    results['xyzs'] = torch.cat(xyzs_list, dim=1)
    sigmas = torch.cat(sigmas_list, dim=1)
    deltas = torch.cat(deltas_list, dim=1)
    alphas = raw2alpha(sigmas, deltas)
    results['weights'] = alphas * torch.cumprod(torch.cat([torch.ones((alphas.shape[0], 1)).cuda(), 1.-alphas + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]
    # print("================================")
    # print("xyzs shape:", results['xyzs'].shape)
    # print("sigma size:", sigmas.shape)
    # print("delta shape:", deltas.shape)
    # print("alphas shape:", 'alphas.shape)
    # print("weights shape:", results['weights'].shape)

    if kwargs.get('use_skybox', False):
        rgb_bg = model.forward_skybox(rays_d)
        rgb += rgb_bg*rearrange(1 - opacity, 'n -> n 1')
    else: # real
        rgb_bg = torch.zeros(3, device=device)
        rgb += rgb_bg*rearrange(1 - opacity, 'n -> n 1')

    return total_samples, results

###################################
##### Adapt from rendering.py #####
###################################
@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Input:
        rays_o: [h*w, 3] rays origin
        rays_d: [h*w, 3] rays direction

    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    hits_t = hits_t[:,0,:]
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    classes = kwargs.get('num_classes', 7)
    # output tensors to be filled in
    N_rays = len(rays_o) # h*w
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    normal_pred = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)
    sem = torch.zeros(N_rays, classes, device=device)
    mask = torch.zeros(N_rays, device=device)
    # store weight values for each sample along each ray
    # weight = torch.zeros(N_rays, MAX_SAMPLES, device=device)

    # print("Chunk size: {}".format(N_rays))
    
    # Perform volume rendering
    total_samples, results_pre_render = \
        volume_render(
            model, rays_o, rays_d, hits_t,
            opacity, depth, rgb, normal_pred, normal_raw, sem,
            **kwargs
        )
    
    results = {}
    results['opacity'] = opacity # (h*w)
    results['depth'] = depth # (h*w)
    results['rgb'] = rgb # (h*w, 3)
    results['normal_pred'] = normal_pred
    results['normal_raw'] = normal_raw
    results['semantic'] = torch.argmax(sem, dim=-1, keepdim=True)
    results['total_samples'] = total_samples # total samples for all rays
    results['points'] = rays_o + rays_d * depth.unsqueeze(-1)
    results['mask'] = mask

    results['weights'] = results_pre_render['weights']
    results['xyzs'] = results_pre_render['xyzs']

    # extend 'weights' and 'xyzs' second dimension to MAX_SAMPLES with zero padding
    # weights_pad = torch.zeros(N_rays, MAX_SAMPLES-results['weights'].shape[1], device=device)
    # xyzs_pad = torch.zeros(N_rays, MAX_SAMPLES-results['xyzs'].shape[1], 3, device=device)
    # results['weights'] = torch.cat([results['weights'], weights_pad], dim=1)
    # results['xyzs'] = torch.cat([results['xyzs'], xyzs_pad], dim=1)
    
    if exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=device)

    return results


def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = None

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)

    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


def render_chunks(model, rays_o, rays_d, chunk_size, **kwargs):
    chunk_n = math.ceil(rays_o.shape[0]/chunk_size)
    results = {}
    for i in range(chunk_n):
        rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
        rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
        ret = render(model, rays_o_chunk, rays_d_chunk, **kwargs)
        for k in ret:
            if k not in results:
                results[k] = []
            results[k].append(ret[k])
    for k in results:
        if k in ['total_samples', 'weights', 'xyzs']:
            continue
        results[k] = torch.cat(results[k], 0)
    ####################################################
    ##### Custom aggregation for weights and xyzs #####
    # weights = results['weights']
    # xyzs = results['xyzs']
    # max_n_samples = max([weight.shape[1] for weight in weights])  # find maximum number of samples along each ray
    # # extend 'weights' and 'xyzs' second dimension to MAX_SAMPLES with zero padding
    # for i in range(len(weights)):
    #     weights[i] = torch.cat([weights[i], torch.zeros(weights[i].shape[0], max_n_samples-weights[i].shape[1], device=weights[i].device)], dim=1)
    #     xyzs[i] = torch.cat([xyzs[i], torch.zeros(xyzs[i].shape[0], max_n_samples-xyzs[i].shape[1], 3, device=xyzs[i].device)], dim=1)
    # results['weights'] = torch.cat(weights, dim=0)
    # results['xyzs'] = torch.cat(xyzs, dim=0)
    ####################################################
    return results


def convert_samples_to_ply(
    pytorch_3d_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_tensor = pytorch_3d_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
    

def extract_semantic_meshes(hparams, split='test'):
    # if hparams.use_gsam_hq:
    #     semantic_mesh_dir = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}/semantic_mesh_hq')
    # else:
    #     semantic_mesh_dir = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}/semantic_mesh')
    semantic_mesh_dir = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}/semantic_mesh_deva')
    os.makedirs(semantic_mesh_dir, exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model = NGP(
        scale=hparams.scale,
        rgb_act=rgb_act, 
        use_skybox=hparams.use_skybox, 
        embed_a=hparams.embed_a, 
        embed_a_len=hparams.embed_a_len,
        classes=hparams.num_classes).cuda()
    if hparams.ckpt_load:
        ckpt_path = hparams.ckpt_load
    else: 
        ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'last_slim.ckpt')

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model'])
    print('Loaded checkpoint: {}'.format(ckpt_path))

    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'

    N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))

    embed_a_length = hparams.embed_a_len
    if hparams.embed_a:
        embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["model", "msk_model"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())    

    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_traj': hparams.render_traj,
            'anti_aliasing_factor': hparams.anti_aliasing_factor}
    
    dataset = dataset(split='test', **kwargs)
    w, h = dataset.img_wh
    if hparams.render_traj or hparams.render_train:
        render_traj_rays = dataset.render_traj_rays
    else:
        # render_traj_rays = dataset.rays
        render_traj_rays = {}
        print("generating rays' origins and directions!")
        for img_idx in trange(len(dataset.poses)):
            rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
            render_traj_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    ##### Use for tracking with DEVA #####
    semantic_dir = os.path.join(hparams.root_dir, "track_with_deva")
    with open(os.path.join(semantic_dir, "id_dict.json"), 'r') as f:
        id_dict = json.load(f)
    categories = id_dict['categories']
    ##### Use for Grounded-SAM #####
    # if hparams.use_gsam_hq:
    #     semantic_dir = os.path.join(hparams.root_dir, "grounded_sam_hq")
    # else:
    #     semantic_dir = os.path.join(hparams.root_dir, "grounded_sam")
    # with open(os.path.join(semantic_dir, "pred.json"), 'r') as f:
    #     id_dict = json.load(f)
    # categories = id_dict['categories']
    num_categories = len(categories)
    print("Number of categories:", num_categories)

    grid_dim = np.array([int(grid_) for grid_ in hparams.grid_dim.split(' ')])
    grid_dim = torch.tensor(grid_dim)
    min_bound = [float(min_) for min_ in hparams.min_bound.split(' ')]
    max_bound = [float(max_) for max_ in hparams.max_bound.split(' ')]
    
    # x_min, x_max = -1, 1
    # y_min, y_max = -0.3, 0.15
    # z_min, z_max = -1, 1
    x_min, x_max = min_bound[0], max_bound[0]
    y_min, y_max = min_bound[1], max_bound[1]
    z_min, z_max = min_bound[2], max_bound[2]
    xyz_min = torch.FloatTensor([[x_min, y_min, z_min]])
    xyz_max = torch.FloatTensor([[x_max, y_max, z_max]])
    bbox = torch.cat([xyz_min, xyz_max], dim=0)

    semantic_grid = torch.zeros((grid_dim[0], grid_dim[1], grid_dim[2], num_categories)).cuda()  # (x, y, z, c) # 0 for bg
    # counting the number of samples in each voxel
    semantic_grid_cnt = torch.zeros((grid_dim[0], grid_dim[1], grid_dim[2], num_categories), dtype=torch.int32).cuda()  # (x, y, z, c) # 0 for bg

    for img_idx in trange(len(render_traj_rays)):
        rays = render_traj_rays[img_idx][:, :6].cuda()
        render_kwargs = {
            'img_idx': img_idx,
            'test_time': True,
            'T_threshold': 1e-2,
            'use_skybox': hparams.use_skybox,
            'render_rgb': hparams.render_rgb,
            'render_depth': hparams.render_depth,
            'render_normal': hparams.render_normal,
            'render_sem': hparams.render_semantic,
            'num_classes': hparams.num_classes,
            'img_wh': dataset.img_wh,
            'anti_aliasing_factor': hparams.anti_aliasing_factor
        }
        if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a

        img_path = dataset.imgs[img_idx]
        file_name = img_path.split('/')[-1].split('.')[0]
        semantic_map_full = torch.from_numpy(np.load(os.path.join(semantic_dir, "semantic_cat", file_name+'.npy'))).long()
        semantic_map_full = rearrange(semantic_map_full, 'h w -> (h w)')

        chunk_size = 512 * 512

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        # results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)

        chunk_n = math.ceil(rays.shape[0]/chunk_size)

        for i in range(chunk_n):
                
            rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
            rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
            semantic_map = semantic_map_full[i*chunk_size: (i+1)*chunk_size]
            
            results = render(model, rays_o_chunk, rays_d_chunk, **render_kwargs)
        
            weights = results['weights']
            xyzs = results['xyzs']

            n_rays, n_samples = weights.shape

            # move to GPUs
            weights = weights.to(rays_o.device)
            xyzs = xyzs.to(rays_o.device)
            xyz_min = xyz_min.to(rays_o.device)
            xyz_max = xyz_max.to(rays_o.device)
            semantic_map = semantic_map.to(rays_o.device)
            grid_dim = grid_dim.to(rays_o.device)

            weights = rearrange(weights, 'n1 n2 -> (n1 n2)')
            xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
            # semantic_map = rearrange(semantic_map, 'h w -> (h w)')
            semantic_map = repeat(semantic_map, 'n1 -> (n1 repeat)', repeat=n_samples)

            assert weights.shape[0] == xyzs.shape[0] == semantic_map.shape[0]

            batch_size = 128 * 128 * 128
            batch_n = math.ceil(weights.shape[0]/batch_size)
            for i in range(batch_n):
                _xyzs = xyzs[i*batch_size: (i+1)*batch_size]
                _weights = weights[i*batch_size: (i+1)*batch_size]
                _semantic_map = semantic_map[i*batch_size: (i+1)*batch_size]
                mask = ((_xyzs >= xyz_min) & (_xyzs < xyz_max)).all(dim=-1)
                if mask.sum() == 0:
                    print("no points in bounds!")
                    continue
                x_idx, y_idx, z_idx = ((_xyzs[mask] - xyz_min) / (xyz_max - xyz_min) * (grid_dim-1)).long().to(rays_o.device).t()
                w_idx = _weights[mask]
                c_idx = _semantic_map[mask]
                semantic_grid[x_idx, y_idx, z_idx, c_idx] += w_idx
                semantic_grid_cnt[x_idx, y_idx, z_idx, c_idx] += 1

    # compute the average semantic values of each voxel
    # semantic_grid_cnt = semantic_grid_cnt.float()
    # mask = semantic_grid_cnt > 0
    # semantic_grid[mask] = semantic_grid[mask] / semantic_grid_cnt[mask]

    # print("max semantic values of each voxel:", torch.amax(semantic_grid, dim=(0,1,2)))
    # print("min semantic values of each voxel:", torch.amin(semantic_grid, dim=(0,1,2)))
    # print("max semantic counts of each voxel:", torch.amax(semantic_grid_cnt, dim=(0,1,2)))
    # print("min semantic counts of each voxel:", torch.amin(semantic_grid_cnt, dim=(0,1,2)))

    ##### compute density values of each voxel (currently not used) #####
    # dense_xyz = torch.stack(torch.meshgrid(
    #     torch.linspace(x_min, x_max, 512),
    #     torch.linspace(y_min, y_max, 128),
    #     torch.linspace(z_min, z_max, 512),
    # ), -1).cuda()
    # samples = dense_xyz.reshape(-1, 3)
    # density = []
    # with torch.no_grad():
    #     for i in range(0, samples.shape[0], chunk_size):
    #         samples_ = samples[i:i+chunk_size]
    #         tmp = model.density(samples_)
    #         density.append(tmp)
    # density = torch.stack(density, dim=0)
    # density = density.reshape((dense_xyz.shape[0], dense_xyz.shape[1], dense_xyz.shape[2]))
    # convert_samples_to_ply(
    #     density.cpu(), 
    #     os.path.join(semantic_mesh_dir, '{}.ply'.format("full_density")), 
    #     bbox=bbox.cpu(), 
    #     level=10
    # )

    # Save raw semantic grid
    print("Saving semantic grid to npy file!")
    np.save(os.path.join(semantic_mesh_dir, 'semantic_grid.npy'), semantic_grid.cpu().numpy())

    # Remove background voxels if they are occupied by other categories
    threshold = 0.2
    non_bg_mask = (semantic_grid[:, :, :, 1:] > threshold).any(dim=-1)
    semantic_grid[non_bg_mask][:, 0] = 0
    # Determine the semantic category of each voxel
    semantic_labels = torch.argmax(semantic_grid, dim=-1)

    print("Saving semantic grid to ply file!")
    # Extract meshes of each semantic category
    for i in range(num_categories):
        semantic_grid_i = semantic_grid[:, :, :, i]
        # Filter out voxels that are not within the semantic category
        semantic_grid_i[semantic_labels != i] = 0
        # None of the voxels are occupied by this category
        if torch.max(semantic_grid_i) <= 1e-3:
            continue
        # category_name = categories[i-1] if i > 0 else 'bg'
        category_name = categories[i]
        print("Category:", category_name)
        # import ipdb; ipdb.set_trace()
        convert_samples_to_ply(
            semantic_grid_i.cpu(), 
            os.path.join(semantic_mesh_dir, '{}.ply'.format(category_name)), 
            bbox=bbox.cpu(),
            level=0.2
        )


def estimate_range_of_scene(hparams, split='test'):
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model = NGP(
        scale=hparams.scale,
        rgb_act=rgb_act, 
        use_skybox=hparams.use_skybox, 
        embed_a=hparams.embed_a, 
        embed_a_len=hparams.embed_a_len,
        classes=hparams.num_classes).cuda()
    if hparams.ckpt_load:
        ckpt_path = hparams.ckpt_load
    else: 
        ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'last_slim.ckpt')

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model'])
    print('Loaded checkpoint: {}'.format(ckpt_path))

    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'

    N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))

    embed_a_length = hparams.embed_a_len
    if hparams.embed_a:
        embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["model", "msk_model"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())    

    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_traj': hparams.render_traj,
            'anti_aliasing_factor': hparams.anti_aliasing_factor}
    
    dataset = dataset(split='test', **kwargs)
    w, h = dataset.img_wh
    if hparams.render_traj or hparams.render_train:
        render_traj_rays = dataset.render_traj_rays
    else:
        # render_traj_rays = dataset.rays
        render_traj_rays = {}
        print("generating rays' origins and directions!")
        for img_idx in trange(len(dataset.poses)):
            rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
            render_traj_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    # compute the min and max bounds of camera positions
    poses = dataset.poses
    camera_pos = torch.stack([pose[:, 3] for pose in poses])
    camera_pos_min = torch.min(camera_pos, dim=0)[0]
    camera_pos_max = torch.max(camera_pos, dim=0)[0]

    x_min = y_min = z_min = np.inf
    x_max = y_max = z_max = -np.inf

    for img_idx in trange(len(render_traj_rays)):
        rays = render_traj_rays[img_idx][:, :6].cuda()
        render_kwargs = {
            'img_idx': img_idx,
            'test_time': True,
            'T_threshold': 1e-2,
            'use_skybox': hparams.use_skybox,
            'render_rgb': hparams.render_rgb,
            'render_depth': hparams.render_depth,
            'render_normal': hparams.render_normal,
            'render_sem': hparams.render_semantic,
            'num_classes': hparams.num_classes,
            'img_wh': dataset.img_wh,
            'anti_aliasing_factor': hparams.anti_aliasing_factor
        }
        if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a

        chunk_size = 512 * 512
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        # results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)

        chunk_n = math.ceil(rays.shape[0]/chunk_size)
        for i in range(chunk_n):
            rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
            rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
            results = render(model, rays_o_chunk, rays_d_chunk, **render_kwargs)
            depth = results['depth']
            points_3d = rays_o_chunk + rays_d_chunk * depth.unsqueeze(-1)
            x_min = min(x_min, torch.min(points_3d[:, 0]))
            x_max = max(x_max, torch.max(points_3d[:, 0]))
            y_min = min(y_min, torch.min(points_3d[:, 1]))
            y_max = max(y_max, torch.max(points_3d[:, 1]))
            z_min = min(z_min, torch.min(points_3d[:, 2]))
            z_max = max(z_max, torch.max(points_3d[:, 2]))

    print(hparams.dataset_name, hparams.exp_name)
    print("x_min: {}, x_max: {}".format(x_min, x_max))
    print("y_min: {}, y_max: {}".format(y_min, y_max))
    print("z_min: {}, z_max: {}".format(z_min, z_max))

    # save x_min, x_max, y_min, y_max, z_min, z_max to txt file
    with open(os.path.join('results', hparams.dataset_name, hparams.exp_name, 'range.txt'), 'w') as f:
        f.write("================ Scene Range ==============\n")
        f.write("x_min: {}, x_max: {}\n".format(x_min, x_max))
        f.write("y_min: {}, y_max: {}\n".format(y_min, y_max))
        f.write("z_min: {}, z_max: {}\n".format(z_min, z_max))
        f.write("================ Camera Range =============\n")
        f.write("x_min: {}, x_max: {}\n".format(camera_pos_min[0], camera_pos_max[0]))
        f.write("y_min: {}, y_max: {}\n".format(camera_pos_min[1], camera_pos_max[1]))
        f.write("z_min: {}, z_max: {}\n".format(camera_pos_min[2], camera_pos_max[2]))


   
if __name__ == '__main__':
    hparams = get_opts()
    extract_semantic_meshes(hparams)  # Extract semantic meshes
    # estimate_range_of_scene(hparams)    # Estimate the min & max bounds of the scene