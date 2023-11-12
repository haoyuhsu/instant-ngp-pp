import os
import torch
import torch.nn.functional as F
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import numpy as np
from tqdm import trange
from models.networks import NGP
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from opt import get_opts
from einops import rearrange
from kornia import create_meshgrid
from utils import guided_filter
import math

"""
x axis: v_right
y axis: v_forward
z axis: v_up
"""

def sample_panorama(
    directions, 
    panorama,
    v_forward, 
    v_up,
    v_right
):  
    '''
    Retirve panorama values according to dirction
    follows the spherical cooridnates, 
    axis (x, y, z) = (v_right, v_forward, v_up)
    Input:
        dirction: (n, 3)
        panorama: (h, w, c)
        v_forward, v_up, v_right: (3, )
    Return:
        samples: (n, c)
    '''
    directions = F.normalize(directions, dim=-1, eps=1e-9)
    basis = torch.stack([v_right, v_forward, v_up]).to(directions.device)
    new_coords = torch.matmul(directions, basis.T) # (n, 3)
    new_x, new_y, new_z = new_coords.unbind(-1)
    tan_theta = new_y / new_x 
    thetas = torch.arctan(tan_theta)
    thetas[torch.logical_and(new_x<0, new_y>0)] += torch.pi
    thetas[torch.logical_and(new_x<0, new_y<0)] -= torch.pi
    phis = torch.arcsin(new_z)
    
    u = thetas/torch.pi # in range (-1, 1)
    v = phis*2/torch.pi # in range (-1, 1)
    grid = torch.stack([u, v], dim=-1) #(n, 2)
    grid = grid[None, None] #(1, 1, n, 2)
    panorama = panorama.permute(2, 0, 1)[None] #(1, c, h, w)
    samples = F.grid_sample(
        panorama, 
        grid,
        mode='bilinear',
        align_corners=True
    ) # (1, c, 1, n)
    samples = samples.permute(0, 2, 3, 1)[0, 0] #(n, c)
    return samples

def render_panorama(hparams):
    dir_out = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}')
    os.makedirs(dir_out, exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    if hparams.ckpt_load:
        ckpt_path = hparams.ckpt_load
    else: 
        ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'last_slim.ckpt')

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net', 'directions', 'density_grid', 'grid_coords'])
    print('Loaded checkpoint: {}'.format(ckpt_path))

    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'

    if hparams.dataset_name=='tnt':
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        embed_a_length = hparams.embed_a_len
        if hparams.embed_a:
            embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params", "normal_net.params"])
            embedding_a = embedding_a(torch.tensor([0]).cuda())

    # the 3 following vectors depend on dataset
    # hparams.pano_hw = (512, 1024)
    hparams.pano_hw = (1024, 2048)
    hparams.v_right = np.array([1, 0, 0])
    hparams.v_forward = np.array([0, 1, 0])
    hparams.v_down = np.array([0, 0, -1])
    hparams.pano_radius = 0.5

    H, W = hparams.pano_hw
    cx = W/2
    cy = H/2

    device = 'cuda'
    origin = torch.zeros(3).to(device)

    right = torch.FloatTensor(hparams.v_right).to(device)
    forward = torch.FloatTensor(hparams.v_forward).to(device)
    down = torch.FloatTensor(hparams.v_down).to(device)
    
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)
    
    thetas = ((u-cx+0.5)*2*torch.pi/W).reshape(-1, 1) # longitude (-pi, pi), angle from forward direction
    phis = ((v-cy+0.5)*torch.pi/H).reshape(-1, 1) # latitude (-pi/2, pi/2), angle from 
    directions = torch.sin(phis)*down.unsqueeze(0) + torch.cos(phis)*torch.sin(thetas)*right.unsqueeze(0) + torch.cos(phis)*torch.cos(thetas)*forward.unsqueeze(0)
    directions = F.normalize(directions, p=2, dim=-1, eps=1e-9)
    rays_d = directions.reshape(-1, 3).cuda()
    rays_o = origin.repeat((rays_d.shape[0], 1)).cuda()
    rays_o += rays_d * hparams.pano_radius
    render_kwargs = {'test_time': True,
                    'T_threshold': 1e-2,
                    'use_skybox': hparams.use_skybox,
                    'render_rgb': hparams.render_rgb,
                    'render_depth': hparams.render_depth,
                    'img_wh': (W, H)}
    if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a
    
    assert rays_o.shape[0] == rays_d.shape[0]

    # batchify rays_o and rays_d to avoid OOM
    chunk_size = 512 * 512
    chunk_n = math.ceil(rays_o.shape[0] / chunk_size)
    rgb_list = []
    opacity_list = []
    for i in range(chunk_n):
        rays_o_chunk = rays_o[i*chunk_size:(i+1)*chunk_size]
        rays_d_chunk = rays_d[i*chunk_size:(i+1)*chunk_size]
        results = render(model, rays_o_chunk, rays_d_chunk, **render_kwargs)
        rgb_list.append(results['rgb'].cpu().numpy())
        opacity_list.append(results['opacity'].cpu().numpy())
    rgb = np.concatenate(rgb_list, axis=0)
    opacity = np.concatenate(opacity_list, axis=0)

    # results = render(model, rays_o, rays_d, **render_kwargs)
    
    rgb = rearrange(rgb, '(h w) c -> h w c', h=H)
    # rgb = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=H)
    rgb = (rgb*255).astype(np.uint8)
    dir_rgb = os.path.join(dir_out, 'panorama/rgb')
    os.makedirs(dir_rgb, exist_ok=True)
    imageio.imsave(os.path.join(dir_rgb, '0.png'), rgb)

    opacity = rearrange(opacity, '(h w) -> h w', h=H)
    opacity = (opacity*255).astype(np.uint8)
    # opacity = rearrange(results['opacity'], '(h w) -> h w', h=H)
    # opacity = guided_filter(opacity, opacity, r=10, eps=0.5)
    # opacity = (opacity*255).cpu().numpy().astype(np.uint8)
    dir_opacity = os.path.join(dir_out, 'panorama/opacity')
    os.makedirs(dir_opacity, exist_ok=True)
    imageio.imsave(os.path.join(dir_opacity, '0.png'), opacity)

    inpaint = opacity < 0.5
    mask = np.zeros_like(opacity)
    mask[inpaint] = 255
    dir_mask = os.path.join(dir_out, 'panorama/mask')
    os.makedirs(dir_mask, exist_ok=True)
    imageio.imsave(os.path.join(dir_mask, '0.png'), mask)

    # validate sample_panorama
    # rgb = torch.FloatTensor(rgb).to(device) / 255
    # samples = sample_panorama(directions, rgb, forward, up, right)
    # samples = rearrange(samples, '(h w) c -> h w c', h=H)
    # print('Diff of rgb & samples:', torch.sum(torch.abs(rgb-samples)))
    # samples = (samples * 255).cpu().numpy().astype(np.uint8)
    # imageio.imsave(os.path.join(dir_out, 'samples.png'), samples)

def test_grid_sample():
    h, w =  10, 20
    values = torch.randn(h, w, 3)
    grid = create_meshgrid(h, w)[0]

    values = values.permute(2, 0, 1)[None]
    grid = grid[None]

    print('values:', values.size())
    print('grid:', grid.size())
    out = F.grid_sample(
        values, 
        grid,
        mode='bilinear',
        align_corners=True
    )

    print('out:', out.size())
    print('diff:', torch.sum(torch.abs(out - values)))

if __name__ == '__main__':
    hparams = get_opts()
    render_panorama(hparams)
    # test_grid_sample()