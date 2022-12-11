import os
import torch
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import numpy as np
import cv2
from tqdm import trange
from models.network_distill import NGP_distill
from models.networks_sem_2 import NGP, Normal
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from opt import get_opts
from einops import rearrange

def semantic2img(sem_label, classes):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    level = 1/(classes-1)
    sem_color = level * sem_label
    # depth = np.clip(depth, a_min=0., a_max=1.)
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    normal_model = Normal().cuda()
    if split=='train':
        ckpt_path = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs+hparams.normal_epochs-1}_slim.ckpt'
    else:
        ckpt_path = hparams.ckpt_path        
    print(f'ckpt specified: {ckpt_path} !')
    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net.params'])
    load_ckpt(normal_model, ckpt_path, model_name='normal_model', prefixes_to_ignore=["embedding_a", "center", "xyz_min", "xyz_max", "half_size", "density_bitfield", "xyz_encoder.params", "dir_encoder.params", "rgb_net.params", "skybox_dir_encoder.params", "skybox_rgb_net.params"])
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
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train}
    dataset_train = dataset(split='train', **kwargs)
    dataset_test = dataset(split='test', **kwargs)
    w, h = dataset_train.img_wh
    
    directions = dataset_train.directions
    
    poses = []
    poses.append(dataset_train.poses)
    poses.append(dataset_test.poses)
    
    names = []
    names.append(dataset_train.imgs)
    names.append(dataset_test.imgs)

    for poses_, names_ in zip(poses, names):
        for img_idx in trange(len(poses_)):
            # import ipdb; ipdb.set_trace()
            # rays = poses_[img_idx][:, :6].cuda()
            rays_o, rays_d = get_rays(directions.cuda(), poses_[img_idx].cuda())
            render_kwargs = {'test_time': True,
                        'T_threshold': 1e-2,
                        'use_skybox': hparams.use_skybox,
                        'render_rgb': hparams.render_rgb,
                        'render_depth': hparams.render_depth,
                        'render_normal': hparams.render_normal,
                        'render_up_sem': hparams.render_normal_up,
                        'render_sem': hparams.render_semantic,
                        'distill_normal': hparams.render_normal,
                        'img_wh': (w, h),
                        'normal_model': normal_model}
            if hparams.dataset_name in ['colmap', 'nerfpp']:
                render_kwargs['exp_step_factor'] = 1/256
            if hparams.embed_a:
                render_kwargs['embedding_a'] = embedding_a
            results = render(model, rays_o, rays_d, **render_kwargs)
            
            if hparams.render_semantic:
                frame = rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h)
                frame = (frame).astype(np.uint8)
                # import ipdb; ipdb.set_trace()
                imageio.imsave(os.path.join(f'{hparams.root_dir}/semantic_pred', names_[img_idx][-11:].replace('png', 'pgm')), frame)
                
            torch.cuda.synchronize()

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.normal_distillation_only:
        assert hparams.ckpt_path is not None, "No ckpt specified when distilling normals"
        hparams.num_epochs = 0
    render_for_test(hparams)