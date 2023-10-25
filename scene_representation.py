import os
import torch
import imageio
import numpy as np
import cv2
import math 
from PIL import Image
from tqdm import trange
from models.networks import NGP
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt, save_image
from opt import get_opts
from einops import rearrange

from render import render_chunks, depth2img, semantic2img
from blender import blend
import json

class SceneRepresentation():
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_dir = hparams.root_dir
        self.results_dir = os.path.join('results', hparams.dataset_name, hparams.exp_name)
        os.makedirs(os.path.join(self.results_dir), exist_ok=True)
        self.load_dataset()
        self.load_model()

        self.semantic_mesh_dir = os.path.join(self.results_dir, 'semantic_mesh_deva')
        self.set_up_vector()

        self.inserted_objects = []
        self.blender_cfg = {}

    def insert_object(self, object_info):
        assert isinstance(object_info, dict)
        assert isinstance(object_info['pos'], np.ndarray)
        self.inserted_objects.append(object_info)

    def set_up_vector(self):
        if self.hparams.dataset_name == 'tnt':
            self.up_vector = np.array([0, -1, 0])
        elif self.hparams.dataset_name == 'lerf':
            self.up_vector = np.array([0, 0, 1])
        elif self.hparams.dataset_name == '360':
            self.up_vector = np.array([0, 1, 0])  # this one not quite sure
        else:
            raise NotImplementedError

    def load_model(self):
        hparams = self.hparams
        rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
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
        self.model = model
        self.ckpt_path = ckpt_path

    def load_dataset(self):
        hparams = self.hparams
        if os.path.exists(os.path.join(hparams.root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
            img_dir_name = 'rgb'
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        dataset = dataset_dict[hparams.dataset_name]
        kwargs = {'root_dir': hparams.root_dir,
                'downsample': hparams.downsample,
                'render_train': hparams.render_train,
                'render_interpolate': hparams.render_interpolate,
                'render_traj': hparams.render_traj,
                'anti_aliasing_factor': hparams.anti_aliasing_factor}
        dataset = dataset(split='test', **kwargs)
        self.dataset = dataset
        self.N_imgs = N_imgs

    def render_scene(self, skip_render_NeRF=False):
        # TODO: render the scene with inserted objects
        # 1. render rgb frames & depth maps of the scene
        if not skip_render_NeRF:
            print('Rendering the rgb frames & depth maps with NeRF...')
            self.render_from_NeRF()
        # 2. use blender to render the scene with objects inserted, then compositing
        if self.inserted_objects:
            print('Rendering the scene with inserted objects...')
            self.render_from_blender()

    def save_cfg(self, cfg, cfg_path):
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)

    def set_basic_blender_cfg(self):
        new_cfg = {}
        new_cfg['results_dir'] = self.results_dir
        new_cfg['im_width'], new_cfg['im_height'] = self.dataset.img_wh
        new_cfg['K'] = self.dataset.K.cpu().numpy().tolist()
        new_cfg['c2w'] = self.dataset.c2w.cpu().numpy().tolist()
        new_cfg['up_vector'] = self.up_vector.tolist()
        new_cfg['env_map_path'] = os.path.join(self.results_dir, 'graveyard_pathways_4k.exr')  # temporary file
        self.blender_cfg.update(new_cfg)

    def render_from_blender(self):    
        cfg_path = os.path.join(self.results_dir, 'blender_cfg.json')
        self.set_basic_blender_cfg()
        if self.inserted_objects:
            insert_object_info = []
            for obj in self.inserted_objects:
                insert_object_info.append({
                    'object_name': obj['object_name'],
                    'object_id': obj['object_id'],
                    'object_path': obj['object_path'],
                    'pos': obj['pos'].tolist(),
                    'rot': obj['rot'].tolist() if obj['rot'] is not None else None,
                    'scale': 0.03
                })
            self.blender_cfg['insert_object_info'] = insert_object_info
        self.save_cfg(self.blender_cfg, cfg_path)
        os.system('{} --background --python ./blender/vc_rendering.py -- --input_config_path={}'.format('/snap/bin/blender', cfg_path))
        blend.blend_frames(self.results_dir)

    def render_from_NeRF(self):
        hparams = self.hparams
        model = self.model
        dataset = self.dataset
        results_dir = self.results_dir

        embed_a_length = hparams.embed_a_len
        if hparams.embed_a:
            embedding_a = torch.nn.Embedding(self.N_imgs, embed_a_length).cuda() 
            load_ckpt(embedding_a, self.ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["model", "msk_model"])
            embedding_a = embedding_a(torch.tensor([0]).cuda())
        
        # setup output directory
        if hparams.render_interpolate:
            frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/frames_interpolate'
        else:
            frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/frames'
        os.makedirs(frames_dir, exist_ok=True)

        # save camera poses
        if hparams.render_interpolate:
            poses_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/poses_interpolate'
        else:
            poses_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/poses'
        os.makedirs(poses_dir, exist_ok=True)
        c2w_list = dataset.c2w
        for i in range(len(c2w_list)):
            np.savetxt(os.path.join(poses_dir, '{:0>4d}-pose.txt'.format(i)), c2w_list[i])

        w, h = dataset.img_wh
        render_traj_rays = dataset.render_traj_rays

        # render frames
        frame_series = []
        depth_raw_series = []
        depth_series = []

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

            rays_o = rays[:, :3]
            rays_d = rays[:, 3:6]
            results = {}
            chunk_size = hparams.chunk_size
            if chunk_size > 0:
                results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)
            else:
                results = render(model, rays_o, rays_d, **render_kwargs)

            if hparams.render_rgb:
                rgb_frame = None
                if hparams.anti_aliasing_factor > 1.0:
                    h_new = int(h*hparams.anti_aliasing_factor)
                    rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h_new)
                    rgb_frame = Image.fromarray((rgb_frame*255).astype(np.uint8)).convert('RGB')
                    rgb_frame = np.array(rgb_frame.resize((w, h), Image.Resampling.BICUBIC))
                else:
                    rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                    rgb_frame = (rgb_frame*255).astype(np.uint8)
                frame_series.append(rgb_frame)
                cv2.imwrite(os.path.join(frames_dir, '{:0>4d}-rgb.png'.format(img_idx)), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            if hparams.render_depth:
                depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
                depth_raw_series.append(depth_raw)
                depth = depth2img(depth_raw, scale=2*hparams.scale)
                depth_series.append(depth)
                cv2.imwrite(os.path.join(frames_dir, '{:0>4d}-depth.png'.format(img_idx)), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
            
            torch.cuda.synchronize()

        if hparams.render_rgb:
            imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_rgb.mp4'),
                            frame_series,
                            fps=30, macro_block_size=1)

        if hparams.render_depth:
            imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_depth.mp4'),
                            depth_series,
                            fps=30, macro_block_size=1)
            # save raw depth maps
            depth_raw_all = np.stack(depth_raw_series) #(n_frames, h ,w)
            if hparams.render_interpolate:
                path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw_interpolate.npy'
            else:
                path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw.npy'
            np.save(path, depth_raw_all)
