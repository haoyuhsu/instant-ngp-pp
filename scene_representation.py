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


class SceneRepresentation():
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_dir = hparams.root_dir
        self.results_dir = os.path.join('results', hparams.dataset_name, hparams.exp_name)
        os.makedirs(os.path.join(self.results_dir), exist_ok=True)
        self.load_dataset()
        self.load_model()

        self.semantic_mesh_dir = os.path.join(self.dataset_dir, 'semantic_mesh_deva')
        self.set_up_vector()

        self.inserted_objects = []

    def insert_object(self, object_info):
        assert isinstance(object_info, dict)
        assert isinstance(object_info['position'], np.ndarray)
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

    def render_scene(self):
        pass
        # TODO: render the scene with inserted objects
        # 1. render rgb frames & depth maps of the scene
        # 2. use blender to render the scene with objects inserted
        # 3. composite the rendered frames into videos
