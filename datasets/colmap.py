import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, read_semantic
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def center_poses(poses, pts3d):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T

    return poses_centered, pts3d_centered


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, render_train=False, render_interpolate=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics(**kwargs)

        # if kwargs.get('read_meta', True):
        #     self.read_extrinsics(split, **kwargs)

        self.read_extrinsics(split, render_train, render_interpolate, **kwargs)

    def read_intrinsics(self, **kwargs):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

    def read_extrinsics(self, split, render_train, render_interpolate, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]

        # dummy line for compatibility
        classes = kwargs.get('num_classes', 7)
        semantics = []
        depths = []
        
        perm = np.argsort(img_names)
        if '360' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
            if kwargs.get('use_sem', False):
                semantics = f'semantic_{int(1/self.downsample)}'
        else:
            folder = 'images'
            if kwargs.get('use_sem', False):
                semantics = 'semantic'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        
        self.imgs = img_paths
        
        if kwargs.get('use_sem', False):
            sem_paths = []
            for name in sorted(img_names):
                sem_file_name = os.path.splitext(name)[0]+'.pgm'             
                sem_paths.append(os.path.join(self.root_dir, semantics, sem_file_name))
        
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices
        all_render_c2w = torch.FloatTensor(poses)

        # pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        # pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)
        # self.poses, self.pts3d = center_poses(poses, pts3d)

        # self.up = torch.FloatTensor(-normalize(all_render_c2w[:, :3, 1].mean(0)))
        # print(f"scene up {self.up}")

        # do interpolation
        if render_interpolate:
            ### Option 1: simple interpolation (linear) ###
            # all_render_c2w_new = []
            # for i, pose in enumerate(all_render_c2w):
            #     # if len(all_render_c2w_new) >= 600:
            #     #     break
            #     all_render_c2w_new.append(pose)
            #     if i>0 and i<len(all_render_c2w)-1:
            #         pose_new = (pose*3+all_render_c2w[i+1])/4
            #         all_render_c2w_new.append(pose_new)
            #         pose_new = (pose+all_render_c2w[i+1])/2
            #         all_render_c2w_new.append(pose_new)
            #         pose_new = (pose+all_render_c2w[i+1]*3)/4
            #         all_render_c2w_new.append(pose_new)
            # all_render_c2w = torch.stack(all_render_c2w_new)
            ### Option 2: interpolate smooth spline path ###
            all_render_c2w = generate_interpolated_path(all_render_c2w.numpy(), 4)
           
        scale = torch.linalg.norm(all_render_c2w[..., 0:3, 3], axis=-1).max()
        print(f"scene scale {scale}")
        all_render_c2w[:, 0:3, 3] /= scale
        # self.pts3d /= scale

        # using only a subset of images for testing
        if split == 'test' and not render_train:
            self.imgs = self.imgs[:20]
            all_render_c2w = all_render_c2w[:20]

        self.c2w = all_render_c2w
        self.poses = all_render_c2w

        # generate rays
        self.rays = None
        self.render_traj_rays = None
        if render_train:
            self.render_traj_rays = self.get_path_rays(all_render_c2w)                    # (h*w, 6) --> ray origin + ray direction
        else: # training NeRF
            self.rays = self.read_meta(split, self.imgs, all_render_c2w, semantics, classes)   # (h*w, 3) --> RGB values
        
        # self.rays = []
        # if kwargs.get('use_sem', False):
        #     self.labels = []
        # if split == 'test_traj': # use precomputed test poses
        #     self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean(), )
        #     self.poses = torch.FloatTensor(self.poses)
        #     return
        
        # if kwargs.get('use_sem', False):
        #     classes = kwargs.get('num_classes', 7)
        #     for sem_path in sem_paths:
        #         label = read_semantic(sem_path=sem_path, sem_wh=self.img_wh, classes=classes)
        #         self.labels += [label]
        #     self.labels = torch.LongTensor(np.stack(self.labels))
            
    def read_meta(self, split, imgs, c2w_list, semantics, classes=7):
        # rays = {} # {frame_idx: ray tensor}
        rays = []
        norms = []
        labels = []

        self.poses = []
        print(f'Loading {len(imgs)} {split} images ...')
        if len(semantics)>0:
            for idx, (img, sem) in enumerate(tqdm(zip(imgs, semantics))):
                c2w = np.array(c2w_list[idx][:3])
                self.poses += [c2w]

                img = read_image(img_path=img, img_wh=self.img_wh)
                if img.shape[-1] == 4:
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                rays += [img]
                
                label = read_semantic(sem_path=sem, sem_wh=self.img_wh, classes=classes)
                labels += [label]
            
            self.poses = torch.FloatTensor(np.stack(self.poses))
            
            return torch.FloatTensor(np.stack(rays)), torch.LongTensor(np.stack(labels))
        else:
            for idx, img in enumerate(tqdm(imgs)):
                c2w = np.array(c2w_list[idx][:3])
                self.poses += [c2w]

                img = read_image(img_path=img, img_wh=self.img_wh)
                if img.shape[-1] == 4:
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                rays += [img]
            
            self.poses = torch.FloatTensor(np.stack(self.poses))
            
            return torch.FloatTensor(np.stack(rays))

    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays
