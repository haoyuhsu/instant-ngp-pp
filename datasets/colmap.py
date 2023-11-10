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
        img_path_list = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        
        normal_path_list = sorted(glob.glob(os.path.join(self.root_dir, 'normal', '*.npy')))
        depth_path_list = sorted(glob.glob(os.path.join(self.root_dir, 'depth', '*.npy')))

        # convert w2c to c2w
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
            all_render_c2w = torch.FloatTensor(generate_interpolated_path(all_render_c2w.numpy(), 4))
           
        scale = torch.linalg.norm(all_render_c2w[..., 0:3, 3], axis=-1).max()
        print(f"scene scale {scale}")

        if kwargs['scale_poses']:
            print('Scaling poses ...')
            # normalize by camera
            all_render_c2w[:, 0:3, 3] /= scale
        # self.pts3d /= scale

        # using only a subset of images for testing
        if split == 'test' and not render_train:
            img_path_list = img_path_list[:20]
            normal_path_list = normal_path_list[:20]
            all_render_c2w = all_render_c2w[:20]

        self.c2w = all_render_c2w

        # poses only use 3x4 matrix
        self.poses = self.c2w[:, :3, :]

        # generate rays
        self.rays = None
        self.render_traj_rays = None
        if render_train:
            self.render_traj_rays = self.get_path_rays(all_render_c2w)                    # (h*w, 6) --> ray origin + ray direction
        else: # training NeRF
            self.rays = torch.FloatTensor(self.read_rgb(img_path_list))
            self.normals = torch.FloatTensor(self.read_normal(normal_path_list))

        self.imgs = img_path_list

    def read_rgb(self, img_path_list):
        """
        Read RGB images from a list of image paths.
        
        Args:
            img_path_list (list): list of image paths.

        Returns:
            rgb_list (np.array): (N, H*W, 3) RGB images.
        """
        rgb_list = []
        for img_path in tqdm(img_path_list):
            img = read_image(img_path=img_path, img_wh=self.img_wh)
            rgb_list += [img]
        rgb_list = np.stack(rgb_list)
        return rgb_list

    def read_depth(self, depth_path_list):
        """
        Read depth maps from a list of depth paths.

        Args:
            depth_path_list (list): list of depth paths.

        Returns:
            depth_list (np.array): (N, H*W) depth maps.
        """
        depth_list = []
        for depth_path in depth_path_list:
            depth_list += [rearrange(np.load(depth_path), 'h w -> (h w)')]
        depth_list = np.stack(depth_list)
        return depth_list
    
    def read_normal(self, norm_path_list):
        """
        Read normal maps from a list of normal paths.

        Args:
            norm_path_list (list): list of normal paths.

        Returns:
            normal_list (np.array): (N, H*W, 3) normal maps.
        """
        poses = self.poses.numpy()
        normal_list = []
        for c2w, norm_path in zip(poses, norm_path_list):
            img = np.load(norm_path).transpose(1, 2, 0)
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

    def get_path_rays(self, render_c2w):
        """
        Get rays from a list of camera poses.

        Args:
            render_c2w (list): list of camera poses.

        Returns:
            rays (dict): {frame_idx: ray tensor}.
        """
        rays = {}
        print(f'Loading {len(render_c2w)} camera path ...')
        for idx in range(len(render_c2w)):
            c2w = np.array(render_c2w[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)
        return rays
