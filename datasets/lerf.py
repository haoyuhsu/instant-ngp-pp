import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import json

from .ray_utils import *
from .color_utils import read_image, read_normal, read_normal_up, read_semantic

from .base import BaseDataset


scene_up_vector_dict = {
    'donuts': [0.0, 0.0, 1.0],
    'dozer_nerfgun_waldo': [-0.76060444, 0.00627117, 0.6491853 ],
    'espresso': [0.0, 0.0, 1.0],
    'figurines': [0.0, 0.0, 1.0],
    'ramen': [0.0, 0.0, 1.0],
    'shoe_rack': [0.0, 0.0, 1.0],
    'teatime': [0.0, 0.0, 1.0],
    'waldo_kitchen': [0.0, 0.0, 1.0],
}


scene_scale_dict = {
    'donuts': 1.0,
    'dozer_nerfgun_waldo': 1.5,
    'espresso': 4.0,
    'figurines': 1.5,
    'ramen': 0.7,
    'shoe_rack': 1.2,
    'teatime': 1.6,
    'waldo_kitchen': 3.2,
}


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


class LeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, render_interpolate=False, render_traj=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        # read poses and intrinsics of each frame from json file
        with open(os.path.join(root_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)

        # Sort the 'frames' list by 'file_path' to make sure that the order of images is correct
        # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
        all_file_paths = [frame_info['file_path'] for frame_info in meta['frames']]
        sort_indices = [i[0] for i in sorted(enumerate(all_file_paths), key=lambda x:x[1])]
        meta['frames'] = [meta['frames'][i] for i in sort_indices]

        # Get images directory
        self.imgs_dir = os.path.join(root_dir, 'images')

        # Make sure the number of images matches the number of frames
        # number_list = [i + 1 for i in range(len(imgs))]
        # for f in meta['frames']:
        #     file_path = f['file_path']
        #     number = int(file_path.split('/')[-1].split('.')[0].split('_')[-1])
        #     # remove the number from the number_list if it is found
        #     if number in number_list:
        #         number_list.remove(number)
        #     print(number)
        # print(number_list)
        # print(len(meta['frames']), len(imgs))

        # using only a subset of images for testing
        if split == 'test' and not render_train and not render_traj:
            meta['frames'] = meta['frames'][:20]

        img_path_list = [os.path.join(root_dir, frame_info['file_path']) for frame_info in meta['frames']]
        normal_path_list = sorted(glob.glob(os.path.join(self.root_dir, 'normal', '*.npy')))
        depth_path_list = sorted(glob.glob(os.path.join(self.root_dir, 'depth', '*.npy')))

        tmp_img = Image.open(img_path_list[0])
        w, h = tmp_img.width, tmp_img.height
        # w, h = int(w*downsample), int(h*downsample)
        self.img_wh = (w, h)

        # get c2w poses
        all_c2w = []
        for frame_info in meta['frames']:
            cam_mtx = np.array(frame_info['transform_matrix'])
            cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
        all_render_c2w = torch.stack(all_c2w).float()

        scene_name = os.path.split(root_dir)[-1]

        # calculate alignment rotation matrix to make z-axis point up
        v1 = normalize(np.array(scene_up_vector_dict[scene_name]))
        v2 = np.array([0, 0, 1])
        R = torch.FloatTensor(get_rotation_matrix_from_vectors(v1, v2))
        # rotate c2w matrix by R
        all_render_c2w[:, 0:3, :] = R @ all_render_c2w[:, 0:3, :]
        
        self.up_vector = v1

        # compute scale factor from all camera poses
        scale = torch.linalg.norm(all_render_c2w[..., 0:3, 3], axis=-1).max()
        print(f"scene scale {scale}")

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

        if kwargs['scale_poses']:
            print('Scaling poses ...')
            # normalize by camera
            all_render_c2w[:, 0:3, 3] /= scale
            
        self.c2w = all_render_c2w

        # poses only use 3x4 matrix
        self.poses = self.c2w[:, :3, :]

        # get camera intrinsics
        all_K = []
        all_directions = []
        for frame_info in meta['frames']:
            if 'fl_x' in meta:
                fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
            else:
                fx, fy, cx, cy = frame_info['fl_x'], frame_info['fl_y'], frame_info['cx'], frame_info['cy']
            cam_K = np.array([
                [fx, 0, cx], 
                [0, fy, cy],
                [0, 0, 1]]
            )
            cam_K *= downsample
            all_K.append(torch.from_numpy(cam_K))
            direction = get_ray_directions(h, w, cam_K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
            all_directions.append(direction)
        self.K = torch.stack(all_K)
        self.directions = torch.stack(all_directions)

        # generate rays
        self.rays = None
        self.render_traj_rays = None
        if render_train:
            self.render_traj_rays = self.get_path_rays(all_render_c2w)                    # (h*w, 6) --> ray origin + ray direction
        elif render_traj is not None:
            with open(render_traj, 'rb') as file:
                traj_info = json.load(file)
            self.traj_name = traj_info["trajectory_name"]
            # intrinsics
            fx, fy, cx, cy = traj_info['fl_x'], traj_info['fl_y'], traj_info['cx'], traj_info['cy']
            cam_K = np.array([
                [fx, 0, cx], 
                [0, fy, cy],
                [0, 0, 1]]
            )
            direction = get_ray_directions(h, w, cam_K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
            self.K = torch.stack([torch.from_numpy(cam_K) for _ in range(len(traj_info['frames']))])
            self.directions = torch.stack([direction for _ in range(len(traj_info['frames']))])
            # extrinsics
            render_traj_c2w = [frame_info['transform_matrix'] for frame_info in traj_info['frames']]  # (N, 3, 4)
            render_traj_c2w = torch.FloatTensor(render_traj_c2w)
            self.c2w = render_traj_c2w
            self.render_traj_rays = self.get_path_rays(render_traj_c2w)
        else: # training NeRF
            self.rays = torch.FloatTensor(self.read_rgb(img_path_list))
            self.normals = torch.FloatTensor(self.read_normal(normal_path_list))
            self.render_traj_rays = self.get_path_rays(all_render_c2w)

        self.imgs = img_path_list

        self.scene_scale = scene_scale_dict[scene_name]
    
    
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
                get_rays(self.directions[idx], torch.FloatTensor(c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)
        return rays


if __name__ == '__main__':
    import pickle 
    # save_path = 'output/dataset_cameras/family.pkl'

    # kwargs = {
    #     'root_dir': '../datasets/TanksAndTempleBG/Family',
    #     'render_traj': True,
    # }
    # dataset = tntDataset(
    #     split='test',
    #     **kwargs
    # )
    
    # cam_info = {
    #     'img_wh': dataset.img_wh,
    #     'K': np.array(dataset.K),
    #     'c2w': np.array(dataset.c2w)
    # }

    # with open(save_path, 'wb') as file:
    #     pickle.dump(cam_info, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(save_path, 'rb') as file:
    #     cam = pickle.load(file)
    
    # print('Image W*H:', cam['img_wh'])
    # print('Camera K:', cam['K'])
    # print('Camera poses:', cam['c2w'].shape)
