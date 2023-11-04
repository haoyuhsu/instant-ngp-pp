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

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

class LeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, render_interpolate=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        def sort_key(x):
            return x
        
        img_dir_name = None 
        sem_dir_name = None 
        depth_dir_name = None
        if os.path.exists(os.path.join(root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(root_dir, 'rgb')):
            img_dir_name = 'rgb'
        # if os.path.exists(os.path.join(root_dir, 'semantic')):
        #     sem_dir_name = 'semantic'
        # if os.path.exists(os.path.join(root_dir, 'semantic_inst')):
        #     sem_dir_name = 'semantic_inst'
        # if os.path.exists(os.path.join(root_dir, 'semantic_cat')):
        #     sem_dir_name = 'semantic_cat'
        if os.path.exists(os.path.join(root_dir, 'depth')):
            depth_dir_name = 'depth'

        semantics = []
        # if kwargs.get('use_sem', False):            
        #     # semantics = sorted(glob.glob(os.path.join(self.root_dir, sem_dir_name, prefix+'*.pgm')), key=sort_key)
        #     semantics = sorted(glob.glob(os.path.join(self.root_dir, sem_dir_name, prefix+'*.npy')), key=sort_key)
        classes = kwargs.get('num_classes', 7)

        depths = []
        # if kwargs.get('depth_mono', False):            
        #     depths = sorted(glob.glob(os.path.join(self.root_dir, depth_dir_name, prefix+'*.npy')), key=sort_key)

        # read poses and intrinsics of each frame from json file
        with open(os.path.join(root_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)

        # Sort the 'frames' list by 'file_path' to make sure that the order of images is correct
        # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
        all_file_paths = [frame_info['file_path'] for frame_info in meta['frames']]
        sort_indices = [i[0] for i in sorted(enumerate(all_file_paths), key=lambda x:x[1])]
        meta['frames'] = [meta['frames'][i] for i in sort_indices]

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
        if split == 'test' and not render_train:
            meta['frames'] = meta['frames'][:20]

        imgs = [os.path.join(root_dir, frame_info['file_path']) for frame_info in meta['frames']]
        self.imgs = imgs

        tmp_img = Image.open(imgs[0])
        w, h = tmp_img.width, tmp_img.height
        # w, h = int(w*downsample), int(h*downsample)
        self.img_wh = (w, h)

        # get c2w poses
        all_c2w = []
        for frame_info in meta['frames']:
            cam_mtx = np.array(frame_info['transform_matrix'])
            cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
        all_render_c2w = torch.stack(all_c2w)
        
        # self.up = -normalize(all_render_c2w[:,:3,1].mean(0))
        # print(f'up vector: {self.up}')

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
            all_render_c2w = generate_interpolated_path(all_render_c2w.numpy(), 4)

        # normalize by camera
        all_render_c2w[:, 0:3, 3] /= scale
            
        self.c2w = all_render_c2w

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
        else: # training NeRF
            self.rays = self.read_meta(split, imgs, all_render_c2w, semantics, classes)   # (h*w, 3) --> RGB values

        # if split.startswith('train'):
        #     if len(semantics)>0:
        #         self.rays, self.labels = self.read_meta('train', imgs, all_render_c2w, semantics, classes)
        #     else:
        #         self.rays = self.read_meta('train', imgs, all_render_c2w, semantics, classes)
        #     if len(depths)>0:
        #         self.depths_2d = self.read_depth(depths)
        # else: # val, test
        #     if len(semantics)>0:
        #         self.rays, self.labels = self.read_meta(split, imgs, all_render_c2w, semantics, classes)
        #     else:
        #         self.rays = self.read_meta(split, imgs, all_render_c2w, semantics, classes)
        #     if len(depths)>0:
        #         self.depths_2d = self.read_depth(depths)
        #     if kwargs.get('render_normal_mask', False):
        #         self.render_normal_rays = self.get_path_rays(render_normal_c2w_f64)

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
    
    def read_depth(self, depths):
        depths_ = []

        for depth in depths:
            depths_ += [rearrange(np.load(depth), 'h w -> (h w)')]
        return torch.FloatTensor(np.stack(depths_))
    
    def get_path_rays(self, c2w_list):

        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])

            rays_o, rays_d = \
                get_rays(self.directions[idx], torch.FloatTensor(render_c2w))

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
