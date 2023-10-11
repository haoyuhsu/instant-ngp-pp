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
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        def sort_key(x):
            if len(x) > 2 and x[-10] == "_":
                return x[-9:]
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
        if os.path.exists(os.path.join(root_dir, 'semantic_inst')):
            sem_dir_name = 'semantic_inst'
        # if os.path.exists(os.path.join(root_dir, 'semantic_cat')):
        #     sem_dir_name = 'semantic_cat'
        if os.path.exists(os.path.join(root_dir, 'depth')):
            depth_dir_name = 'depth'

        self.has_render_traj = False
        if split == "test" and not render_train:
            self.has_render_traj = os.path.exists(os.path.join(root_dir, 'camera_path'))
        
        prefix = ''
        
        imgs = sorted(glob.glob(os.path.join(self.root_dir, img_dir_name, prefix+'*.jpg')), key=sort_key)
        for img_file_path in imgs:
            img = Image.open(img_file_path)
            w, h = img.width, img.height
            break
        w, h = int(w*downsample), int(h*downsample)
        self.img_wh = (w, h)
        
        semantics = []
        if kwargs.get('use_sem', False):            
            # semantics = sorted(glob.glob(os.path.join(self.root_dir, sem_dir_name, prefix+'*.pgm')), key=sort_key)
            semantics = sorted(glob.glob(os.path.join(self.root_dir, sem_dir_name, prefix+'*.npy')), key=sort_key)
        depths = []
        if kwargs.get('depth_mono', False):            
            depths = sorted(glob.glob(os.path.join(self.root_dir, depth_dir_name, prefix+'*.npy')), key=sort_key)

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

        ########################################################## get g.t. poses:
        all_c2w = []
        for frame_info in meta['frames']:
            cam_mtx = np.array(frame_info['transform_matrix'])
            cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
        c2w_f64 = torch.stack(all_c2w)
        
        # center = c2w_f64[:, :3, 3].mean(axis=0)
        # # radius = np.linalg.norm((c2w_f64[:, :3, 3]-center), axis=0).mean(axis=0)
        # up = -normalize(c2w_f64[:, :3, 1].sum(0))
        self.up = -normalize(c2w_f64[:,:3,1].mean(0))
        print(f'up vector: {self.up}')

        # compute scale of poses
        scale = torch.linalg.norm(c2w_f64[..., 3], axis=-1).max()
        print(f"scene scale {scale}")

        ########################################################## get g.t. intrinsics:
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
    
###########################################################
        if self.has_render_traj or render_train:
            print("render camera path" if not render_train else "render train interpolation")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose"))
                if x.endswith(".txt")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_c2w_f64 = torch.stack(all_render_c2w)
############ here we generate the test trajectories
############ we store the poses in render_c2w_f64
            ###### do interpolation ######
            if render_train:
                all_render_c2w_new = []
                for i, pose in enumerate(all_render_c2w):
                    if len(all_render_c2w_new) >= 600:
                        break
                    all_render_c2w_new.append(pose)
                    if i>0 and i<len(all_render_c2w)-1:
                        pose_new = (pose*3+all_render_c2w[i+1])/4
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1])/2
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1]*3)/4
                        all_render_c2w_new.append(pose_new)

                render_c2w_f64 = torch.stack(all_render_c2w_new)
            render_c2w_f64 = generate_interpolated_path(render_c2w_f64.numpy(), 2)
            self.c2w = render_c2w_f64

        if kwargs.get('render_normal_mask', False):
            print("render up normal mask for train data!")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "pose"))
                if x.endswith(".txt") and x.startswith("0_")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_normal_c2w_f64 = torch.stack(all_render_c2w)

############################################################ normalize by camera
        c2w_f64[..., 3] /= scale
        
        if self.has_render_traj or render_train:
            render_c2w_f64[..., 3] /= scale
        
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 /=scale
                                    
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 = np.array(render_normal_c2w_f64)
            # render_normal_c2w_f64 = pose_avg_inv @ render_normal_c2w_f64
            render_normal_c2w_f64 = render_normal_c2w_f64[:, :3]
           
########################################################### gen rays
        classes = kwargs.get('num_classes', 7)
        self.imgs = imgs
        if split.startswith('train'):
            if len(semantics)>0:
                self.rays, self.labels = self.read_meta('train', imgs, c2w_f64, semantics, classes)
            else:
                self.rays = self.read_meta('train', imgs, c2w_f64, semantics, classes)
            if len(depths)>0:
                self.depths_2d = self.read_depth(depths)
        else: # val, test
            if len(semantics)>0:
                self.rays, self.labels = self.read_meta(split, imgs, c2w_f64, semantics, classes)
            else:
                self.rays = self.read_meta(split, imgs, c2w_f64, semantics, classes)
            if len(depths)>0:
                self.depths_2d = self.read_depth(depths)
            
            if self.has_render_traj or render_train:
                self.render_traj_rays = self.get_path_rays(render_c2w_f64)
            if kwargs.get('render_normal_mask', False):
                self.render_normal_rays = self.get_path_rays(render_normal_c2w_f64)

    def read_meta(self, split, imgs, c2w_list, semantics, classes=7):
        # rays = {} # {frame_idx: ray tensor}
        rays = []
        norms = []
        labels = []
        
        # if split == 'train': prefix = '0_'
        # elif split == 'val': prefix = '1_'
        # elif 'Synthetic' in self.root_dir: prefix = '2_'
        # elif split == 'test': prefix = '1_' # test set for real scenes

        self.poses = []
        print(f'Loading {len(imgs)} {split} images ...')
        if len(semantics)>0:
            for idx, (img, sem) in enumerate(tqdm(zip(imgs, semantics))):
                c2w = np.array(c2w_list[idx][:3])
                self.poses += [c2w]

                img = read_image(img_path=img, img_wh=self.img_wh)
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
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
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
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
