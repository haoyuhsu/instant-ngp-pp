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
from utils import load_ckpt, save_image, convert_normal
from opt import get_opts
from einops import rearrange
import skimage
import glob


def generate_video_from_frames(frames_path, video_name, fps=30):
    frame_series = [np.array(Image.open(frame_path)) for frame_path in frames_path]  # return (0~255 in uint8)
    # reshape the size of the frames to be divisible by 2 for video rendering
    h, w = frame_series[0].shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    frame_series = [(skimage.transform.resize(frame, (new_h, new_w)) * 255.).astype(np.uint8) for frame in frame_series]
    imageio.mimsave(video_name,
        frame_series,
        fps=fps, macro_block_size=1)


def depth2img(depth, scale=16):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def semantic2img(sem_label, classes):
    level = 1/(classes-1)
    sem_color = level * sem_label
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color


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
        if k in ['total_samples']:
            continue
        results[k] = torch.cat(results[k], 0)
    return results


def render_for_test(hparams, split='test'):

    print("##### rendering for test on {}/{} #####".format(hparams.dataset_name, hparams.exp_name))

    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
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

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model', 'density_grid', 'grid_coords'])
    print('Loaded checkpoint: {}'.format(ckpt_path))    
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_interpolate': hparams.render_interpolate,
            'render_traj': hparams.render_traj,
            'anti_aliasing_factor': hparams.anti_aliasing_factor,
            'scale_poses': hparams.scale_poses}

    if hparams.dataset_name == 'kitti':
            kwargs['seq_id'] = hparams.kitti_seq
            kwargs['frame_start'] = hparams.kitti_start
            kwargs['frame_end'] = hparams.kitti_end
            kwargs['test_id'] = hparams.kitti_test_id

    if hparams.dataset_name == 'mega':
            kwargs['mega_frame_start'] = hparams.mega_frame_start
            kwargs['mega_frame_end'] = hparams.mega_frame_end

    dataset_test = dataset(split='test', **kwargs)
    w, h = dataset_test.img_wh
    
    if hparams.embed_a:
        dataset_train = dataset(split='train', **kwargs)
        embed_a_length = hparams.embed_a_len
        embedding_a = torch.nn.Embedding(len(dataset_train.poses), embed_a_length).cuda() 
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["model", "msk_model"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())    
    
    poses, render_traj_rays = None, None
    if hparams.render_traj or hparams.render_train:
        render_traj_rays = dataset_test.render_traj_rays
        poses = dataset_test.c2w
    else:
        pass
        # render_traj_rays = dataset.rays
        # render_traj_rays = {}
        # print("generating rays' origins and directions!")
        # poses = dataset_test.poses
        # for img_idx in trange(len(poses)):
        #     rays_o, rays_d = get_rays(dataset_test.directions.cuda(), dataset_test[img_idx]['pose'].cuda())
        #     render_traj_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    if hparams.render_train:
        BASE_RESULT_DIR = f'results/{hparams.dataset_name}/{hparams.exp_name}/trajectory_train'
    if hparams.render_traj is not None:
        BASE_RESULT_DIR = f'results/{hparams.dataset_name}/{hparams.exp_name}/{dataset_test.traj_name}'

    frames_dir = f'{BASE_RESULT_DIR}/frames'
    depth_dir = f'{BASE_RESULT_DIR}/depth'

    if hparams.render_interpolate:
        frames_dir += '_interp'
        depth_dir += '_interp'
    
    # if hparams.upsample > 1:
    #     frames_dir += f'_upsample{hparams.upsample}'
    #     depth_dir += f'_upsample{hparams.upsample}'

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # save interpolated poses
    # if hparams.render_interpolate:
    #     poses_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/poses_interpolate'
    #     os.makedirs(poses_dir, exist_ok=True)
    #     c2w_list = dataset.c2w
    #     for i in range(len(c2w_list)):
    #         np.savetxt(os.path.join(poses_dir, '{:0>4d}-pose.txt'.format(i)), c2w_list[i])

    # frame_series = []
    # depth_raw_series = []
    # depth_series = []
    # points_series = []
    # normal_series = []
    # normal_raw_series = []
    # semantic_series = []

    for img_idx in trange(len(render_traj_rays)):
        rays = render_traj_rays[img_idx][:, :6].cuda()
        pose = poses[img_idx]
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
            'img_wh': dataset_test.img_wh,
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
            # frame_series.append(rgb_frame)
            cv2.imwrite(os.path.join(frames_dir, '{:0>4d}-rgb.png'.format(img_idx)), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        # if hparams.render_semantic:
        #     sem_frame = semantic2img(rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h), hparams.num_classes)
        #     # semantic_series.append(sem_frame)

        if hparams.render_depth:
            depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
            np.save(os.path.join(depth_dir, '{:0>4d}-depth.npy'.format(img_idx)), depth_raw.astype(np.float32))
            # depth_raw_series.append(depth_raw.astype(np.float32))
            depth = depth2img(depth_raw, scale=2*hparams.scale)
            # depth_series.append(depth)
            cv2.imwrite(os.path.join(frames_dir, '{:0>4d}-depth.png'.format(img_idx)), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
        
        # if hparams.render_points:
        #     points = rearrange(results['points'].cpu().numpy(), '(h w) c -> h w c', h=h)
            # points_series.append(points)

        if hparams.render_normal:
            pose = pose.cpu().numpy()
            normal_pred = rearrange(results['normal_pred'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6
            normal_vis = (convert_normal(normal_pred, pose) + 1)/2
            save_image((normal_vis), os.path.join(frames_dir, '{:0>4d}-normal.png'.format(img_idx)))
            # normal_series.append((255*normal_vis).astype(np.uint8))
            normal_raw = rearrange(results['normal_raw'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6
            normal_vis = (convert_normal(normal_raw, pose) + 1)/2
            save_image((normal_vis), os.path.join(frames_dir, '{:0>4d}-normal-raw.png'.format(img_idx)))
            # normal_raw_series.append((255*normal_vis).astype(np.uint8))
                        
        torch.cuda.synchronize()

    if hparams.render_rgb:
        frames_path = sorted(glob.glob(os.path.join(frames_dir, '*rgb.png')))
        generate_video_from_frames(frames_path, os.path.join(BASE_RESULT_DIR, 'render_rgb.mp4'))

    # if hparams.render_semantic:
    #     frames_path = sorted(glob.glob(os.path.join(frames_dir, '*semantic.png')))
    #     generate_video_from_frames(frames_path, os.path.join(BASE_RESULT_DIR, 'render_semantic.mp4'))

    if hparams.render_depth:
        frames_path = sorted(glob.glob(os.path.join(frames_dir, '*depth.png')))
        generate_video_from_frames(frames_path, os.path.join(BASE_RESULT_DIR, 'render_depth.mp4'))
        
        # depth_raw_all = np.stack(depth_raw_series) #(n_frames, h ,w)
        # if hparams.render_interpolate:
        #     path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw_interpolate.npy'
        # else:
        #     path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw.npy'
        # np.save(path, depth_raw_all)

    # if hparams.render_points:
    #     points_all = np.stack(points_series)
    #     path = f'results/{hparams.dataset_name}/{hparams.exp_name}/points.npy'
    #     np.save(path, points_all)

    if hparams.render_normal:
        frames_path = sorted(glob.glob(os.path.join(frames_dir, '*normal.png')))
        generate_video_from_frames(frames_path, os.path.join(BASE_RESULT_DIR, 'render_normal.mp4'))
        frames_path = sorted(glob.glob(os.path.join(frames_dir, '*normal-raw.png')))
        generate_video_from_frames(frames_path, os.path.join(BASE_RESULT_DIR, 'render_normal_raw.mp4'))


if __name__ == '__main__':
    hparams = get_opts()
    render_for_test(hparams)