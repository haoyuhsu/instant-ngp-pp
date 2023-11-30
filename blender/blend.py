import numpy as np
import os 
import argparse
from PIL import Image
import cv2
import imageio
from tqdm import tqdm
import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def sort_key(x):
    if len(x) > 2 and x[-10] == "_":
        return x[-9:]
    return x

# def read_video_frames(video_path):
#     vidcap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         success, image = vidcap.read()
#         if not success:
#             break
#         else:
#             # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             frames.append(image)
#     if len(frames) == 0:
#         print('Not valid path: {}'.format(video_path))
#     frames = np.array(frames)
#     return frames

def read_rgb_frames(dir_path):
    paths = sorted(glob.glob(os.path.join(dir_path, '*rgb.png')), key=sort_key)
    rgbs = []
    for path in paths:
        rgb = Image.open(path).convert("RGBA")
        rgb = np.array(rgb)
        rgbs.append(rgb)
    rgbs = np.array(rgbs)
    return rgbs

def read_depth_frames(dir_path):
    paths = sorted(glob.glob(os.path.join(dir_path, '*depth.npy')), key=sort_key)
    depths = []
    for path in paths:
        d = np.load(path)
        depths.append(d)
    depths = np.array(depths)
    return depths

def parse_obj_rgb(dir_path):
    paths = sorted(os.path.join(dir_path, img) for img in os.listdir(dir_path))
    rgbs = []
    for path in paths:
        rgb = Image.open(path).convert("RGBA")
        rgb = np.array(rgb)
        rgbs.append(rgb)
    rgbs = np.array(rgbs)
    return rgbs

def parse_obj_depth(dir_path):
    dirs = sorted(os.path.join(dir_path, depth) for depth in os.listdir(dir_path))
    paths = [os.path.join(d, os.listdir(d)[0]) for d in dirs]
    depths = []
    for path in paths:
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depths.append(d[:, :, 0])
    depths = np.array(depths)
    return depths

def blend_frames(root_dir):
    blend_results_dir = os.path.join(root_dir, 'blend_results')

    vdo_rgb = read_rgb_frames(os.path.join(root_dir, 'frames'))
    vdo_depth  = read_depth_frames(os.path.join(root_dir, 'depth'))
    obj_rgb = parse_obj_rgb(os.path.join(blend_results_dir, 'rgb'))
    obj_depth = parse_obj_depth(os.path.join(blend_results_dir, 'depth'))
    obj_rgb_shadow = parse_obj_rgb(os.path.join(blend_results_dir, 'rgb_shadow'))
    obj_depth_shadow = parse_obj_depth(os.path.join(blend_results_dir, 'depth_shadow'))

    n_frame = vdo_rgb.shape[0]
    out_img_dir = os.path.join(blend_results_dir, 'frames')
    os.makedirs(out_img_dir, exist_ok=True)
    frames = []
    for i in tqdm(range(n_frame)):
        v_c = vdo_rgb[i]
        v_d = vdo_depth[i]
        o_c = obj_rgb[i]
        o_d = obj_depth[i]
        o_c_shadow = obj_rgb_shadow[i]
        o_d_shadow = obj_depth_shadow[i]
        
        # Original Implementation of blending (consider planar shadow catcher)
        # frame = v_c.copy()
        # mask = o_d < v_d
        # alpha = o_c[mask, 3] / 255.
        # frame[mask] = o_c[mask] * alpha[..., None] + v_c[mask] * (1 - alpha[..., None])

        # Current Implementation of blending (consider meshes shadow catcher)
        frame = v_c.copy()
        obj_mask = o_c[..., 3] / 255.
        obj_mask[obj_mask > 0.0] = 1
        o_c = o_c + 1. * o_c_shadow * (1 - obj_mask[..., None])  # blend visible shadow into object image
        o_d = o_d * obj_mask + o_d_shadow * (1 - obj_mask)  # blend visible shadow into object depth
        mask = o_d < v_d
        alpha = o_c[mask, 3] / 255.
        frame[mask] = o_c[mask] * alpha[..., None] + v_c[mask] * (1 - alpha[..., None])   # blend object image with background image

        frames.append(frame)
        path = os.path.join(out_img_dir, '{:0>4d}.png'.format(i))
        Image.fromarray(frame).save(path)
    
    frames = np.array(frames)
    imageio.mimsave(os.path.join(blend_results_dir, 'blended.mp4'),
        frames, fps=10, macro_block_size=1)


if __name__ == '__main__':
    pass