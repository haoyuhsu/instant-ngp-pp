import numpy as np
import os 
import argparse
from PIL import Image, ImageFilter
import cv2
import imageio
from tqdm import tqdm
import glob


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def sort_key(x):
    if len(x) > 2 and x[-10] == "_":
        return x[-9:]
    return x


def downsample_image(image, new_size):
    img = Image.fromarray(image)
    if len(image.shape) == 3:
        img = img.filter(ImageFilter.GaussianBlur(radius=2))  # anti-aliasing for rgb image
        img = img.resize(new_size)
    else:
        img = img.resize(new_size, Image.NEAREST)   # use nearest neighbour for depth map
    return np.array(img)


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
    for path in tqdm(paths):
        rgb = Image.open(path).convert("RGBA")
        rgb = np.array(rgb)
        rgbs.append(rgb)
    rgbs = np.array(rgbs)
    return rgbs


def read_depth_frames(dir_path):
    paths = sorted(glob.glob(os.path.join(dir_path, '*depth.npy')), key=sort_key)
    depths = []
    for path in tqdm(paths):
        d = np.load(path)
        depths.append(d)
    depths = np.array(depths)
    return depths


def parse_obj_rgb(dir_path):
    paths = sorted(os.path.join(dir_path, img) for img in os.listdir(dir_path))
    rgbs = []
    for path in tqdm(paths):
        rgb = Image.open(path).convert("RGBA")
        rgb = np.array(rgb)
        rgbs.append(rgb)
    rgbs = np.array(rgbs)
    return rgbs


def parse_obj_depth(dir_path):
    dirs = sorted(os.path.join(dir_path, depth) for depth in os.listdir(dir_path))
    paths = [os.path.join(d, os.listdir(d)[0]) for d in dirs]
    depths = []
    for path in tqdm(paths):
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depths.append(d[:, :, 0])
    depths = np.array(depths)
    return depths


def blend_frames(root_dir):
    blend_results_dir = os.path.join(root_dir, 'blend_results')

    bg_rgb = read_rgb_frames(os.path.join(root_dir, 'frames'))
    bg_depth  = read_depth_frames(os.path.join(root_dir, 'depth'))
    obj_rgb = parse_obj_rgb(os.path.join(blend_results_dir, 'rgb_obj'))
    obj_depth = parse_obj_depth(os.path.join(blend_results_dir, 'depth_obj'))
    shadow_rgb = parse_obj_rgb(os.path.join(blend_results_dir, 'rgb_shadow'))
    shadow_depth = parse_obj_depth(os.path.join(blend_results_dir, 'depth_shadow'))
    obj_shadow_rgb = parse_obj_rgb(os.path.join(blend_results_dir, 'rgb_obj_shadow'))
    obj_shadow_depth = parse_obj_depth(os.path.join(blend_results_dir, 'depth_obj_shadow'))

    n_frame = bg_rgb.shape[0]
    out_img_dir = os.path.join(blend_results_dir, 'frames')
    os.makedirs(out_img_dir, exist_ok=True)

    # anti-aliasing (use gaussian blur to downsample the image)
    new_size = (bg_rgb.shape[2], bg_rgb.shape[1])
    obj_rgb = np.array([downsample_image(obj_rgb[i], new_size) for i in range(n_frame)])
    obj_depth = np.array([downsample_image(obj_depth[i], new_size) for i in range(n_frame)])
    shadow_rgb = np.array([downsample_image(shadow_rgb[i], new_size) for i in range(n_frame)])
    shadow_depth = np.array([downsample_image(shadow_depth[i], new_size) for i in range(n_frame)])
    obj_shadow_rgb = np.array([downsample_image(obj_shadow_rgb[i], new_size) for i in range(n_frame)])
    obj_shadow_depth = np.array([downsample_image(obj_shadow_depth[i], new_size) for i in range(n_frame)])

    ############################################################
    # store temporary results
    ############################################################
    save_temp_results = True
    if save_temp_results:
        orig_frames = []
        fg_obj_frames = []
        fg_obj_mask_frames = []
        fg_obj_shadow_frames = []
        shadow_frames = []
        shadow_catcher_frames = []
    ############################################################

    frames = []
    for i in tqdm(range(n_frame)):
        bg_c = bg_rgb[i]
        bg_d = bg_depth[i]
        o_c = obj_rgb[i]
        o_d = obj_depth[i]
        s_c = shadow_rgb[i]
        s_d = shadow_depth[i]
        o_s_c = obj_shadow_rgb[i]
        o_s_d = obj_shadow_depth[i]

        bg_c = bg_c.astype(np.float32)
        o_c = o_c.astype(np.float32)
        s_c = s_c.astype(np.float32)
        o_s_c = o_s_c.astype(np.float32)

        # New Implementation of blending
        frame = bg_c.copy()

        # step 1: blend foreground object
        obj_alpha = o_c[..., 3] / 255.
        obj_mask = obj_alpha > 0.0
        depth_mask = o_d < bg_d
        mask = np.logical_and(obj_mask, depth_mask)
        frame[mask] = o_s_c[mask] * obj_alpha[mask, None] + bg_c[mask] * (1 - obj_alpha[mask, None])

        # step 2: blend shadow
        non_object_alpha = 1. - obj_alpha
        fg_alpha = o_s_c[..., 3] / 255.
        shadow_catcher_alpha = non_object_alpha * fg_alpha
        shadow_catcher_mask = shadow_catcher_alpha > 0.0

        color_diff = np.ones_like(o_c)
        color_diff[shadow_catcher_mask, 0:3] = o_s_c[shadow_catcher_mask, :3] / (s_c[shadow_catcher_mask, :3] + 1e-6)

        # option 1: get the mask of shadow area (color_diff not all 1 with 0.01 tolerance)
        shadow_mask = np.logical_not(np.all(np.abs(color_diff - 1) < 0.01, axis=-1))
        # option 2: get the mask of shadow area (color_diff all smaller than 1)
        # shadow_mask = np.logical_and(color_diff[..., 0] < 1, color_diff[..., 1] < 1, color_diff[..., 2] < 1)

        depth_mask = o_s_d < bg_d
        mask = np.logical_and(shadow_mask, depth_mask)

        # frame[mask, 0:3] *= color_diff[mask, 0:3]
        frame[mask] = frame[mask] * color_diff[mask] * shadow_catcher_alpha[mask, None] + frame[mask] * (1 - shadow_catcher_alpha[mask, None])

        ############################################################
        # temporary results (original frame, foreground object, foreground object mask, foreground object with shadow, shadow only)
        ############################################################
        if save_temp_results:
            orig_frame = bg_c.copy()
            orig_frame = orig_frame.astype(np.uint8)

            fg_obj_frame = o_c.copy()
            fg_obj_frame = fg_obj_frame.astype(np.uint8)

            fg_obj_mask = np.zeros_like(o_c)
            fg_obj_mask[obj_mask] = 255
            fg_obj_mask[obj_mask, 3] = o_c[obj_mask, 3]  # keep the original alpha value
            fg_obj_mask = fg_obj_mask.astype(np.uint8)

            fg_obj_shadow_frame = o_s_c.copy()
            fg_obj_shadow_frame = fg_obj_shadow_frame.astype(np.uint8)

            color_diff = np.clip(color_diff, 0, 1)  # to avoid numerical issue (clip color_diff to [0, 1])
            shadow_frame = color_diff.copy() * 255
            shadow_frame = shadow_frame.astype(np.uint8)

            shadow_catcher_frame = s_c.copy()
            shadow_catcher_frame = shadow_catcher_frame.astype(np.uint8)

            orig_frames.append(orig_frame)
            fg_obj_frames.append(fg_obj_frame)
            fg_obj_mask_frames.append(fg_obj_mask)
            fg_obj_shadow_frames.append(fg_obj_shadow_frame)
            shadow_frames.append(shadow_frame)
            shadow_catcher_frames.append(shadow_catcher_frame)
        ############################################################

        # convert frame to uint8
        frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)

        frames.append(frame)
        path = os.path.join(out_img_dir, '{:0>4d}.png'.format(i))
        Image.fromarray(frame).save(path)
    
    frames = np.array(frames)
    imageio.mimsave(os.path.join(blend_results_dir, 'blended.mp4'),
        frames, fps=15, macro_block_size=1)
    
    ############################################################
    # save video for temporary results
    ############################################################
    if save_temp_results:
        orig_frames = np.array(orig_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'orig.mp4'),
            orig_frames, fps=15, macro_block_size=1)
        fg_obj_frames = np.array(fg_obj_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'fg_obj.mp4'),
            fg_obj_frames, fps=15, macro_block_size=1)
        fg_obj_mask_frames = np.array(fg_obj_mask_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'fg_obj_mask.mp4'),
            fg_obj_mask_frames, fps=15, macro_block_size=1)
        fg_obj_shadow_frames = np.array(fg_obj_shadow_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'fg_obj_shadow.mp4'),
            fg_obj_shadow_frames, fps=15, macro_block_size=1)
        shadow_frames = np.array(shadow_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'shadow.mp4'),
            shadow_frames, fps=15, macro_block_size=1)
        shadow_catcher_frames = np.array(shadow_catcher_frames)
        imageio.mimsave(os.path.join(blend_results_dir, 'shadow_catcher.mp4'),
            shadow_catcher_frames, fps=15, macro_block_size=1)
    ############################################################
    


if __name__ == '__main__':
    blend_frames('/home/max/Documents/maxhsu/instant-ngp-pp/results/lerf/teatime/trajectory_001/')