import os
import numpy as np
from opt import get_opts
import glob
import json
from PIL import Image
from tqdm import tqdm, trange

"""
First, you have to run tracking-with-deva on the object categories like 'floor', 'table, 'ground' or 'plane'...etc.
Second, you have to render the normal maps in world frame (i.e., no convert_normal) and store them in .npy format.
"""

def estimate_up_vector(hparams, split='test'):

    frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/frames'
    track_with_deva_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/track_with_deva'
    # id_maps_dir = os.path.join(track_with_deva_dir, 'ID_maps')
    annot_maps_dir = os.path.join(track_with_deva_dir, 'Annotations')

    # Note that the normal maps are already in world coordinate!
    normal_maps_path = sorted(glob.glob(os.path.join(frames_dir, '*.npy')))

    pred_json_path = os.path.join(track_with_deva_dir, 'pred.json')
    with open(pred_json_path, 'r') as f:
        pred_json = json.load(f)

    annotations = pred_json['annotations']

    # only use the first N_frames
    # N_frames = len(annotations)
    N_frames = 200

    all_normal_dirs = []

    for idx, annoation in enumerate(annotations):

        if idx >= N_frames:
            break

        # load normal map
        normal_map = np.load(normal_maps_path[idx])

        # load segment map
        file_name = annoation['file_name'].split('.')[0]
        # segment_map = np.load(os.path.join(id_maps_dir, f'{file_name}.npy'))
        segment_map = Image.open(os.path.join(annot_maps_dir, f'{file_name}.png'))

        # mask the normal map with segment map
        # mask = segment_map != 0

        # [0, 0, 0] is the background for rgb segmented images
        mask_r = np.array(segment_map)[:, :, 0] != 0
        mask_g = np.array(segment_map)[:, :, 1] != 0
        mask_b = np.array(segment_map)[:, :, 2] != 0
        mask = mask_r | mask_g | mask_b

        normal_dirs = normal_map[mask]

        # normalize the normal vectors
        normal_dirs = normal_dirs / np.linalg.norm(normal_dirs, axis=1, keepdims=True)

        # subsample normal_dirs in each frame
        normal_dirs = normal_dirs[::10]
        all_normal_dirs.append(normal_dirs)

    all_normal_dirs = np.concatenate(all_normal_dirs, axis=0)

    print("number of normal vectors: ", len(all_normal_dirs))

    # run RANSAC to estimate the best aligned global vector.
    # for each sample we compute a normal vectors by averaging from 3 randomly sampled normal vectors.
    # then we compute the best aligned vector by RANSAC.
    best_up_vector = None
    best_inliers = None
    best_score = 0
    for i in trange(2000):
        idx = np.random.choice(len(all_normal_dirs), 1)
        normal_dir = np.mean(all_normal_dirs[idx], axis=0)
        normal_dir = normal_dir / np.linalg.norm(normal_dir)
        # an inlier is a normal vector that is within 1 degrees of the estimated vector
        inlier_deg = 1
        inliers = np.dot(all_normal_dirs, normal_dir) > np.cos(inlier_deg * np.pi / 180)
        score = np.sum(inliers)
        if score > best_score:
            best_score = score
            best_up_vector = normal_dir
            best_inliers = inliers

    print(f'====={hparams.dataset_name}/{hparams.exp_name}=====')
    print(f'best score: {best_score}')
    print(f'best up vector: {best_up_vector}')


if __name__ == '__main__':
    hparams = get_opts()
    estimate_up_vector(hparams)