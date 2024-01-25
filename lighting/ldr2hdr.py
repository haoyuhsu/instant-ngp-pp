import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from PIL import Image

# TODO: better way to fix the import error?
sys.path.append(os.path.join(os.getcwd(), 'lighting'))
from sritmo.global_sritmo import SRiTMO


def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)


@torch.no_grad()
def convert_ldr2hdr(ldr_env_map_path: str):

    params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sritmo': '/home/max/Documents/maxhsu/instant-ngp-pp/lighting/text2light_released_model/sritmo.pth',
        'sr_factor': 2,
    }

    # load LDR panorama
    ldr_panorama = np.array(Image.open(ldr_env_map_path))

    xsample = torch.Tensor(ldr_panorama).permute(2, 0, 1).unsqueeze(0).to(params['device'])
    xsample = xsample / 127.5 - 1.0

    # super-resolution inverse tone mapping
    ldr_hr_samples, hdr_hr_samples = SRiTMO(xsample, params)

    # save results
    hdr_env_map_path = ldr_env_map_path[:-4] + ".exr"
    for i in range(xsample.shape[0]):
        cv2.imwrite(hdr_env_map_path, hdr_hr_samples[i].permute(1, 2, 0).detach().cpu().numpy())

    return hdr_env_map_path


@torch.no_grad()
def ldr2hdr_test(outdir: str, params: dict):

    # load LDR panorama
    ldr_panorama = np.array(Image.open(params['ldr_path']))

    xsample = torch.Tensor(ldr_panorama).permute(2, 0, 1).unsqueeze(0).to(params['device'])
    xsample = xsample / 127.5 - 1.0

    # super-resolution inverse tone mapping
    if params['sritmo'] is not None:
        ldr_hr_samples, hdr_hr_samples = SRiTMO(xsample, params)
    else:
        print("no checkpoint provided, skip Stage II (SR-iTMO)...")
        return
    
    filename = os.path.basename(params['ldr_path']).split('.')[0]
    for i in range(xsample.shape[0]):
        # cv2.imwrite(os.path.join(outdir, "ldr", "hrldr_[{}].png".format(filename)), (ldr_hr_samples[i].permute(1, 2, 0).detach().cpu().numpy() + 1) * 127.5)
        cv2.imwrite(os.path.join(outdir, "{}.exr".format(filename)), hdr_hr_samples[i].permute(1, 2, 0).detach().cpu().numpy())


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sritmo",
        type=str,
        nargs="?",
        default="/home/max/Documents/maxhsu/instant-ngp-pp/lighting/text2light_released_model/sritmo.pth",
        help="load super-resolution inverse tone mapping operator from the given path.",
    )
    parser.add_argument(
        "--sr_factor",
        type=int,
        nargs="?",
        default=2,
        help="upscaling factor for super-resolution."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="output directory.",
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=4,
        help="batch size. Tune it according to your GPU capacity.",
    )
    parser.add_argument(
        '--ldr_path',
        type=str,
        default='./test_images/ldr_teatime.png',
        help="path to the LDR panorama.",
    )
    return parser


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    gpu = True
    eval_mode = True
    show_config = False

    base = list()

    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)
    print("Writing samples to ", outdir)
    # for k in ["ldr", "hdr"]:
    #     os.makedirs(os.path.join(outdir, k), exist_ok=True)

    input_params = {
        'device': 'cuda' if gpu else 'cpu',
        'sritmo': opt.sritmo,
        'sr_factor': opt.sr_factor,
        'ldr_path': opt.ldr_path,
    }

    ldr2hdr_test(outdir, input_params)
