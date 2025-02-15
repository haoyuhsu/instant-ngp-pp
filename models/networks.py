from venv import create
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh
import numpy as np
from einops import rearrange
from .rendering import NEAR_DISTANCE
from .ref_util import *
    
class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', use_skybox=False, embed_a=False, embed_a_len=12, classes=7):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.use_skybox = use_skybox
        self.embed_a = embed_a
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 17; N_min = 16
        # L = 16; F = 8; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding for spital: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                })
        
        self.xyz_net = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
        )
        self.sigma_act = nn.Softplus()

        # constants
        L_ = 16; F_ = 2; log2_T_ = 19; N_min_ = 16
        # L_ = 16; F_ = 8; log2_T_ = 21; N_min_ = 16
        b_ = np.exp(np.log(2048*scale/N_min_)/(L_-1))
        print(f'GridEncoding for RGB: Nmin={N_min_} b={b_:.5f} F={F_} T=2^{log2_T_} L={L_}')

        self.rgb_encoder = \
            tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": L_,
                "n_features_per_level": F_,
                "log2_hashmap_size": log2_T_,
                "base_resolution": N_min_,
                "per_level_scale": b_,
                "interpolation": "Linear"
            })

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        
        rgb_input_dim = self.rgb_encoder.n_output_dims + self.dir_encoder.n_output_dims
        print(f'rgb_input_dim: {rgb_input_dim}')
        self.rgb_net = \
            tcnn.Network(
                n_input_dims=rgb_input_dim+embed_a_len if embed_a else rgb_input_dim,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": rgb_act,
                    "n_neurons": 128,
                    "n_hidden_layers": 1,
                }
            )
                
        self.norm_pred_header = tcnn.Network(
                    n_input_dims=self.rgb_encoder.n_output_dims, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
        # self.norm_pred_act = nn.Tanh()

        self.semantic_header = tcnn.Network(
                    n_input_dims=self.rgb_encoder.n_output_dims, n_output_dims=classes,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
        self.semantic_act = nn.Softmax(dim=-1)
            
        if use_skybox:
            print("Use skybox!")
            self.skybox_dir_encoder = \
                tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 3,
                    },
                )

            self.skybox_rgb_net = \
                tcnn.Network(
                    n_input_dims=9,
                    n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": rgb_act,
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
            
        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "CutlassMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False, grad=True, grad_feat=True):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)

        with torch.set_grad_enabled(grad):
            h = self.xyz_encoder(x)
            h = self.xyz_net(h)
            sigmas = self.sigma_act(h[:, 0])
        if return_feat: 
            with torch.set_grad_enabled(grad_feat):
                feat_rgb = self.rgb_encoder(x)
                return sigmas, feat_rgb
        return sigmas

    @torch.enable_grad()
    def grad(self, x):
        x = x.requires_grad_(True)
        sigmas, feat_rgb = self.density(x, return_feat=True)
        grads = torch.autograd.grad(
                outputs=sigmas,
                inputs=x,
                grad_outputs=torch.ones_like(sigmas, requires_grad=False).cuda(),
                create_graph=True
                )[0]
        return sigmas, feat_rgb, grads
    
    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb, grads = self.grad(x)
        # check_tensor(grads, 'grads')
        normals_raw = -F.normalize(grads, p=2, dim=-1, eps=1e-6)
        # check_tensor(normals_raw, 'normals_raw')
        
        # up_sem = self.up_label_header(feat_rgb)
        normals_pred = self.norm_pred_header(feat_rgb)
        normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-6)
        # check_tensor(normals_pred, 'normals_pred')

        semantic = self.semantic_header(feat_rgb)
        semantic = self.semantic_act(semantic)
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        d = self.dir_encoder((d+1)/2)

        if self.embed_a:
            embed_a = kwargs['embedding_a']
            if embed_a.size(0) < feat_rgb.size(0):
                repeat = int(feat_rgb.size(0)/embed_a.size(0))
                embed_a = torch.repeat_interleave(embed_a, repeat, 0)
            rgbs = self.rgb_net(torch.cat([d, feat_rgb, embed_a], 1))
        else:
            rgbs = self.rgb_net(torch.cat([d, feat_rgb], 1))
            

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
            
        return sigmas, rgbs, normals_raw, normals_pred, semantic
    
    def forward_test(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb, grads = self.grad(x)
        normals_raw = -F.normalize(grads, p=2, dim=-1, eps=1e-6)
        
        # up_sem = self.up_label_header(feat_rgb)
        with torch.no_grad():
            normals_pred = self.norm_pred_header(feat_rgb)
            normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-6)

            semantic = self.semantic_header(feat_rgb)
            semantic = self.semantic_act(semantic)
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        d = self.dir_encoder((d+1)/2)

        if self.embed_a:
            embed_a = kwargs['embedding_a']
            if embed_a.size(0) < feat_rgb.size(0):
                repeat = int(feat_rgb.size(0)/embed_a.size(0))
                embed_a = torch.repeat_interleave(embed_a, repeat, 0)
            rgbs = self.rgb_net(torch.cat([d, feat_rgb, embed_a], 1))
        else:
            rgbs = self.rgb_net(torch.cat([d, feat_rgb], 1))
            

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
            
        return sigmas, rgbs, normals_pred, normals_raw, semantic
    
    def forward_skybox(self, d):
        if not self.use_skybox:
            return None
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.skybox_dir_encoder((d+1)/2)
        rgbs = self.skybox_rgb_net(d)
        
        return rgbs     

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3) batch transpose
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)

        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)
        
        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
    
    def uniform_sample(self, resolution=128):
        half_grid_size = self.scale / resolution
        samples = torch.stack(torch.meshgrid(
                torch.linspace(0, 1-half_grid_size, resolution),
                torch.linspace(0, 1-half_grid_size, resolution),
                torch.linspace(0, 1-half_grid_size, resolution),
            ), -1).cuda()
        dense_xyz = self.xyz_min * (1-samples) + self.xyz_max * samples
        dense_xyz += half_grid_size*torch.rand_like(dense_xyz).cuda()
        density = self.density(dense_xyz.view(-1,3))
        return density

def check_tensor(tensor, name):
    if torch.any(torch.isnan(tensor)):
        print('{} contains nan'.format(name))
    if torch.any(torch.isinf(tensor)):
        print('{} contains inf'.format(name))