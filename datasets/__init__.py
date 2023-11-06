from datasets.tnt import tntDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .tnt import tntDataset
from .nerf import NeRFDataset
from .nerfpp import NeRFPPDataset
from .kitti360 import KittiDataset
from .mega_nerf.dataset import MegaDataset
from .highbay import HighbayDataset

from .mipnerf360 import MipNeRF360Dataset
from .lerf import LeRFDataset

dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'tnt': tntDataset,
                'nerfpp': NeRFPPDataset,
                'kitti': KittiDataset,
                'mega': MegaDataset,
                'highbay': HighbayDataset,
                'mega': MegaDataset,
                'mipnerf360': MipNeRF360Dataset,
                'lerf': LeRFDataset,
}
