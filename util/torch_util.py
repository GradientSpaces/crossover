import os.path as osp
from safetensors.torch import load_file
import MinkowskiEngine as ME
import numpy as np

def load_weights(model, ckpt_path, device):
    ckpt = osp.join(ckpt_path, 'model.safetensors')
    ckpt = load_file(ckpt,  device = str(device))
    model.load_state_dict(ckpt, strict=False)

def convert_to_sparse_tensor(points, feats, voxel_size=0.02):
    _, sel = ME.utils.sparse_quantize(points / voxel_size, return_index=True)
    coords,  feats = points[sel], feats[sel]
    coords = np.floor(coords / voxel_size)
    coords-= coords.min(0)
    
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    return coords, feats