import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Any, Dict
 
from common import load_utils
from util import point_cloud
from modules.build import build_module

class Base3DProcessor:
    """Base 3D feature (point cloud + CAD) processor class."""
    def __init__(self, config_data: DictConfig, config_3D: DictConfig, split: str) -> None:
        load_utils.set_random_seed(42)
        # get device 
        if not torch.cuda.is_available(): 
            raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        self.config_3D = config_3D
        
        # load feature extractor
        self.feature_extractor = self.loadFeatureExtractor(config_3D, "3D")
    
    def loadFeatureExtractor(self, config: DictConfig, modality: str) -> torch.nn.Module:
        """Loads and initializes the feature extractor model."""
        model = build_module(modality, config.feature_extractor.model)
        state_dict = torch.load(config.feature_extractor.ckpt, map_location='cpu')
        state_dict = {k.replace('point_encoder.', '') : v for k, v in state_dict.items() if k.startswith('point_encoder')}
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(self.device)
        
        return model

    def normalizeObjectPCLAndExtractFeats(self, points: np.ndarray) -> Dict[str, Any]:
        """Normalizes the object point cloud and extracts features."""
        sampled_points = point_cloud.sample_and_normalize_pcl(points)
        object_loc, object_box = point_cloud.get_object_loc_box(points)
        points_pt = [torch.from_numpy(sampled_points)]
        points_pt = torch.stack(points_pt, dim=0).to(self.device).float()
        
        with torch.no_grad():
            feats = self.feature_extractor(points_pt).cpu().numpy()
        
        return {'points': sampled_points, 'feats': feats, 
                'loc' : object_loc, 'box' : object_box}
    
    def compute3DFeatures(self) -> None:
        """Computes 3D features for all scans in the dataset."""
        for scan_id in tqdm(self.scan_ids):
            self.compute3DFeaturesEachScan(scan_id)
    
    def run(self) -> None:
        """Execute the complete 3D feature extraction pipeline."""
        self.compute3DFeatures()