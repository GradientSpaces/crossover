import numpy as np
import torch
from omegaconf import DictConfig
from typing import List

from common import load_utils
from modules.build import build_module

class Base2DProcessor:
    """Base 2D feature processor class."""
    def __init__(self, config_data: DictConfig, config_2D: DictConfig, split: str) -> None: 
        load_utils.set_random_seed(42)   
        # get device 
        if not torch.cuda.is_available(): 
            raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
         
        self.model = build_module("2D", config_2D.feature_extractor.model, 
                                 ckpt =  config_2D.feature_extractor.ckpt, device = self.device)

    def extractFeatures(self, images: List[torch.tensor], return_only_cls_mean: bool = True) -> np.ndarray:
        """Extract 2D features from a list of images."""
        image_input = torch.stack(images).to(self.device) 
        with torch.no_grad(): 
            ret = self.model.feature_extractor(image_input) 
            cls_token = ret[:, 0, :]
        if return_only_cls_mean:
            return cls_token.mean(dim=0).cpu().detach().numpy().reshape(1, -1)
        else:
            return ret.cpu().detach().numpy()
    
    def run(self) -> None:
        """Execute the complete 2D feature extraction pipeline."""
        self.compute2DFeatures()