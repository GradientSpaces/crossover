import os.path as osp
import numpy as np
from typing import List, Any
from omegaconf import DictConfig

from ..build import DATASET_REGISTRY
from .scanbase import ScanObjectBase, ScanBase

@DATASET_REGISTRY.register()
class ScannetObject(ScanObjectBase):
    """Scannet dataset class for instance level baseline"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)

@DATASET_REGISTRY.register()
class Scannet(ScanBase):
    """Scannet dataset class"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)
        
        filepath = osp.join(self.files_dir, 'scannetv2_{}.txt'.format(self.split))
        self.scan_ids = np.genfromtxt(filepath, dtype = str)
        
        # Ignore Scan IDs because of extra object data
        ignore_scan_ids = {'train': ['scene0515_02', 'scene0673_04', 'scene0673_05', 
                        'scene0101_00', 'scene0101_01', 'scene0218_01',
                        'scene0092_03', 'scene0092_04', 'scene0453_01',
                        'scene0336_01', 'scene0384_00', 'scene0394_00',
                        'scene0331_01', 'scene0054_00'],
                        'val': ['scene0645_00', 'scene0030_00', 'scene0030_02',
                                'scene0208_00', 'scene0231_00', 'scene0231_01',
                                'scene0515_02', 'scene0673_04', 'scene0673_05']}
        
        self.scan_ids = [scan_id for scan_id in self.scan_ids if scan_id not in ignore_scan_ids[split]]        
    
    def get_temporal_scan_pairs(self) -> List[List[Any]]:
        """Gets pairs of temporal scans from the dataset."""
        scene_pairs = []
        
        ref_scan_ids = [scan_id for scan_id in self.scan_ids if scan_id.endswith('00')]
        
        for ref_scan_id in ref_scan_ids:    
            rescan_list = []
            
            for rescan_id in self.scan_ids:
                rescan = {}
                if rescan_id.startswith(ref_scan_id.split('_')[0]) and rescan_id != ref_scan_id:
                    rescan['scan_id'] = rescan_id
                    rescan_list.append(rescan)
            if len(rescan_list) == 0: 
                continue
            
            scene_pairs.append([ref_scan_id, rescan_list])
        return scene_pairs