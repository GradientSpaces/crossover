import os.path as osp
import numpy as np
import csv
import torch
from typing import List, Dict, Any, Union
from omegaconf import DictConfig
from common import load_utils
from ..build import DATASET_REGISTRY
from .scanbase import ScanObjectBase, ScanBase
from ..transform_utils import inverse, rotation_error, translation_error

# Scan3R object categories used in LivingScenes evaluation
RIO_CATE = [
    "dinning chair", "rocking chair", "armchair", "chair", # chair
    "couching table", "dining table", "computer desk", "round table", "side table", "stand", "desk", "coffee table", # table
    "bench", # bench
    "sofa", "sofa chair", "couch", "ottoman", "footstool", # sofa
    "cushion", "pillow", # pillow
    "bed", # bed
    "trash can", # trash bin
]

@DATASET_REGISTRY.register()
class Scan3RObject(ScanObjectBase):
    """Scan3R dataset class for instance level baseline"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)

@DATASET_REGISTRY.register()
class Scan3R(ScanBase):
    """Scan3R dataset class"""
    def __init__(self, data_config: DictConfig, split: str) -> None:
        super().__init__(data_config, split)
        filepath = osp.join(self.files_dir, '{}_scans.txt'.format(self.split))
        self.scan_ids = np.genfromtxt(filepath, dtype = str)
        
        self.label_map = self.read_label_map(osp.join(self.files_dir, '3RScan.v2 Semantic Classes - Mapping.csv'))
        self.label_map_name = self.read_label_map(osp.join(self.files_dir, '3RScan.v2 Semantic Classes - Mapping.csv'), label_from='Global ID', label_to='Label')
        
        
        f = open(osp.join(self.files_dir, f'{split}.txt'), "r")
        self.ref_scans = f.read().splitlines()
        
        self.scene_json = load_utils.load_json(osp.join(self.files_dir, '3RScan.json'))
        self.scene_json = [ scene for scene in self.scene_json if scene['reference'] in self.ref_scans]
    
    def get_temporal_scan_pairs(self) -> List[List[Any]]:
        """Returns pairs of scans with temporal relationships and object motion information."""
        scene_pairs = []
        for scene_json_data in self.scene_json:
            ref_scan_id = scene_json_data['reference']
            ambiguity = scene_json_data['ambiguity']
            rescan_list = []
            for scan in scene_json_data['scans']:
                if 'reference' not in scan:
                    continue
                
                if 'transform' not in scan:
                    continue
                
                rescan = {} 
                rescan['scan_id'] = scan['reference']
                
                scene_tsfm = torch.Tensor(scan['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)
                moving_id_lst = []
                static_id_lst = []
                refscan_id_lst = []
                for rigid in scan['rigid']:
                    obj_tsfm = inverse(torch.Tensor(rigid['transform']).cuda().reshape(1, 4, 4).transpose(-1, -2)) # obj from rescan to ref
                    rot_diff = rotation_error(obj_tsfm[:,:3,:3], scene_tsfm[:,:3,:3])
                    t_diff = translation_error(obj_tsfm[:,:3,3], scene_tsfm[:,:3,3])
                    if rot_diff>1 or t_diff > 0.05:
                        moving_id_lst.append(rigid['instance_reference'])
                    else: 
                        static_id_lst.append(rigid['instance_reference'])
                    
                    refscan_id_lst.append(rigid['instance_reference'])
                    
                rescan['moving_ids'] = moving_id_lst
                rescan['static_ids'] = static_id_lst
                rescan['refscan_ids'] = refscan_id_lst
                rescan['rescan2ref_tsfm'] = scene_tsfm
                rescan_list.append(rescan)
            
            scene_pairs.append([ref_scan_id, ambiguity, rescan_list])
        
        return scene_pairs

    def read_label_map(self, file_name: str, label_from: str = 'Global ID', 
                      label_to: str = 'NYU40ID') -> Dict[Union[str, int], str]:
        """Reads CSV mapping file for label conversion."""
        assert osp.exists(file_name)
        
        mapping = dict()
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                key = row[label_from].strip()  # Ensure any spaces are stripped
                value = row[label_to].strip()
                mapping[key] = value
        
        if int(list(mapping.keys())[0]):
            mapping = {int(k):v for k,v in mapping.items()}

        return mapping

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene_dict = super().__getitem__(index)
        
        # scene_dict['label_ids'] = scene_dict['label_ids'].numpy()
        # scene_dict['label_ids_rio_subset_mask'] = np.zeros((scene_dict['label_ids'].shape[0], ))
        
        # # For 3RScan temporal evaluation
        # for idx, label_id in enumerate(scene_dict['label_ids']):
        #     if label_id in self.label_map_name and self.label_map_name[label_id] in RIO_CATE:
        #         scene_dict['label_ids_rio_subset_mask'][idx] = 1.0
        
        # for idx, label_id in enumerate(scene_dict['label_ids']):
        #     if label_id == -100: 
        #         continue
        #     scene_dict['label_ids'][idx] = self.label_map[label_id]
        
        # scene_dict['label_ids'] = torch.from_numpy(scene_dict['label_ids'])
        # scene_dict['label_ids_rio_subset_mask'] = torch.from_numpy(scene_dict['label_ids_rio_subset_mask'])
        
        return scene_dict