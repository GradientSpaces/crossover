import os.path as osp
import os
import numpy as np
from typing import Any, Dict, Tuple, List
from tqdm import tqdm

from common import load_utils
from utils import scannet, labelmap

def get_scan_load_files(scans_dir: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    load_files = {}
    scan_ids = os.listdir(scans_dir)
    
    for scan_id in scan_ids:
        scene_folder = osp.join(scans_dir, scan_id)
        mesh_file = osp.join(scene_folder, scan_id + '_vh_clean_2.labels.ply')
        aggre_file = osp.join(scene_folder, scan_id + '_vh_clean.aggregation.json')
        seg_file = osp.join(scene_folder, scan_id + '_vh_clean_2.0.010000.segs.json')
        meta_file = osp.join(scene_folder, scan_id + '.txt')
        
        load_files[scan_id] = {}
        load_files[scan_id]['mesh'] = mesh_file
        load_files[scan_id]['aggre'] = aggre_file
        load_files[scan_id]['seg']  = seg_file
        load_files[scan_id]['meta'] = meta_file

    return scan_ids, load_files

def convertObjectData(scan_ids: List[str], load_files: Dict[str, Dict[str, str]], out_dir: str) -> None:
    object_data = {"scans": []}
    for scan_id in tqdm(scan_ids):
        object_data['scans'].append(convertObjectDataEachScan(scan_id, load_files[scan_id]))
    
    load_utils.save_json(osp.join(out_dir, 'objects.json'), object_data)
        
            
def convertObjectDataEachScan(self, scan_id: str, load_file: Dict[str, str]) -> Dict[str, Any]:
    mesh_file = load_file['mesh']
    aggre_file = load_file['aggre'] 
    seg_file = ['seg']
    meta_file = ['meta']
    
    _, _, instance_ids, _, object_id_to_label_id, _ \
                                                            = scannet.export(mesh_file, aggre_file, seg_file, meta_file, self.label_map, 
                                                                            axis_alignment = True, output_file=None)
    unique_instance_ids = np.unique(instance_ids)
    
    object_data = {}
    object_data['scan'] = scan_id
    object_data['objects'] = []
    for instance_id in unique_instance_ids:
        if instance_id == 0: 
            continue
        
        object_data['objects'].append({ 'id' : str(instance_id), 'global_id' : str(object_id_to_label_id[instance_id]), 
                                        'label' : labelmap.NYU40_Label_Names[object_id_to_label_id[instance_id] - 1]})
        
    return object_data

if __name__ == '__main__':
    base_dataset_dir = ''
    scans_dir = osp.join(base_dataset_dir, 'scans')
    files_dir = osp.join(base_dataset_dir, 'files')
    
    scan_ids, load_files = get_scan_load_files(scans_dir)
    convertObjectData(scan_ids, load_files, files_dir)