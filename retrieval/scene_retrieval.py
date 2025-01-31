from safetensors.torch import load_file
import torch
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
import MinkowskiEngine as ME
from tqdm import tqdm

from data.build import build_dataloader
from model.build import build_model
from .build import EVALUATION_REGISTRY
from evaluator import eval_utils
from common import misc

import os.path as osp
import itertools
import numpy as np

@EVALUATION_REGISTRY.register()
class SceneRetrieval():
    def __init__(self, cfg) -> None:
        super().__init__()
        
        set_seed(cfg.rng_seed)
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        
        task_config = cfg.task.get(cfg.task.name)
        
        self.modalities = task_config.scene_modalities
        self.modalities.append('object')
        
        modality_combinations = list(itertools.combinations(self.modalities, 2))
        same_element_combinations = [(item, item) for item in self.modalities]
        modality_combinations.extend(same_element_combinations)
        self.modality_combinations = modality_combinations
                
        key = "val"
        self.data_loader = build_dataloader(cfg, split=key, is_training=False)
        self.dataset_name = misc.rgetattr(task_config, key)[0]
        
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        
        # Accelerator preparation
        self.model = build_model(cfg)
        self.model, self.data_loader = self.accelerator.prepare(self.model, self.data_loader)
        
        # Load from ckpt
        self.ckpt_path = Path(task_config.ckpt_path)
        self.load_from_ckpt()
    
    def forward(self, data_dict):
        return self.model(data_dict)
    
    @torch.no_grad()
    def inference_step_semantics(self):
        self.model.eval()
        
        loader = self.data_loader
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        self.logger.info('Running validation...')
        
        outputs = []
        for iter, data_dict in enumerate(loader):
            data_dict['pcl_sparse'] = ME.SparseTensor(
                    coordinates=data_dict['coordinates'],
                    features=data_dict['features'].to(torch.float32),
                    device=self.accelerator.device)
            
            data_dict = self.forward(data_dict)
            num_scans = len(data_dict['scan_id'])
            for idx in range(num_scans):
                output = { 'scan_id' : data_dict['scan_id'][idx], 'scene_label': data_dict['scene_label'][idx], 'outputs' : {}, 'masks' : {}}
                
                point_mask = data_dict['masks']['point'][idx]
                rgb_mask   = data_dict['masks']['rgb'][idx]
                referral_mask   = data_dict['masks']['referral'][idx]
                
                output['masks']['point'] = data_dict['scene_masks']['point'][idx]
                output['masks']['rgb'] = data_dict['scene_masks']['rgb'][idx]
                output['masks']['floorplan'] = data_dict['scene_masks']['floorplan'][idx]
                output['masks']['referral'] = data_dict['scene_masks']['referral'][idx]
                
                output['outputs']['point'] = torch.mean(data_dict['object_modality_embeddings']['point'][idx][point_mask.to(torch.bool)], dim=0).reshape(-1,)
                output['outputs']['rgb'] = torch.mean(data_dict['object_modality_embeddings']['rgb'][idx][rgb_mask.to(torch.bool)], dim=0).reshape(-1,)
                output['outputs']['floorplan'] = data_dict['embeddings']['floorplan'][idx].reshape(-1,)
                output['outputs']['referral'] = torch.mean(data_dict['object_modality_embeddings']['referral'][idx][referral_mask.to(torch.bool)], dim=0).reshape(-1,)
                outputs.append(output)   
                      
            pbar.update(1)

        return outputs
    
    @torch.no_grad()
    def inference_step_no_semantics(self):
        self.model.eval()
        
        loader = self.data_loader
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        self.logger.info('Running validation...')
        
        outputs = []
        for iter, data_dict in enumerate(loader):
            data_dict['pcl_sparse'] = ME.SparseTensor(
                    coordinates=data_dict['coordinates'],
                    features=data_dict['features'].to(torch.float32),
                    device=self.accelerator.device)
            
            data_dict = self.forward(data_dict)
            
            num_scans = len(data_dict['scan_id'])
            
            for idx in range(num_scans):
                output = { 'scan_id' : data_dict['scan_id'][idx], 'scene_label': data_dict['scene_label'][idx], 'outputs' : {}, 'masks' : {}}
                for modality in self.modalities:                                
                    output['outputs'][modality] = data_dict['embeddings'][modality][idx]
                    output['masks'][modality] = data_dict['scene_masks'][modality][idx]      
                outputs.append(output)             
            pbar.update(1)

        return outputs
 
    def eval(self, output_dict):
        scan_data = np.array([{ 'scan_id': output_data['scan_id'], 'label' : output_data['scene_label']} for output_data in output_dict])
        unique_labels = {data['label'] for data in scan_data}

        for src_modality, ref_modality in self.modality_combinations:
            
            src_embed = torch.stack([output_data['outputs'][src_modality] for output_data in output_dict])
            ref_embed = torch.stack([output_data['outputs'][ref_modality] for output_data in output_dict])
        
            src_mask  = torch.stack([output_data['masks'][src_modality] for output_data in output_dict])
            ref_mask  = torch.stack([output_data['masks'][ref_modality] for output_data in output_dict])
            mask = torch.logical_and(src_mask, ref_mask).reshape(-1,)
            
            if mask.sum() == 0:
                continue
            
            src_embed = src_embed[mask]
            ref_embed = ref_embed[mask]
            
            scan_data_masked = scan_data[mask.cpu().numpy()]
            
            sim = torch.softmax(src_embed @ ref_embed.t(), dim=-1)
            rank_list = torch.argsort(1.0 - sim, dim = 1)
            top_k_indices = rank_list[:, :20]
                
            correct_index = torch.arange(src_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
            matches = top_k_indices == correct_index   
            
            recall_top1 = matches[:, 0].float().mean() * 100.
            recall_top5 = matches[:, :5].any(dim=1).float().mean() * 100.
            recall_top10 = matches[:, :10].any(dim=1).float().mean() * 100.
            recall_top20 = matches.any(dim=1).float().mean() * 100.
            
            message = f"{src_modality} -> {ref_modality}:" 
            self.logger.info(message)
            
            message = f'Recall: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}, top20 - {recall_top20:.2f}'
            self.logger.info(message)
            
            # Temporal eval
            recall_top1, recall_top5, recall_top10 = eval_utils.evaluate_temporal_scene_matching(rank_list.cpu().numpy().tolist(), scan_data_masked, self.data_loader.dataset.get_temporal_scan_pairs())
            message = f'Temporal: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}'
            self.logger.info(message)  
            
            if self.dataset_name == 'Scannet':
                st_recall_top1, st_recall_top5, st_recall_top10 = eval_utils.calculate_scene_label_recall(rank_list.cpu().numpy().tolist(), scan_data_masked)
                st_recall_top1 *= 100.
                st_recall_top5 *= 100.
                st_recall_top10 *= 100.
                
                message  =  f"Category: top1 - {st_recall_top1:.2f}, top5 - {st_recall_top5:.2f}, top10 - {st_recall_top10:.2f}"
                self.logger.info(message)     
                
                ic_recall_top1, ic_recall_top3, ic_recall_top5 = eval_utils.evaluate_intra_category_scene_matching(scan_data_masked, src_embed, ref_embed, unique_labels)
                message  =  f"Intra-Category: top1 - {ic_recall_top1:.2f}, top3 - {ic_recall_top3:.2f}, top5 - {ic_recall_top5:.2f}"
                self.logger.info(message)  
                   
    def load_from_ckpt(self):
        if self.ckpt_path.exists():
            self.logger.info(f"Loading from {self.ckpt_path}")
            # Load model weights from safetensors files
            ckpt = osp.join(self.ckpt_path, 'model.safetensors')
            ckpt = load_file(ckpt,  device = str(self.accelerator.device))
            self.model.load_state_dict(ckpt)
            self.logger.info(f"Successfully loaded from {self.ckpt_path}")
        
        else:
            raise FileNotFoundError
    
    def run(self):
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Inference Step -- CrossOver
        output_dict = self.inference_step_no_semantics()
        self.logger.info('Scene Retrieval Evaluation (CrossOver)...')
        self.eval(output_dict)
        
        # Inference Step -- Instance CrossOver (for scene retrieval)
        output_dict = self.inference_step_semantics()
        self.logger.info('Scene Retrieval Evaluation (Instance CrossOver)...')
        self.eval(output_dict)