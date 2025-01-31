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
from common import misc

import os.path as osp
import numpy as np

@EVALUATION_REGISTRY.register()
class SingleRGBSceneRetrieval():
    def __init__(self, cfg) -> None:
        super().__init__()
        
        set_seed(cfg.rng_seed)
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        
        task_config = cfg.task.get(cfg.task.name)
                
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
    def inference_step(self):
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
                output['outputs']['object'] = data_dict['embeddings']['referral'][idx]
                output['masks']['object'] = data_dict['scene_masks']['referral'][idx]
                outputs.append(output)   
                      
            pbar.update(1)

        return outputs
            
    def eval(self, output_dict):
        resplit_scan_ids = np.genfromtxt('/home/sayan/Downloads/test_resplit_scans.txt', dtype = str)
        
        temporal_scan_pairs = self.data_loader.dataset.get_temporal_scan_pairs()
        split_scan_ids = []
        
        for scene_pair in temporal_scan_pairs:
            ref_scan_id, _, rescan_list = scene_pair[0], scene_pair[1], scene_pair[2]
            if ref_scan_id not in resplit_scan_ids: 
                continue
            
            if len(rescan_list) == 0 or ref_scan_id is None: 
                continue
            
            split_scan_ids.append(ref_scan_id)
            for rescan in rescan_list:
                rescan_id = rescan['scan_id']
                split_scan_ids.append(rescan_id)
        
        
        scan_data = np.array([{ 'scan_id': output_data['scan_id'], 'label' : output_data['scene_label']} for output_data in output_dict])
        unique_labels = {data['label'] for data in scan_data}
        
        scan_embeds = torch.stack([output_data['outputs']['object'] for output_data in output_dict])
        scan_embed_masks = torch.stack([output_data['masks']['object'] for output_data in output_dict])
        
        recall_top1 = 0
        recall_top3 = 0
        recall_top5 = 0
        
        total = 0
        for idx, data in tqdm(enumerate(scan_data)):
            if data['scan_id'] not in split_scan_ids:
                continue
            
            scan_image_embeddings_path = osp.join(self.data_loader.dataset.process_dir, data['scan_id'], 'data2D_all_images.pt')
            
            scan_image_embeddings = torch.load(scan_image_embeddings_path)
            scan_image_embeddings = torch.from_numpy(scan_image_embeddings['scene']['scene_embeddings'])
            scan_image_embeddings = torch.concatenate([scan_image_embeddings[:, 0, :], scan_image_embeddings[:, 1:, :].mean(dim=1)], dim=1)
        
            for image_idx, image_embed in enumerate(scan_image_embeddings):
                image_embed = image_embed.to(self.accelerator.device)
                image_embed = self.model.frame_mlp(image_embed.unsqueeze(0))
                image_embed = self.model.encoder2D_mlp_head(image_embed)
                
                random_indices = np.random.choice(np.delete(np.arange(scan_embeds.shape[0]), idx), 49, replace=False)
                selected_embeds = torch.cat([scan_embeds[idx].unsqueeze(0), scan_embeds[random_indices]], dim=0)

                sim = torch.softmax(image_embed.to(scan_embeds.device) @ selected_embeds.t(), dim=-1)
                rank_list = torch.argsort(1.0 - sim, dim = 1)
                
                if rank_list[0][0] == 0:
                    recall_top1 += 1
                
                if 0 in rank_list[0][:3]:
                    recall_top3 += 1
                
                if 0 in rank_list[0][:5]:
                    recall_top5 += 1
                
                total += 1 
        
        recall_top1 = recall_top1 / total * 100.
        recall_top3 = recall_top3 / total * 100.
        recall_top5 = recall_top5 / total * 100.
        
        message = f'top1 - {recall_top1:.2f}, top3 - {recall_top3:.2f}, top10 - {recall_top5:.2f}'
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
        # Inference Step
        output_dict = self.inference_step()
        
        # Evaluation
        self.eval(output_dict)