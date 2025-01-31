import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from typing import List, Dict, Union, Optional

from common import load_utils
from third_party.BLIP.models.blip import blip_feature_extractor

class Base1DProcessor:
    """Base 1D feature (relationships) processor class."""
    def __init__(self, config_data: DictConfig, config_1D: DictConfig, split: str) -> None:
        # get device 
        load_utils.set_random_seed(42)
        if not torch.cuda.is_available(): 
            raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        self.config_1D = config_1D
        self.embed_dim = self.config_1D.feature_extractor.embed_dim
        
        model_url = config_1D.feature_extractor.model_path
        self.model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='large').to(self.device)
        self.model.eval()
    
    def extractTextFeats(self, texts: List[str], return_text: bool = False) -> Optional[Union[List[Dict[str, Union[str, np.ndarray]]], np.ndarray]]:
        text_feats = []
        
        for text in texts:
            encoded_text = self.model.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)  
            if encoded_text['input_ids'].shape[1] > 512: 
                continue
            
            with torch.no_grad():
                encoded_text = self.model.text_encoder(encoded_text.input_ids, attention_mask = encoded_text.attention_mask,                      
                                                return_dict = True, mode = 'text').last_hidden_state[:, 0].cpu().numpy().reshape(1, -1)
                
            text_feats.append({'text' : text, 'feat' : encoded_text})
        
        if len(text_feats) == 0:
            return None
        
        if return_text:
            return text_feats
         
        text_feats = [text_feat['feat'] for text_feat in text_feats]
        text_feats = np.concatenate(text_feats)
        return text_feats
    
    def compute1DFeatures(self) -> None:
        for scan_id in tqdm(self.scan_ids):
            self.compute1DFeaturesEachScan(scan_id)
    
    def run(self) -> None:
        """Execute the complete 1D feature extraction pipeline."""
        self.compute1DFeatures()