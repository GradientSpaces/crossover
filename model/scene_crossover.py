import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List
import MinkowskiEngine as ME

from third_party.BLIP.models.blip import blip_feature_extractor
from modules.layers.patch_encoder import Mlps
from modules.layers.sparse_conv_encoder import ResNet34
from modules.basic_modules import get_mlp_head
from modules.encoder2D.dinov2 import DinoV2

class SceneCrossOverModel(nn.Module):
    def __init__(self, args: DictConfig, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device        
        self.out_dim = args.out_dim
        self.encoder3D = ResNet34(3, D=3)
        self.encoder3D_mlp_head = get_mlp_head(args.input_dim_3d, args.input_dim_3d, args.out_dim)
        
        blip_model = blip_feature_extractor(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth', 
                                            image_size=224, vit='large').to(self.device)
        self.encoder1D = blip_model.text_encoder
        self.tokenizer = blip_model.tokenizer
        
        self.encoder1D_mlp_head = get_mlp_head(args.input_dim_1d, args.input_dim_1d, args.out_dim)
        
        self.encoder2D = DinoV2('dinov2_vitg14', self.device).feature_extractor
        self.frame_mlp = Mlps(args.input_dim_2d * 2, [args.input_dim_2d], args.input_dim_2d)
        self.encoder2D_mlp_head = get_mlp_head(args.input_dim_2d, args.input_dim_2d // 2, args.out_dim)        
    
    def encode_3d(self, coordinates, features) -> torch.Tensor:
        pcl_sparse = ME.SparseTensor(
                        coordinates=coordinates,
                        features=features.to(torch.float32),
                        device=self.device)
        
        scene_feats_pc = self.encoder3D(pcl_sparse)
        scene_feats_pc = self.encoder3D_mlp_head(scene_feats_pc)
        return scene_feats_pc
    
    def encode_rgb(self, images: torch.Tensor) -> torch.Tensor:
        rgb_embedding = self.encoder2D(images[0]) 
        rgb_embedding = torch.concatenate([rgb_embedding[:, 0, :], rgb_embedding[:, 1:, :].mean(dim=1)], dim=1)
        rgb_embedding = rgb_embedding[list(range(0, rgb_embedding.shape[0])), :].unsqueeze(0)
        rgb_embedding = self.frame_mlp(rgb_embedding)
        rgb_embedding = rgb_embedding.mean(dim=1)
        rgb_embedding = self.encoder2D_mlp_head(rgb_embedding)
        
        return rgb_embedding
    
    def encode_floorplan(self, floorplan: torch.Tensor) -> torch.Tensor:
        floorplan_embedding = self.encoder2D(floorplan)
        floorplan_embedding = floorplan_embedding[:, 0]
        floorplan_embedding = self.encoder2D_mlp_head(floorplan_embedding)
        
        return floorplan_embedding
    
    def encode_1d(self, texts: List[torch.Tensor]) -> torch.Tensor:
        text_embedding = None
        for text in texts:
            encoded_text = self.tokenizer(text, padding=True, add_special_tokens=True, return_tensors="pt").to(self.device)  
            if encoded_text['input_ids'].shape[1] > 512: 
                continue
            
            encoded_text = self.encoder1D(encoded_text.input_ids, attention_mask = encoded_text.attention_mask,                      
                                                return_dict = True, mode = 'text').last_hidden_state[:, 0]
                
            text_embedding = encoded_text if text_embedding is None else torch.cat((text_embedding, encoded_text), dim=0)

        
        text_embedding = text_embedding[0, :].unsqueeze(0)
        text_embedding = self.encoder1D_mlp_head(text_embedding)
        
        return text_embedding      

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        embedding_dict = { 'embeddings': {}, 'masks':  {}}    
        
        embedding_dict['embeddings']['point'] = self.encode_3d(data_dict['coordinates'], data_dict['features'])
        embedding_dict['embeddings']['rgb'], embedding_dict['embeddings']['floorplan'] = self.encode_rgb(data_dict['rgb'].to(self.device)), self.encode_floorplan(data_dict['floorplan'].to(self.device))
        embedding_dict['embeddings']['referral'] = self.encode_1d(data_dict['referral'])
        embedding_dict['masks'] = data_dict['masks']

        return embedding_dict