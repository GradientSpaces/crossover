import einops
import torch.nn as nn
from omegaconf import DictConfig
from typing import Any, Dict, List

from modules.layers.pointnet import PointTokenizeEncoder
from common.constants import ModalityType
from modules.basic_modules import get_mlp_head
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class SceneLevelEncoder(nn.Module):
    def __init__(self, args: DictConfig, modalities: List[str]) -> None:
        super().__init__()

        self.modalities = modalities
        
        self.feat_dims: Dict[str, int] = {}
        self.feat_dims[ModalityType.POINT] = args.point.embed_dim
        self.feat_dims[ModalityType.CAD] = args.cad.embed_dim
        self.feat_dims[ModalityType.RGB] = args.image.embed_dim
        self.feat_dims[ModalityType.REF] = args.referral.embed_dim
        
        self.out_dim = args.out_dim
        
        self.modality_encoders = self._create_modality_encoders()
        self.modality_projections = self._create_modality_projections()
    
    def _create_modality_encoders(self) -> nn.ModuleDict:
        modality_encoders = {}
        
        if 'point' in self.modalities:
            modality_encoders[ModalityType.POINT] = PointTokenizeEncoder(hidden_size=self.feat_dims[ModalityType.POINT])
        if 'cad' in self.modalities:
            modality_encoders[ModalityType.CAD] = PointTokenizeEncoder(use_attn=False, hidden_size=self.feat_dims[ModalityType.CAD])
        
        return nn.ModuleDict(modality_encoders)
    
    def _create_modality_projections(self) -> nn.ModuleDict:
        modality_projections = {}
        if 'point' in self.modalities:
            modality_projections[ModalityType.POINT] = get_mlp_head(self.feat_dims[ModalityType.POINT], self.feat_dims[ModalityType.POINT], self.out_dim)
        if 'rgb' in self.modalities:
            modality_projections[ModalityType.RGB] = get_mlp_head(self.feat_dims[ModalityType.RGB], self.feat_dims[ModalityType.RGB] // 2, self.out_dim)
        if 'cad' in self.modalities:
            modality_projections[ModalityType.CAD] = get_mlp_head(self.feat_dims[ModalityType.CAD], self.feat_dims[ModalityType.CAD], self.out_dim)
        if 'referral' in self.modalities:
            modality_projections[ModalityType.REF] = get_mlp_head(self.feat_dims[ModalityType.REF], self.feat_dims[ModalityType.REF], self.out_dim)
        
        return nn.ModuleDict(modality_projections)
    
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict['embeddings'] = {}
        
        for modality_key in self.modalities:
            modality_value = data_dict['objects']['inputs'][modality_key].float()
            batch_size = modality_value.size()[0]
            
            if modality_key in [ModalityType.POINT, ModalityType.CAD]:
                obj_feats = data_dict['objects']['inputs'][modality_key]
                obj_locs  = data_dict['objects']['object_locs'][modality_key]
                obj_masks = data_dict['masks'][modality_key]
                modality_value = self.modality_encoders[modality_key](obj_feats, obj_locs, obj_masks)
            
            modality_value = self.modality_projections[modality_key](einops.rearrange(modality_value, 'b o d -> (b o) d'))
            modality_value = einops.rearrange(modality_value, '(b o) d -> b o d', b=batch_size)
            
            data_dict['embeddings'][modality_key] = modality_value
        
        return data_dict