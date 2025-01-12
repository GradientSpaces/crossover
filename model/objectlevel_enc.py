import torch.nn as nn
from typing import Dict, Any, List
from omegaconf import DictConfig

from common.constants import ModalityType
from modules.basic_modules import get_mlp_head
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ObjectLevelEncoder(nn.Module):
    def __init__(self, args: DictConfig, modalities: List[str]) -> None:
        super().__init__()

        self.modalities: List[str] = modalities
        
        self.feat_dims: Dict[str, int] = {}
        self.feat_dims[ModalityType.POINT] = args.point.embed_dim
        self.feat_dims[ModalityType.CAD] = args.cad.embed_dim
        self.feat_dims[ModalityType.RGB] = args.image.embed_dim
        self.feat_dims[ModalityType.REF] = args.referral.embed_dim
        
        self.out_dim: int = args.out_dim
        
        self.modality_projections: nn.ModuleDict = self._create_modality_projections()
    
    def _create_modality_projections(self) -> nn.ModuleDict:
        modality_projections: Dict[str, nn.Module] = {}
        if 'point' in self.modalities:
            modality_projections[ModalityType.POINT] = get_mlp_head(self.feat_dims[ModalityType.POINT], self.feat_dims[ModalityType.POINT], self.out_dim)
        if 'rgb' in self.modalities:
            modality_projections[ModalityType.RGB] = get_mlp_head(self.feat_dims[ModalityType.RGB], self.feat_dims[ModalityType.RGB], self.out_dim)
        if 'cad' in self.modalities:
            modality_projections[ModalityType.CAD] = get_mlp_head(self.feat_dims[ModalityType.CAD], self.feat_dims[ModalityType.CAD], self.out_dim)
        if 'referral' in self.modalities:
            modality_projections[ModalityType.REF] = get_mlp_head(self.feat_dims[ModalityType.REF], self.feat_dims[ModalityType.REF], self.out_dim)
        
        return nn.ModuleDict(modality_projections)
    
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict['embeddings'] = {} 
        for modality in self.modalities:
            modality_value = data_dict['inputs'][modality].float()
            modality_value = self.modality_projections[modality](modality_value)
            
            data_dict['embeddings'][modality] = modality_value
            
        return data_dict