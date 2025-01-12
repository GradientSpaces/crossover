import torch
from torch import nn

from modules.basic_modules import (_get_clones, calc_pairwise_locs,
                                        get_mlp_head, init_weights)
from .transformers import TransformerSpatialEncoderLayer

class PointTokenizeEncoder(nn.Module):
    def __init__(self, use_attn: bool = True, hidden_size: int =768, path: str =None,
                num_attention_heads: int =12, spatial_dim: int =5, num_layers: int =4, dim_loc: int =6, pairwise_rel_type: str='center'):
        super().__init__()
        
        self.point_mlp = get_mlp_head(hidden_size // 2, hidden_size // 2, hidden_size)    
        self.dropout = nn.Dropout(0.1) 
        
        # build spatial encoder layer
        pc_encoder_layer = TransformerSpatialEncoderLayer(hidden_size, num_attention_heads, dim_feedforward=2048, dropout=0.1, activation='gelu', 
                                                        spatial_dim=spatial_dim, spatial_multihead=True, spatial_attn_fusion='cond')
        self.spatial_encoder = _get_clones(pc_encoder_layer, num_layers)
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = _get_clones(loc_layer, 1)
        self.pairwise_rel_type = pairwise_rel_type
        self.spatial_dim = spatial_dim
        self.use_attn = use_attn
        
        # load weights
        self.apply(init_weights)
        if path is not None:
            self.load_state_dict(torch.load(path), strict=False)
            print('finish load backbone')

    def forward(self, obj_feats: torch.Tensor, obj_locs: torch.Tensor, obj_masks: torch.Tensor) -> torch.Tensor:
        obj_embeds = self.dropout(obj_feats)
        
        # spatial reasoning
        pairwise_locs = calc_pairwise_locs(obj_locs[:, :, :3], obj_locs[:, :, 3:], pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True, spatial_dim=self.spatial_dim)
        for i , pc_layer in enumerate(self.spatial_encoder):
            query_pos = self.loc_layers[0](obj_locs)
            obj_embeds = obj_embeds + query_pos
            if self.use_attn:
                obj_embeds, _ = pc_layer(obj_embeds, pairwise_locs, tgt_key_padding_mask=obj_masks.logical_not())
        
        return obj_embeds

if __name__ == '__main__':
    x = PointTokenizeEncoder(use_pretrained=True, hidden_size=768).cuda()
    obj_feats = torch.ones((10, 10, 384)).float().cuda()
    obj_locs = torch.ones((10, 10, 6)).cuda()
    obj_masks = torch.ones((10, 10)).cuda()
    out = x(obj_feats, obj_locs, obj_masks)
    print(out.shape)