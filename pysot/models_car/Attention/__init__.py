
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
# from torch import nn
import paddle
from paddle import nn
# from timm.models.layers import DropPath

from .self_attention_block import SelfAttentionBlock
from .cross_attention_block import CrossAttentionBlock



# from pysot.models.Attention.model import SelfAttentionBlock
#
# SelfAttentionBlock = {
#          'SelfAttentionBlock': SelfAttentionBlock,
#         }

class FUSION(nn.Layer):
    # def __init__(self,  encoder, decoder):
    def __init__(self, encoder):        # 11/23;10:58
        
        super(FUSION, self).__init__()
        self.layers1 = nn.LayerList(encoder)
        # self.layers2 = nn.ModuleList(decoder)             # 11/23;10:55

    def forward(self, x_1, xpos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''     
        B_1, C_1, W_1, H_1 = x_1[0].shape
        x_1[0] = x_1[0].reshape([B_1, C_1, -1]).transpose([0, 2, 1])
        x_1[1] = x_1[1].reshape([B_1, C_1, -1]).transpose([0, 2, 1])
        x_1[2] = x_1[2].reshape([B_1, C_1, -1]).transpose([0, 2, 1])
        x_cat_1 = paddle.concat((x_1[0], x_1[1], x_1[2]), axis=1)

        # B_2, C_2, W_2, H_2 = x_2[0].shape
        # x_2[0] = x_2[0].reshape(B_2, C_2, -1).permute(0, 2, 1)
        # x_2[1] = x_2[1].reshape(B_2, C_2, -1).permute(0, 2, 1)
        # x_2[2] = x_2[2].reshape(B_2, C_2, -1).permute(0, 2, 1)
        # x_cat_2 = torch.cat((x_2[0], x_2[1], x_2[2]), dim=1)

        #---------------------------------------------------------------------------------------------------------------
        # add_template = (x_1[0] + x_1[1] + x_1[2]) / 3
        # add_search = (x_2[0] + x_2[1] + x_2[2]) / 3
        # add_template[0] = add_template[0].reshape(B_1, C_1, -1).permute(0, 2, 1)
        # add_template[1] = add_template[1].reshape(B_1, C_1, -1).permute(0, 2, 1)
        # add_template[2] = add_template[2].reshape(B_1, C_1, -1).permute(0, 2, 1)
        #
        # add_search[0] = add_search[0].reshape(B_1, C_1, -1).permute(0, 2, 1)
        # add_search[1] = add_search[1].reshape(B_1, C_1, -1).permute(0, 2, 1)
        # add_search[2] = add_search[2].reshape(B_1, C_1, -1).permute(0, 2, 1)


        # for attention in self.layers1:
        #     x_1 = attention(x_cat_1, xpos)
        #     x_2 = attention(x_cat_2, y_pos)


        
        # x_tem = x_1 / 3               # 11/23;11:01
        # x_sear = x_2 / 3

        for attention in self.layers1:            # 11/23;10:56
            x = attention(x_cat_1, xpos)
        # for attention in self.layers1:            # 11/23;10:56
        #     y = attention(x_cat_2, y_pos)
            
        return x
            
def get_self_attention(config, **kwargs):
    fusion_spec = config.FUSION
    drop_path_allocator1 = DropPath(0.1)
    drop_path_allocator2 = nn.Identity()
    num_encoders = fusion_spec.num_encoders
    num_decoders = fusion_spec.num_decoders  
    dim = fusion_spec.dim
    num_heads = fusion_spec.num_heads
    mlp_ratio = fusion_spec.mlp_ratio
    qkv_bias = fusion_spec.qkv_bias
    drop_rate = fusion_spec.drop_rate
    attn_drop_rate = fusion_spec.attn_drop_rate
    
    encoder = []
    # decoder = []          # 11/23;10:56
    
    for index_of_encoder in range(num_encoders):
        encoder.append(SelfAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator1))
        # drop_path_allocator.increase_depth()
    
    # for index_of_encoder in range(num_decoders):              # 11/23;10:56
    #     decoder.append(CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_allocator2))
    #     # drop_path_allocator.increase_depth()
    
    # Fusion_netork = FUSION(encoder, decoder)              # 11/23;10:56
    Fusion_netork = FUSION(encoder)

    return Fusion_netork

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)