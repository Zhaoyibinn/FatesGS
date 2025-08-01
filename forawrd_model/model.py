import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

import tinycudann as tcnn
import json
from forawrd_model.layers.network_utils import *
from forawrd_model.layers.attention import Attention
from forawrd_model.layers.block import Block
from forawrd_model.layers.rope import RotaryPositionEmbedding2D, PositionGetter


from forawrd_model.head.dpt_head import DPTHead



class Hash_gs_init(nn.Module):
    def __init__(self):
        super().__init__()
        with open("submodules/TCNN/hash_config.json") as f:
            config = json.load(f)


        self.encoding = tcnn.Encoding(3, config["encoding"])

        self.dinov2_feature = Dinov2(upsample=1.0)

        self.attention_depth = 6
        # self.attention_block = nn.ModuleList(
        #     [Attention(self.dinov2_feature.dinov2_vits14.num_features)
        #         for _ in range(self.attention_depth)
        #     ]
        # )
        
        self.attention_block = nn.ModuleList(
            [Block(dim = self.dinov2_feature.dinov2_vits14.num_features,num_heads=8)
                for _ in range(self.attention_depth)
            ]
        )
        
        self.dpt_head = DPTHead(dim_in = self.dinov2_feature.dinov2_vits14.num_features)


        self.relu = nn.ReLU()

        self.GS_head_fc1 = LinearReLU(32 *2, 64)
        self.GS_head_fc2 = LinearReLU(64, 64)
        self.GS_head_out = nn.Linear(64, 19)

        self.fc_pcd = LinearReLU(config["encoding"]['n_levels'] * config["encoding"]['n_features_per_level'], 32)
        self.fc_color = LinearReLU(self.dpt_head.out_channels[0]//2, 32)

        self.rope = RotaryPositionEmbedding2D(frequency=1000)
        self.position_getter = PositionGetter() if self.rope is not None else None




        
        # self.attention_block = Attention(self.dinov2_feature.dinov2_vits14.num_features)
        # self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        
        # network = tcnn.Network(encoding.n_output_dims, 19, config["network"])
    def forward(self,x,img_shape):
        pcd = x[:,:3]
        color = x[:,-3:]
        reshape_color = color.reshape([img_shape[0],img_shape[1],3]).permute(2,0,1).unsqueeze(0)

        H, W = img_shape
        B, S, C_in = 1,1,3
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.dinov2_feature.dinov2_vits14.patch_size, W // self.dinov2_feature.dinov2_vits14.patch_size, device="cuda")

        dinov2_feature,reshape14_color = self.dinov2_feature(reshape_color)
        attention_tokens = dinov2_feature
        attention_tokens_list = []
        for attention_idx in range(self.attention_depth):
            attention_tokens = self.attention_block[attention_idx](attention_tokens,pos)
            attention_tokens_list.append(attention_tokens)

            
        dpt_out = self.dpt_head(attention_tokens_list,reshape14_color.unsqueeze(0),0)
        # dpt_out_feature,dpt_out_conf = dpt_out[0],dpt_out[1]
        # reshape_dinov2_feature = F.interpolate(
        #                             dinov2_feature,
        #                             size=(reshape_color.shape[2], reshape_color.shape[3]),  # 调整为48x48
        #                             mode='bilinear',  # 插值方式：双线性插值（常用）
        #                             align_corners=False  # 边缘对齐参数
        #                         )
        
        flatten_reshape_dpt_feature = dpt_out.squeeze().flatten(-2,-1).permute(1,0)
        # result_reshape_vis = (dinov2_feature[:,:, 0] - dinov2_feature[:,:, 0].min()) / (dinov2_feature[:,:, 0].max() - dinov2_feature[:,:, 0].min())

        ecoded_pcd = self.encoding(pcd)
        fc_pcd = self.fc_pcd(ecoded_pcd)

        fc_color = self.fc_color(flatten_reshape_dpt_feature)

        fc_cated = torch.cat([fc_pcd,fc_color],dim=1)

        GS_fc1 = self.GS_head_fc1(fc_cated)
        GS_fc2 = self.GS_head_fc2(GS_fc1)
        GS_out = self.GS_head_out(GS_fc2)

        return GS_out


        
    