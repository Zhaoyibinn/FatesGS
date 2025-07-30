import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

import tinycudann as tcnn
import json
from forawrd_model.network_utils import *


class Hash_gs_init(nn.Module):
    def __init__(self):
        super().__init__()
        with open("submodules/TCNN/hash_config.json") as f:
            config = json.load(f)


        self.encoding = tcnn.Encoding(3, config["encoding"])

        self.dinov2_feature = Dinov2(upsample=2.0)

        self.fc_pcd = LinearReLU(config["encoding"]['n_levels'] * config["encoding"]['n_features_per_level'], 32)
        self.fc_color = LinearReLU(self.dinov2_feature.dinov2_vits14.num_features, 32)
        self.relu = nn.ReLU()

        self.GS_head_fc1 = LinearReLU(32 *2, 64)
        self.GS_head_fc2 = LinearReLU(64, 64)
        self.GS_head_out = nn.Linear(64, 19)

        # self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        
        # network = tcnn.Network(encoding.n_output_dims, 19, config["network"])
    def forward(self,x,img_shape):
        pcd = x[:,:3]
        color = x[:,-3:]
        reshape_color = color.reshape([img_shape[0],img_shape[1],3]).permute(2,0,1).unsqueeze(0)

        dinov2_feature = self.dinov2_feature(reshape_color)

        reshape_dinov2_feature = F.interpolate(
                                    dinov2_feature,
                                    size=(reshape_color.shape[2], reshape_color.shape[3]),  # 调整为48x48
                                    mode='bilinear',  # 插值方式：双线性插值（常用）
                                    align_corners=False  # 边缘对齐参数
                                )
        
        flatten_reshape_dinov2_feature = reshape_dinov2_feature.squeeze().flatten(-2,-1).permute(1,0)
        # result_reshape_vis = (dinov2_feature[:,:, 0] - dinov2_feature[:,:, 0].min()) / (dinov2_feature[:,:, 0].max() - dinov2_feature[:,:, 0].min())

        ecoded_pcd = self.encoding(pcd)
        fc_pcd = self.fc_pcd(ecoded_pcd)

        fc_color = self.fc_color(flatten_reshape_dinov2_feature)

        fc_cated = torch.cat([fc_pcd,fc_color],dim=1)

        GS_fc1 = self.GS_head_fc1(fc_cated)
        GS_fc2 = self.GS_head_fc2(GS_fc1)
        GS_out = self.GS_head_out(GS_fc2)

        return GS_out


        
    