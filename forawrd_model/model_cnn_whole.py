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
from forawrd_model.layers.UNet import Unet


from forawrd_model.head.dpt_head import DPTHead
import time



class Hash_gs_init(nn.Module):
    def __init__(self):
        super().__init__()
        with open("submodules/TCNN/hash_config.json") as f:
            config = json.load(f)


        self.encoding_color = tcnn.Encoding(3, config["encoding"])
        self.encoding_pcd = tcnn.Encoding(3, config["encoding"])

        self.feature_unet_color = Unet(in_channels = 32, base_channels  = 16)
        self.feature_unet_pcd = Unet(in_channels = 32, base_channels  = 16)

        self.unet_gs_out = LinearReLU(64, 19)



        
        # self.attention_block = Attention(self.dinov2_feature.dinov2_vits14.num_features)
        # self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')


        # network = tcnn.Network(encoding.n_output_dims, 19, config["network"])
    def forward(self,x,img_shape):
        pcd = x[:,:,:3]
        color = x[:,:,-3:]
        
        # reshape_color = color.reshape([img_shape[0],img_shape[1],3]).permute(2,0,1).unsqueeze(0)
        
        B, S, C_in ,H, W = x.shape

        flatten_pcd = pcd.reshape(B,S,C_in // 2,H*W)
        flatten_color = color.reshape(B,S,C_in // 2,H*W)

        flatten_pcd_for_tcnn = flatten_pcd.permute(3,0,1,2).reshape(-1,C_in // 2)
        flatten_color_for_tcnn = flatten_color.permute(3,0,1,2).reshape(-1,C_in // 2)



        flatten_pcd_tcnn_encode = self.encoding_pcd(flatten_pcd_for_tcnn).reshape(-1,B, S,32).permute(1,2,3,0)
        flatten_color_tcnn_encode = self.encoding_color(flatten_color_for_tcnn).reshape(-1,B, S,32).permute(1,2,3,0)
        
        pcd_tcnn_encode = flatten_pcd_tcnn_encode.reshape(B, S,32,H, W)
        color_tcnn_encode = flatten_color_tcnn_encode.reshape(B, S,32,H, W)
        
        Unet_align_transforms = transforms.Resize(((H//4 + 1) * 4, (W//4 + 1) * 4))
        Unet_align_Untransforms = transforms.Resize((H,W))

        pcd_unet = Unet_align_Untransforms(self.feature_unet_pcd(Unet_align_transforms(pcd_tcnn_encode.reshape(B*S,32,H, W)))).reshape(B,S,32,H, W)
        color_unet = Unet_align_Untransforms(self.feature_unet_pcd(Unet_align_transforms(color_tcnn_encode.reshape(B*S,32,H, W)))).reshape(B,S,32,H, W)

        flatten_pcd_unet = pcd_unet.reshape(B, S, 32 ,H * W).permute(3,0,1,2).reshape(-1,32)
        flatten_color_unet = color_unet.reshape(B, S, 32 ,H * W).permute(3,0,1,2).reshape(-1,32)

        mix_gs_out = self.unet_gs_out(torch.cat([flatten_pcd_unet,flatten_color_unet],dim = 1)).reshape(-1,B,S,19).permute(1,2,3,0)

        GS_out = mix_gs_out[:,0,:,:].permute(0,2,1)
        # 暂时只用了第一张图片的特征






        return GS_out


        
    