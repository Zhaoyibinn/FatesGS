import torch
import torch.nn as nn
from torchvision import transforms

class LinearReLU(nn.Module):
    """自定义层 包含全连接层和ReLU激活函数"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # 定义全连接层
        self.linear = nn.Linear(in_features, out_features)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前向传播：先经过全连接层，再经过ReLU
        x = self.linear(x)
        x = self.relu(x)
        return x
    
class Dinov2(nn.Module):
    """自定义层 包含全连接层和ReLU激活函数"""
    def __init__(self,upsample = 1.0):
        super().__init__()
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.dinov2_vits14.parameters():
            param.requires_grad = False
            # 冻结了Dinov2的参数 不要优化
        self.upsample = upsample
        if upsample!=1.0:
            self.Upsample = nn.Upsample(
                scale_factor=upsample,
                mode="bilinear",  # 插值方式
                align_corners=True  # 是否对齐角落像素（影响精度）
            )

        
        
    def forward(self, x):
        if self.upsample!=1.0:
            x = self.Upsample(x)
        patch_size = self.dinov2_vits14.patch_size
        patch_h = x.shape[-2] // patch_size
        patch_w = x.shape[-1] // patch_size

        trans_14 = transforms.Resize((patch_h * patch_size, patch_w * patch_size))
        reshape14_color = trans_14(x)
        result = self.dinov2_vits14.forward_features(reshape14_color)['x_norm_patchtokens']
        result_reshape = result.reshape(patch_h, patch_w,result.shape[-1]).unsqueeze(0).permute(0, 3, 1, 2)

        # result_reshape_vis = (result_reshape[:,:, 0] - result_reshape[:,:, 0].min()) / (result_reshape[:,:, 0].max() - result_reshape[:,:, 0].min())

        return result_reshape