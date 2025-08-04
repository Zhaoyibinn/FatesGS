import torch.nn as nn
from forawrd_model.layers.network_utils import *
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels, base_channels, norm_act=nn.BatchNorm2d):
        super(Unet, self).__init__()
        self.conv0 = nn.Sequential(
                        ConvBnReLU(in_channels, base_channels, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(base_channels, base_channels, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(base_channels, base_channels*2, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels*2, base_channels*2, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(base_channels*2, base_channels*4, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels*4, base_channels*4, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(base_channels*4, base_channels*4, 1)
        self.lat1 = nn.Conv2d(base_channels*2, base_channels*4, 1)
        self.lat0 = nn.Conv2d(base_channels, base_channels*4, 1)

        self.smooth0 = nn.Conv2d(base_channels*4, in_channels, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat0 = self.smooth0(feat0)
        return feat0