import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def create_lowpass_filter(size, cutoff, device='cpu'):
    """
    创建低通滤波器
    
    参数:
    size: 滤波器大小，通常与输入图像大小相同
    cutoff: 截止频率，0-1之间的值，表示保留的低频比例
    device: 运行设备，如'cpu'或'cuda'
    
    返回:
    低通滤波器，形状为(1, 1, size[0], size[1])
    """
    # 创建频率网格
    h, w = size
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
    center_y, center_x = h // 2, w // 2
    
    # 计算每个点到中心的距离（频率）
    distances = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    
    # 归一化频率
    max_distance = torch.max(distances)
    normalized_distances = distances / max_distance
    
    # 创建低通滤波器（圆形）
    lowpass_filter = torch.zeros_like(normalized_distances)
    lowpass_filter[normalized_distances <= cutoff] = 1.0
    
    # 应用汉宁窗以减少振铃效应
    window = torch.outer(torch.hann_window(h), torch.hann_window(w))
    lowpass_filter *= window
    
    # 调整形状以适应卷积操作
    return lowpass_filter.to(device).unsqueeze(0).unsqueeze(0)

def apply_lowpass_filter(image_tensor, filter_tensor):
    """
    应用低通滤波器到图像
    
    参数:
    image_tensor: 输入图像张量，形状为(1, C, H, W)
    filter_tensor: 低通滤波器张量，形状为(1, 1, H, W)
    
    返回:
    滤波后的图像张量
    """
    # 将图像转换到频域
    image_fft = torch.fft.fft2(image_tensor, dim=(2, 3))
    image_fft_shifted = torch.fft.fftshift(image_fft, dim=(2, 3))
    
    # 应用滤波器
    filtered_fft_shifted = image_fft_shifted * filter_tensor
    
    # 将滤波后的图像转换回空间域
    filtered_fft = torch.fft.ifftshift(filtered_fft_shifted, dim=(2, 3))
    filtered_image = torch.fft.ifft2(filtered_fft, dim=(2, 3))
    filtered_image = torch.abs(filtered_image)
    
    return filtered_image