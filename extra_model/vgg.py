import torch
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from VGG19 import VGG19
from collections import OrderedDict


# model = torchvision.models.vgg19()
model = VGG19()
state_dict = torch.load("extra_model/vgg19-dcbb9e9d.pth")

keys_to_delete = [k for k, v in state_dict.items() if not k.startswith("feature")]
for key in keys_to_delete:
    del state_dict[key]

new_state_dict = OrderedDict()
for key, value in state_dict.items():
    new_key = "conv" + key.lstrip('featrues.')
    # new_key = "layers." +  key.lstrip('featrues.')
    new_state_dict[new_key] = value





model.load_state_dict(new_state_dict)
# model = model.features
model.eval()

# model_child = nn.Sequential(*list(model.children())[:10 + 1])

image = cv2.cvtColor(cv2.imread("DTU/diff/scan24/images/0001.png"),cv2.COLOR_BGR2RGB)

image = image.astype(np.float32) / 255.0
# 将图像从 (H, W, C) 格式转换为 (C, H, W) 格式
image = np.transpose(image, (2, 0, 1))
# 将 numpy 数组转换为 torch.Tensor
image_tensor = torch.from_numpy(image)
# 添加一个批次维度，VGG - 19 模型期望输入为 (batch_size, C, H, W) 格式
image_tensor = image_tensor.unsqueeze(0)

preprocess = transforms.Compose([
    # 调整图像大小为 224x224
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)),
    # 归一化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = preprocess(image_tensor)

if torch.cuda.is_available():
    model = model.to('cuda')
    image_tensor = image_tensor.to('cuda')

with torch.no_grad():
    output = model(image_tensor)

_, predicted_idx = torch.max(output, 1)

