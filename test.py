import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
 
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
 
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
 
import torchvision
from torchvision import datasets, models, transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
 
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14 = dinov2_vits14.to(device)

# 可视化
patch_size = dinov2_vits14.patch_size
 
patch_h = 520 // patch_size
patch_w = 520 // patch_size
feat_dim = 384
 
transform1 = transforms.Compose([
    transforms.Resize(520),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])
 
total_features = []
# EXAMPLE_PATH = '../dinov2_format_data/classification'
 
# 需要准备一个examples文件夹，里面放一些图像，例如dog和cat
folder_path = f'DTU/set_23_24_33/scan24/images'

with torch.no_grad():
  list_img_path = os.listdir(folder_path)[0:4]
  for img_path in list_img_path:
    img_path = os.path.join(folder_path, img_path)
    img = Image.open(img_path).convert('RGB')
    img_t = transform1(img).to(device)
    
    features_dict = dinov2_vits14.forward_features(img_t.unsqueeze(0))
    features = features_dict['x_norm_patchtokens']
    total_features.append(features)
 
total_features = torch.cat(total_features, dim=0)
 
# PCA分离背景
total_features = total_features.reshape(len(list_img_path) * patch_h * patch_w, feat_dim)
total_features = total_features.cpu()
 
pca = PCA(n_components=3)
# scaler = MinMaxScaler(clip=True)
pca.fit(total_features)
pca_features = pca.transform(total_features)
 
plt.subplot(2, 2, 1)
plt.hist(pca_features[:, 0])
plt.subplot(2, 2, 2)
plt.hist(pca_features[:, 1])
plt.subplot(2, 2, 3)
plt.hist(pca_features[:, 2])
# plt.show()
# plt.savefig("test_1.png")
plt.close()
 
# min_max缩放
pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                     (pca_features[:, 0].max() - pca_features[:, 0].min())
# pca_features = sklearn.processing.minmax_scale(pca_features)
 
for i in range(3):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))
 
# plt.show()
plt.savefig("test_2.png")
 
# 分离前景和背景
pca_features_bg = pca_features[:, 0] < 0.5
pca_features_fg = ~pca_features_bg
 
for i in range(3):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
# plt.show()
# plt.savefig("test_3.png")
 
pca.fit(total_features[pca_features_fg]) 
pca_features_left = pca.transform(total_features[pca_features_fg])
 
for i in range(3):
    # min_max scaling
    pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())
 
pca_features_rgb = pca_features.copy()
 
pca_features_rgb[pca_features_bg] = 0
 
pca_features_rgb[pca_features_fg] = pca_features_left
 
 
pca_features_rgb = pca_features_rgb.reshape(3, patch_h, patch_w, 3)
for i in range(3):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_rgb[i])
 
# plt.show()
# plt.savefig("test_4.png")

