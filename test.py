import sys
sys.path.append("submodules/Viewcrafter")
sys.path.append("submodules/Viewcrafter/extern/dust3r")


import os
import cv2
import open3d as o3d
import numpy as np
import torch

from Dust3r_class import Dust3r

model = Dust3r()

path_root = "DTU/set_23_24_33/scan40"
img_path_root = os.path.join(path_root, "images")

img_path_list = sorted(os.listdir(img_path_root))
img_path_list = [os.path.join(img_path_root, img_path) for img_path in img_path_list]
img_list = [cv2.imread(img_path) for img_path in img_path_list]

model.load_data(img_path_list,sparse_colmap_path_root=os.path.join(path_root, "sparse","0"))
model.run_dust3r()
# model.save_pointcloud_with_normals(filter=True, save_path=os.path.join(path_root, "sparse","0","points3D_dust3r.ply"))

model.save_pointcloud_with_gt(save_path = "test.ply")
print("end")