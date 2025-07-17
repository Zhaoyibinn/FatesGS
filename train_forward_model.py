import sys
sys.path.append("submodules/Viewcrafter")
sys.path.append("submodules/Viewcrafter/extern/dust3r")
import cv2
import os
import numpy as np
import torch
import json
from torch import nn
import open3d as o3d





from gaussian_renderer_nopose import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace


from Dust3r_class import Dust3r
import tinycudann as tcnn




parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)



with open("submodules/TCNN/hash_config.json") as f:
	config = json.load(f)


encoding = tcnn.Encoding(3, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, 19, config["network"])
model = torch.nn.Sequential(encoding, network)

# optimizer = config["Adam"]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

Dust3r_model = Dust3r()

img1,img2 = cv2.imread("DTU/set_23_24_33/scan24/images/0000.png"), cv2.imread("DTU/set_23_24_33/scan24/images/0001.png")

imgs_list = ["DTU/set_23_24_33/scan24/images/0000.png", "DTU/set_23_24_33/scan24/images/0001.png"]


full_out,pcd_np,align_model = Dust3r_model.run_only_model(imgs_list)

# pcd = full_out['pred1']['pts3d'].cuda().detach()
pcd = align_model.get_pts3d()[0].detach()
color = (full_out['view1']['img'].cuda().detach()+1)/2

reshaped_pcd = pcd.reshape(pcd.shape[0] * pcd.shape[1],pcd.shape[2])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(reshaped_pcd.cpu().detach().numpy())
# o3d.io.write_point_cloud("test.ply", pcd)

reshaped_color = color.permute(2,3,1,0).reshape(color.shape[2] * color.shape[3],color.shape[1])
bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

gt_rgb = (full_out['view1']['img'].cuda())/2 + 0.5
for i in range(10000):
	tcnn_pred = model(reshaped_pcd)

	render_pkg = render(full_out,reshaped_pcd,reshaped_color, tcnn_pred,align_model,pp, background)

	render_rgb = render_pkg['render'].unsqueeze(0)
	
	img_vis = cv2.hconcat([cv2.cvtColor((render_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB), cv2.cvtColor((gt_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB)])
	cv2.imwrite("test.png",img_vis)

	loss = torch.abs(render_rgb-gt_rgb).mean()

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(f"loss = {loss.item()}")

