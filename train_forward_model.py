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
from plyfile import PlyData, PlyElement
from torch.utils.data import Dataset, DataLoader
import datetime
from torch.utils.tensorboard import SummaryWriter

from Dust3r_class import Dust3r





from gaussian_renderer_nopose import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace
from fused_ssim import fused_ssim


# from Dust3r_class import Dust3r
import tinycudann as tcnn

from forawrd_model.train_dataloader import forward_model_dataset
from forawrd_model.model import Hash_gs_init


def save_gs(gs_scene):
	pass

def inverse_sigmoid(x):
    return torch.log(x/(1-x))



parser = ArgumentParser(description="Training script parameters")
# lp = ModelParams(parser)
# op = OptimizationParams(parser)
pp = PipelineParams(parser)




with open("submodules/TCNN/hash_config.json") as f:
	config = json.load(f)


encoding = tcnn.Encoding(3, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, 19, config["network"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Hash_gs_init().to(device)

# optimizer = config["Adam"]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)


root_path = "forawrd_model/datasets"
vis_out_dir = "forawrd_model/vis_out"
weight_save_dir = "forawrd_model/weights"
dataset = forward_model_dataset(root_path)
print(f"Whole Image Pair Num: {len(dataset)}")
dataloader = DataLoader(
    dataset,
    batch_size=1,    # 批次大小
    shuffle=True,    # 打乱数据
    num_workers=0    # 加载数据的进程数（0表示主线程）
)


i = 0


if not os.path.exists(vis_out_dir):
	os.mkdir(vis_out_dir)

if not os.path.exists(weight_save_dir):
	os.mkdir(weight_save_dir)



now = datetime.datetime.now()
datetime_num = now.strftime("%Y%m%d%H%M%S")
exp_vis_out_dir = os.path.join(vis_out_dir,str(datetime_num))
exp_weight_save_dir = os.path.join(weight_save_dir,str(datetime_num))
os.mkdir(exp_vis_out_dir)
os.mkdir(exp_weight_save_dir)

writer = SummaryWriter(f"{exp_weight_save_dir}")



for epoch in range(1000):
	loss_record = []
	for batch_data in dataloader:

	# img1,img2 = cv2.imread("DTU/set_23_24_33/scan24/images/0000.png"), cv2.imread("DTU/set_23_24_33/scan24/images/0001.png")
		# img1,img2 = batch_data
		# img1,img2 = img1[0],img2[0]
		# imgs_list = ["DTU/set_23_24_33/scan24/images/0000.png", "DTU/set_23_24_33/scan24/images/0001.png"]
		pcd0,pcd1,color0,color1 = batch_data
		pcd0,pcd1,color0,color1 = pcd0[0],pcd1[0],color0[0],color1[0]

		# full_out,pcd_np,align_model = Dust3r_model.run_only_model(imgs_list)

		# # pcd = full_out['pred1']['pts3d'].cuda().detach()
		# pcd0,pcd1 = align_model.get_pts3d()[0].detach(),align_model.get_pts3d()[1].detach()
		# color0,color1 = (full_out['view1']['img'].cuda().detach()+1)/2,(full_out['view2']['img'].cuda().detach()+1)/2
		
		reshaped_pcd0,reshaped_pcd1 = pcd0.reshape(pcd0.shape[0] * pcd0.shape[1],pcd0.shape[2]),pcd1.reshape(pcd1.shape[0] * pcd1.shape[1],pcd1.shape[2])

		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(reshaped_pcd.cpu().detach().numpy())
		# o3d.io.write_point_cloud("test.ply", pcd)

		reshaped_color0,reshaped_color1 = color0.permute(2,3,1,0).reshape(color0.shape[2] * color0.shape[3],color0.shape[1]),color1.permute(2,3,1,0).reshape(color1.shape[2] * color1.shape[3],color1.shape[1])
		
		bg_color = [0, 0, 0]
		background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

		gt_rgb = color0
		# for i in range(10000):
		encoded_pcd = encoding(reshaped_pcd1)
		
		tcnn_pred = model( torch.cat([reshaped_pcd1, reshaped_color1], dim=1),color0.shape[2:])

		render_pkg = render(color0,reshaped_pcd1,reshaped_color0, tcnn_pred,pp, background)

		render_rgb = render_pkg['render'].unsqueeze(0)
		
		

		loss_l1 = torch.abs(render_rgb-gt_rgb).mean()
		loss_ssim = 1.0 - fused_ssim(render_rgb, gt_rgb)

		loss = 0.8 * loss_l1 + 0.2 * loss_ssim

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_record.append(loss.item())

	writer.add_scalar('Loss/train', np.array(loss_record).mean(), epoch)

	if epoch%10 == 0:
		
		img_vis = cv2.hconcat([cv2.cvtColor((render_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB), cv2.cvtColor((gt_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB)])

		cv2.imwrite(os.path.join(exp_vis_out_dir,f"render_{epoch}.png"),img_vis)
		
		torch.save(model.state_dict(), os.path.join(exp_weight_save_dir,f"model_weight_{epoch}.pth"))

		world_xyz = reshaped_pcd1
		opacity_out = tcnn_pred[:,0].unsqueeze(-1) - 2
		scale_out = tcnn_pred[:,1:3]-10
		rot_out = tcnn_pred[:,3:7]

		features_dc = torch.reshape(tcnn_pred[:,7:10],(tcnn_pred[:,7:10].shape[0],-1,3))
		features_rest = torch.reshape(tcnn_pred[:,10:],(tcnn_pred[:,10:].shape[0],-1,3))
		shs = torch.cat((features_dc.squeeze(), features_rest.flatten(1)), dim=1)
		color_out = shs

		xyz = world_xyz.squeeze().detach().cpu().numpy()
		normals = np.zeros_like(xyz)
		color = color_out.squeeze().detach().cpu().numpy()
		# C0 = 0.28209479177387814
		# color = (color - 0.5) / C0
		opacities = opacity_out.squeeze(0).detach().cpu().numpy()
		scale = scale_out.squeeze().detach().cpu().numpy()
		rotation = rot_out.squeeze().detach().cpu().numpy()

		l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
		# All channels except the 3 DC
		for ii in range(3):
			l.append('f_dc_{}'.format(ii))
		for ii in range(9):
			l.append('f_rest_{}'.format(ii))
		l.append('opacity')
		for ii in range(2):
			l.append('scale_{}'.format(ii))
		for ii in range(4):
			l.append('rot_{}'.format(ii))
		# l.append('red')
		# l.append('green')
		# l.append('blue')


		dtype_full = [(attribute, 'f4') for attribute in l]


		elements = np.empty(xyz.shape[0], dtype=dtype_full)


		attributes = np.concatenate((xyz, normals, color, opacities, scale, rotation), axis=1)


		elements[:] = list(map(tuple, attributes))
		el = PlyElement.describe(elements, 'vertex')
		PlyData([el]).write(os.path.join(exp_vis_out_dir,f"GS_{epoch}.ply"))
		# print("ply saved")
	# save_gs(gs_scene)
	print(f"Epoch {epoch} loss = {np.array(loss_record).mean()}")

writer.close()

