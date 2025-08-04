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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Dust3r_class import Dust3r





from gaussian_renderer_nopose import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace
from fused_ssim import fused_ssim


# from Dust3r_class import Dust3r
import tinycudann as tcnn

from forawrd_model.train_dataloader import forward_model_dataset
# from forawrd_model.model_attention import Hash_gs_init
from forawrd_model.model_cnn import Hash_gs_init


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
attention_lr = 2e-5
base_lr = 1e-3

weight_decay = 1e-4

# optimizer = torch.optim.Adam([
#     # attention_block 使用特殊学习率
#     {'params': model.attention_block.parameters(), 'lr': attention_lr,"weight_decay": weight_decay},
    
#     # 其余所有模块使用基础学习率
#     {'params': model.encoding.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     # {'params': model.dinov2_feature.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.dpt_head.parameters(), 'lr': attention_lr ,"weight_decay": weight_decay},
#     {'params': model.relu.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.GS_head_fc1.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.GS_head_fc2.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.GS_head_out.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.fc_pcd.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.fc_color.parameters(), 'lr': base_lr ,"weight_decay": weight_decay},
#     {'params': model.rope.parameters(), 'lr': attention_lr ,"weight_decay": weight_decay}
# ])


optimizer = torch.optim.Adam(model.parameters(), lr=attention_lr,weight_decay=weight_decay)


root_path = "forawrd_model/datasets"
vis_out_dir = "forawrd_model/vis_out"
weight_save_dir = "forawrd_model/weights"
method = "vggt"
eval_epoch = 20


# dataset = forward_model_dataset(root_path,method="dust3r")
train_dataset = forward_model_dataset(root_path,method=method,val=False)
val_dataset = forward_model_dataset(root_path,method=method,val=True)
print(f"Whole Image Pair Num: Train {len(train_dataset)}; Val {len(val_dataset)}")
dataloader = DataLoader(
    train_dataset,
    batch_size=2,    # 批次大小
    shuffle=False,    # 打乱数据
    num_workers=0    
)

val_train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,    # 批次大小
    shuffle=False,    # 打乱数据
    num_workers=0    
)
# 有时候拿训练集测试
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,    # 批次大小
    shuffle=False,    # 打乱数据
    num_workers=0    
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

def render_batch(batch_data):
	pcd,color,mask = batch_data

	pcd = pcd.to(device)
	color = color.to(device)
	mask = mask.to(device)


	B,I,C,H,W = pcd.shape
	# pcd0,pcd1,color0,color1 = pcd0[0],pcd1[0],color0[0],color1[0]
	flattened_pcd = pcd.reshape(B,I,C,H*W)
	flattened_color = color.reshape(B,I,C,H*W)

	bg_color = [0, 0, 0]
	background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

	color0 = color[:,0,:,:,:].to(device)


	# if not render_0:
	# 	render_reshaped_pcd = reshaped_pcd1
	# 	render_reshaped_color = reshaped_color1
		
	# else:
	# 	render_reshaped_pcd = reshaped_pcd0
	# 	render_reshaped_color = reshaped_color0

	# gt_color = color0

	input = torch.cat([pcd, color], dim=2)
	tcnn_pred = model( input,color0.shape[1:])
	render_rgb_batch = []
	gt_rgb_batch = []
	for batch_idx in range(B):
		render_pkg = render(color[batch_idx][0],flattened_pcd[batch_idx][0],flattened_color[batch_idx][0], tcnn_pred[batch_idx],pp, background)

		render_rgb = render_pkg['render'].unsqueeze(0)
		
		if mask is not None :
			render_rgb = render_rgb * mask[batch_idx][0].unsqueeze(0).to(device)
			gt_rgb = color0[batch_idx].unsqueeze(0) * mask[batch_idx][0].unsqueeze(0).to(device)
		render_rgb_batch.append(render_rgb)
		gt_rgb_batch.append(gt_rgb)
	return torch.cat(render_rgb_batch,dim = 0), torch.cat(gt_rgb_batch,dim = 0),tcnn_pred

pbar = tqdm(range(2000))
for epoch in pbar:
	loss_record = []
	for batch_data in dataloader:

		render_rgb, gt_rgb,_ = render_batch(batch_data)
		loss_l1 = torch.abs(render_rgb-gt_rgb).mean()
		loss_ssim = 1.0 - fused_ssim(render_rgb, gt_rgb)

		loss = 0.8 * loss_l1 + 0.2 * loss_ssim

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_record.append(loss.item())

	writer.add_scalar('Loss/train', np.array(loss_record).mean(), epoch)

	if epoch%eval_epoch == 0:
		idx = 0
		torch.save(model.state_dict(), os.path.join(exp_weight_save_dir,f"model_weight_{epoch}.pth"))

		if epoch%100 == 0:
			render_dataloader = val_train_dataloader
			tensor_board_dir = "val_train_data"
		else:
			render_dataloader = val_dataloader
			tensor_board_dir = "val_data"
		
		
		for batch_data in render_dataloader:
			idx = idx + 1
			render_rgb, gt_rgb,tcnn_pred = render_batch(batch_data)

			pcd,color,mask = batch_data

			pcd = pcd.to(device)
			color = color.to(device)
			mask = mask.to(device)

			B,I,C,H,W = pcd.shape
			# pcd0,pcd1,color0,color1 = pcd0[0],pcd1[0],color0[0],color1[0]
			flattened_pcd = pcd.reshape(B,I,C,H*W)
			flattened_color = color.reshape(B,I,C,H*W)

			
			img_vis = cv2.hconcat([cv2.cvtColor((render_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB), cv2.cvtColor((gt_rgb[0].cpu().detach().numpy().transpose(1,2,0)) * 255,cv2.COLOR_BGR2RGB)])
			img_vis_torch = torch.tensor(cv2.cvtColor(img_vis,cv2.COLOR_RGB2BGR)).permute(2,0,1)/255
			writer.add_image(f'{tensor_board_dir}/val_img_{idx}', img_vis_torch, global_step=epoch)
			
			# cv2.imwrite(os.path.join(exp_vis_out_dir,f"render_{epoch}_{idx}.png"),img_vis)
			
			
			continue
			
			tcnn_pred = tcnn_pred.squeeze()
			world_xyz = flattened_pcd[0][0].permute(1,0)
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
			PlyData([el]).write(os.path.join(exp_vis_out_dir,f"GS_{epoch}_{idx}.ply"))
			
			# print("ply saved")
		# save_gs(gs_scene)
	pbar.set_description(f"Epoch {epoch} loss = {np.array(loss_record).mean():.4f}")
	# print(f"Epoch {epoch} loss = {np.array(loss_record).mean():.4f}")

writer.close()

