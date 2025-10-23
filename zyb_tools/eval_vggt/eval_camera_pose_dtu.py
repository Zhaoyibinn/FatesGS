from colmap_align import *
import os
import torch 
import numpy as np
import open3d as o3d
import pandas as pd

def trans_with_rst(source_points,s,R,t):
    target_points = np.dot(s * R,source_points.T).T + np.array(t)
    return target_points

def trans_with_rt(source_points,R,t):
    target_points = np.dot(R,source_points.T).T + np.array(t)
    return target_points

def trans_with_st(source_points,s,t):
    target_points = s * source_points + t
    return target_points


import argparse
results = []
parser = argparse.ArgumentParser(description="你的脚本说明")
# parser.add_argument('--scans', type=int, nargs='+', help='要处理的scan编号列表')
parser.add_argument('--input_root', type=str, required=True, help='输入根目录')
parser.add_argument('--iteration', type=int, required=True, help='迭代次数')
# parser.add_argument('--vggt_root', type=str, default="pilianghua_out/gs_init/pilianghua_output_gsinit/vggt_pcd")
args = parser.parse_args()

iteration = args.iteration

for scan in [24 ,37 ,40 ,55 ,63 ,65 ,69 ,83, 97,105, 106, 110, 114, 118, 122]:
# for scan in [24]:
    # scan = 24
    # vggt_origin_root = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt"
    # vggt_origin_root = args.vggt_root
    input_path_root = args.input_root
    output_path_root = input_path_root

    extra_pose_path = os.path.join(output_path_root,f"scan{scan}/point_cloud/iteration_{iteration}/extra_trans.pth")
    extra_pose = torch.load(extra_pose_path)

    # input_path_vggt = os.path.join(vggt_origin_root,f"scan{scan}/train/ours_1/fuse_post.ply")
    input_path = os.path.join(input_path_root,f"scan{scan}/train/ours_{iteration}/fuse.ply")
    # input_path_2 = os.path.join(input_path_root,f"scan{scan}/train/ours_1/fuse_post.ply")

    # output_path_vggt = os.path.join(output_path_root,f"scan{scan}/train/vggt_align_culled.ply")
    output_path = os.path.join(output_path_root,f"scan{scan}/train/ours_{iteration}_align_culled.ply")
    # output_path_2 = os.path.join(output_path_root,f"scan{scan}/train/ours_1_align_culled.ply")

    # if not os.path.exists(input_path_vggt):
    #     print(f"vggt mesh 不存在: {input_path}")
    #     assert os.path.exists( os.path.join(vggt_origin_root,f"scan{scan}/sparse/vggt/points3D.ply")), print(f"vggt pointcloud也不存在 有问题 请检查")
    #     input_ply_vggt_pcd = o3d.io.read_point_cloud(os.path.join(vggt_origin_root,f"scan{scan}/sparse/vggt/points3D.ply"))
    #     alpha = 0.005  # 调整 alpha 值以控制网格的细节程度
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(input_ply_vggt_pcd, alpha)
    #     o3d.io.write_triangle_mesh(input_path_vggt, mesh)
    #     print("生成vggt mesh完成")
        

    input_ply = o3d.io.read_point_cloud(input_path)
    # input_ply_vggt = o3d.io.read_point_cloud(input_path_vggt)
    # input_ply_vggt = o3d.io.read_point_cloud(input_path_vggt)
    # input_ply_2 = o3d.io.read_point_cloud(input_path_2)

    # free_gs_dataroot = "DTU/set_23_23_33_freegs"
    # cameras_txt = os.path.join(free_gs_dataroot,f"scan{scan}","sparse/0/cameras.txt")
    # with open(cameras_txt, 'r') as f:
    #     cameras_content = f.read()
    # est_focal = float(cameras_content.split("\n")[3].split(" ")[-3])
    # focal_scale = 2892.3 / est_focal

    s,R,t,source_ply,colmap_ply,pose_align,pose_gt = align1_camera_pose(scan,input_path,extra_pose = None)
    # 和colmap位姿对齐 但是由于VGGT的位姿和内参没有那么准 所以还需要配准
    # 计算旋转角度误差
    R_align = pose_align[:,:3,:3]
    R_gt = pose_gt[:,:3,:3]
    
    R_diff = torch.matmul(R_align, R_gt.transpose(1, 2))
    # 将旋转矩阵转为欧拉角（以度为单位）
    def rotation_matrix_to_euler_angles(R):
        sy = torch.sqrt(R[:,0,0] ** 2 + R[:,1,0] ** 2)
        singular = sy < 1e-6
        x = torch.atan2(R[:,2,1], R[:,2,2])
        y = torch.atan2(-R[:,2,0], sy)
        z = torch.atan2(R[:,1,0], R[:,0,0])
        x[singular] = torch.atan2(-R[singular,1,2], R[singular,1,1])
        y[singular] = torch.atan2(-R[singular,2,0], sy[singular])
        z[singular] = 0
        return torch.stack([x, y, z], dim=1) * 180.0 / np.pi

    euler_error = rotation_matrix_to_euler_angles(R_diff).abs().mean(dim=0)
    
    # print(f"{euler_error[0].item()} {euler_error[1].item()} {euler_error[2].item()}")
    error = torch.abs((pose_align[:,:3,3] - pose_gt[:,:3,3])).mean().item() * 1000

    
    

    # Append new result for each scan
    results.append([
        scan,
        min(float(euler_error[0]),180-float(euler_error[0])),
        min(float(euler_error[1]),180-float(euler_error[1])),
        min(float(euler_error[2]),180-float(euler_error[2])),
        float(error)
    ])

xls_path = "test.xlsx"
data = [['Scan', 'Euler_X', 'Euler_Y', 'Euler_Z', 'Trans_Error(mm)']]
data.extend(results)
df = pd.DataFrame(data[1:], columns=data[0])
df.to_excel(xls_path, index=False)
all_results_saved = True


    # input_ply_posealign_numpy = trans_with_rst(np.array(input_ply.points),np.array(s),np.array(R),np.array(t))
    # input_ply_vggt_posealign_numpy = trans_with_rst(np.array(input_ply_vggt.points),np.array(s),np.array(R),np.array(t))
    # input_ply_2_posealign_numpy = trans_with_rst(np.array(input_ply_2.points),np.array(s),np.array(R),np.array(t))
