from colmap_align import *
import os
import torch 
import numpy as np
import open3d as o3d

def trans_with_rst(source_points,s,R,t):
    target_points = np.dot(s * R,source_points.T).T + np.array(t)
    return target_points

def trans_with_rt(source_points,R,t):
    target_points = np.dot(R,source_points.T).T + np.array(t)
    return target_points

def trans_with_st(source_points,s,t):
    target_points = s * source_points + t
    return target_points


def align1_camera_pose(scan,pose):
    # vggt_pose_root = pose
    gt_pose_root =f"DTU/set_23_24_33/scan{scan}/sparse/0"


    # assert os.path.exists(os.path.join(vggt_pose_root,"images.txt")) or os.path.exists(os.path.join(vggt_pose_root,"images.bin")), "vggt txt和bin文件都不存在 请检查"
    assert os.path.exists(os.path.join(gt_pose_root,"images.txt")) or os.path.exists(os.path.join(gt_pose_root,"images.bin")), "gt txt和bin文件都不存在 请检查"

    # assert os.path.exists(os.path.join(vggt_pose_root,"points3D.ply")), "vggt points3D.ply文件不存在 请检查"
    assert os.path.exists(os.path.join(gt_pose_root,"points3D.ply")) or os.path.exists(os.path.join(gt_pose_root,"points3D_colmap.ply")), "colmap points3D.ply文件不存在 请检查"

    # vggt_image_txt_path = os.path.join(vggt_pose_root,"images.txt")
    gt_image_txt_path = os.path.join(gt_pose_root,"images.txt")
    # vggt_ply_path = os.path.join(vggt_pose_root,"points3D.ply")
    # vggt_ply_path = input_ply
    if not os.path.exists(os.path.join(gt_pose_root,"points3D.ply")):
        colmap_ply_path = os.path.join(gt_pose_root,"points3D_colmap.ply")
    else:
        colmap_ply_path = os.path.join(gt_pose_root,"points3D.ply")

    gt_intrinsic = os.path.join(gt_pose_root,"cameras.txt")

    # vggt_ply = o3d.io.read_point_cloud(vggt_ply_path)
    colmap_ply = o3d.io.read_point_cloud(colmap_ply_path)


    # if not os.path.exists(vggt_image_txt_path):
        # os.system(f"colmap model_converter --input_path {vggt_pose_root} --output_path {vggt_pose_root} --output_type TXT")
        # print("vggt colmap bin转txt完成")
    if not os.path.exists(gt_image_txt_path):
        os.system(f"colmap model_converter --input_path {gt_pose_root} --output_path {gt_pose_root} --output_type TXT")
        print("gt colmap bin转txt完成")




    image_W,image_H,focal_x,focal_y,cx,cy = read_colmap_camera(gt_intrinsic)

    # vggt_pose = read_colmap_gt(vggt_image_txt_path)
    # if focal_scale != None:
        # vggt_pose[:,:,3] = vggt_pose[:,:,3] * focal_scale
    gt_pose = read_colmap_gt(gt_image_txt_path)  
    last_row = torch.tensor([0, 0, 0, 1]).expand(gt_pose.shape[0], 1, 4)
    gt_pose44 = torch.cat([gt_pose, last_row], dim=1)

    pose_c2ws = []
    if pose.shape[1] !=4:
        q_normal = torch.nn.functional.normalize(pose[:,0:4], dim=1)
        rot = kornia.geometry.quaternion_to_rotation_matrix(q_normal)
        # vggt_pose = torch.cat([vggt_pose, last_row], dim=1)
        t = pose[:3, 4:]
        for i in range(pose.shape[0]):
            pose_c2w = torch.eye(4)
            pose_c2w[:3, :3] = rot[i]
            pose_c2w[:3, 3] = t[i].squeeze()
            pose_c2ws.append(pose_c2w)
        pose_c2ws = torch.stack(pose_c2ws, dim=0).cpu().detach()
    else:
        pose_c2ws = pose
        # vggt_pose[i] = torch.matmul(extra_pose_4x4.T, vggt_pose[i])


    s_2colmap, R_2colmap, T_2colmap = align_multiple_poses(pose_c2ws,gt_pose44)
    SRT = torch.eye(4)
    SRT[:3, :3] = s_2colmap * R_2colmap
    SRT[:3, 3] = T_2colmap
    poses_new = torch.matmul(SRT[None, ...], pose_c2ws)  # [N, 4, 4]

    return s_2colmap,R_2colmap,T_2colmap,poses_new,gt_pose44

import argparse
import pandas as pd
results = []
parser = argparse.ArgumentParser(description="你的脚本说明")
# parser.add_argument('--scans', type=int, nargs='+', help='要处理的scan编号列表')
parser.add_argument('--input_root', type=str, required=True, help='输入根目录')
# parser.add_argument('--iteration', type=int, required=True, help='迭代次数')
# parser.add_argument('--vggt_root', type=str, default="pilianghua_out/gs_init/pilianghua_output_gsinit/vggt_pcd")
args = parser.parse_args()

# iteration = args.iteration

for scan in [24 ,37 ,40 ,55 ,63 ,65 ,69 ,83, 97,105, 106, 110, 114, 118, 122]:
# for scan in [24]:
    # scan = 24
    # vggt_origin_root = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt"
    # vggt_origin_root = args.vggt_root
    input_path_root = args.input_root
    output_path_root = input_path_root

    pose_path = os.path.join(input_path_root,f"scan{scan}","poses.pth")
    pose = torch.load(pose_path)



    s,R,t,pose_align,pose_gt = align1_camera_pose(scan,pose)
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
        float(euler_error[0]),
        float(euler_error[1]),
        float(euler_error[2]),
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
