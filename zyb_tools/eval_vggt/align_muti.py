from colmap_align import *
import os
import torch 
import numpy as np

def trans_with_rst(source_points,s,R,t):
    target_points = np.dot(s * R,source_points.T).T + np.array(t)
    return target_points

def trans_with_rt(source_points,R,t):
    target_points = np.dot(R,source_points.T).T + np.array(t)
    return target_points

def trans_with_st(source_points,s,t):
    target_points = s * source_points + t
    return target_points

scan = 40
input_path_2 = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan40/sparse/vggt/points3D_vggt_mesh.ply"
input_path_3 = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/ours_1/fuse_post.ply"
input_path = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/ours_1000/fuse_post.ply"

output_path_2 = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/vggt_align_culled.ply"
output_path_3 = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/ours_1_align_culled.ply"
output_path = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/ours_1000_align_culled.ply"

input_ply = o3d.io.read_point_cloud(input_path)
input_ply_2 = o3d.io.read_point_cloud(input_path_2)
input_ply_3 = o3d.io.read_point_cloud(input_path_3)

s,R,t,source_ply,colmap_ply = align1_camera_pose(scan,input_path)
# 和colmap位姿对齐 但是由于VGGT的位姿和内参没有那么准 所以还需要配准
input_ply_posealign_numpy = trans_with_rst(np.array(input_ply.points),np.array(s),np.array(R),np.array(t))
input_ply_2_posealign_numpy = trans_with_rst(np.array(input_ply_2.points),np.array(s),np.array(R),np.array(t))
input_ply_3_posealign_numpy = trans_with_rst(np.array(input_ply_3.points),np.array(s),np.array(R),np.array(t))



_,s,t,gt_ply = align1_rescale(scan,input_ply_posealign_numpy)
# 因为DTU的colmap和GT里面相差了一个尺度 所以rescale 在这里就理论和gt对齐了 但是由于VGGT误差还需要一个配准
input_ply_rescale_numpy = trans_with_st(input_ply_posealign_numpy,s,t)
input_ply_2_rescale_numpy = trans_with_st(input_ply_2_posealign_numpy,s,t)
input_ply_3_rescale_numpy = trans_with_st(input_ply_3_posealign_numpy,s,t)

T = align1_reg(input_ply_rescale_numpy,np.array(gt_ply.points))
R = T[:3,:3]
t = T[:3,3]
input_ply_reg_numpy = trans_with_rt(input_ply_rescale_numpy,R,t)
input_ply_2_reg_numpy = trans_with_rt(input_ply_2_rescale_numpy,R,t)
input_ply_3_reg_numpy = trans_with_rt(input_ply_3_rescale_numpy,R,t)
# vis_o3d_pcd_3(np.array(gt_ply.points),input_ply_reg_numpy,input_ply_2_reg_numpy,color1=[1,0,0],color2=[0,1,0],color3=[0,0,1],down=10000)

input_ply_regmasked_numpy = mask_dtu(scan,input_ply_reg_numpy)
input_ply_2_regmasked_numpy = mask_dtu(scan,input_ply_2_reg_numpy)
# input_ply_3_regmasked_numpy = mask_dtu(scan,input_ply_3_reg_numpy)
# 用DTU的mask裁剪点云

o3d.io.write_point_cloud(output_path,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_regmasked_numpy)),write_ascii=True)
o3d.io.write_point_cloud(output_path_2,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_2_regmasked_numpy)),write_ascii=True)
# o3d.io.write_point_cloud(output_path_3,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_3_regmasked_numpy)),write_ascii=True)

os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --vis_out_dir pilianghua_out/gs_init/pilianghua_output_gsinit/scan{scan}/train/eval_ours_1")
os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path_2} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --vis_out_dir pilianghua_out/gs_init/pilianghua_output_gsinit/scan{scan}/train/eval_vggt")
# os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path_3} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU--vis_out_dir pilianghua_out/gs_init/pilianghua_output_gsinit/scan{scan}/train/eval_ours_500")







# vis_o3d_pcd_2(np.array(gt_ply.points),input_ply_regmasked_numpy,color1=[1,0,0],color2=[0,1,0],down=10000)



# vis_o3d_pcd_2(input_ply_rescale_numpy,np.array(gt_ply.points),color1=[1,0,0],color2=[0,1,0],down=10000)
# vis_o3d_pcd_3(np.array(colmap_ply.points),input_ply_posealign_numpy,input_ply_2_posealign_numpy,color1=[1,0,0],color2=[0,1,0],color3=[0,0,1],down=10000)



print("end")

