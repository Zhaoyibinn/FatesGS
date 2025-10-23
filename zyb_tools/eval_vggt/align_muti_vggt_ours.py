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


import argparse

parser = argparse.ArgumentParser(description="你的脚本说明")
# parser.add_argument('--scans', type=int, nargs='+', help='要处理的scan编号列表')
parser.add_argument('--input_root', type=str, required=True, help='输入根目录')
parser.add_argument('--vggt_root', type=str, default="pilianghua_out/gs_init/DTU/gsinit/vggt_pcd")
args = parser.parse_args()



# for scan in [24 ,37 ,40 ,55 ,63 ,65 ,69 ,83, 97,105, 106, 110, 114, 118, 122]:

for scan in [24]:
    # scan = 24
    # vggt_origin_root = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt"
    vggt_origin_root = args.vggt_root
    input_path_root = args.input_root
    output_path_root = input_path_root

    input_path_vggt = os.path.join(vggt_origin_root,f"scan{scan}/train/ours_1/fuse_post.ply")
    input_path = os.path.join(input_path_root,f"scan{scan}/train/ours_1000/fuse_post.ply")
    input_path_2 = os.path.join(input_path_root,f"scan{scan}/train/ours_1/fuse_post.ply")

    output_path_vggt = os.path.join(output_path_root,f"scan{scan}/train/vggt_align_culled.ply")
    output_path = os.path.join(output_path_root,f"scan{scan}/train/ours_1000_align_culled.ply")
    output_path_2 = os.path.join(output_path_root,f"scan{scan}/train/ours_1_align_culled.ply")

    if not os.path.exists(input_path_vggt):
        print(f"vggt mesh 不存在: {input_path}")
        assert os.path.exists( os.path.join(vggt_origin_root,f"scan{scan}/sparse/vggt/points3D.ply")), print(f"vggt pointcloud也不存在 有问题 请检查")
        input_ply_vggt_pcd = o3d.io.read_point_cloud(os.path.join(vggt_origin_root,f"scan{scan}/sparse/vggt/points3D.ply"))
        alpha = 0.005  # 调整 alpha 值以控制网格的细节程度
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(input_ply_vggt_pcd, alpha)
        o3d.io.write_triangle_mesh(input_path_vggt, mesh)
        print("生成vggt mesh完成")
        

    input_ply = o3d.io.read_point_cloud(input_path)
    input_ply_vggt = o3d.io.read_point_cloud(input_path_vggt)
    input_ply_2 = o3d.io.read_point_cloud(input_path_2)

    s,R,t,source_ply,colmap_ply,_,_ = align1_camera_pose(scan,input_path)
    # 和colmap位姿对齐 但是由于VGGT的位姿和内参没有那么准 所以还需要配准
    input_ply_posealign_numpy = trans_with_rst(np.array(input_ply.points),np.array(s),np.array(R),np.array(t))
    input_ply_vggt_posealign_numpy = trans_with_rst(np.array(input_ply_vggt.points),np.array(s),np.array(R),np.array(t))
    input_ply_2_posealign_numpy = trans_with_rst(np.array(input_ply_2.points),np.array(s),np.array(R),np.array(t))



    _,s,t,gt_ply = align1_rescale(scan,input_ply_posealign_numpy)
    # 因为DTU的colmap和GT里面相差了一个尺度 所以rescale 在这里就理论和gt对齐了 但是由于VGGT误差还需要一个配准
    input_ply_rescale_numpy = trans_with_st(input_ply_posealign_numpy,s,t)
    input_ply_vggt_rescale_numpy = trans_with_st(input_ply_vggt_posealign_numpy,s,t)
    input_ply_2_rescale_numpy = trans_with_st(input_ply_2_posealign_numpy,s,t)

    T = align1_reg(input_ply_rescale_numpy,np.array(gt_ply.points))
    R = T[:3,:3]
    t = T[:3,3]

    input_ply_reg_numpy = trans_with_rt(input_ply_rescale_numpy,R,t)
    input_ply_vggt_reg_numpy = trans_with_rt(input_ply_vggt_rescale_numpy,R,t)
    input_ply_2_reg_numpy = trans_with_rt(input_ply_2_rescale_numpy,R,t)
    vis_o3d_pcd_3(np.array(gt_ply.points),input_ply_reg_numpy,input_ply_vggt_reg_numpy,color1=[1,0,0],color2=[0,1,0],color3=[0,0,1],down=10000)

    input_ply_regmasked_numpy = mask_dtu(scan,input_ply_reg_numpy)
    input_ply_vggt_regmasked_numpy = mask_dtu(scan,input_ply_vggt_reg_numpy)
    input_ply_2_regmasked_numpy = mask_dtu(scan,input_ply_2_reg_numpy)
    # 用DTU的mask裁剪点云

    o3d.io.write_point_cloud(output_path,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_regmasked_numpy)),write_ascii=True)
    o3d.io.write_point_cloud(output_path_vggt,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_vggt_regmasked_numpy)),write_ascii=True)
    o3d.io.write_point_cloud(output_path_2,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_ply_2_regmasked_numpy)),write_ascii=True)


    eval_cd_savedir_root = os.path.join(output_path_root,f"scan{scan}/train")
    eval_cd_savedir_vggt = os.path.join(eval_cd_savedir_root,"eval_vggt")
    eval_cd_savedir = os.path.join(eval_cd_savedir_root,"eval_ours_1000")
    eval_cd_savedir_2 = os.path.join(eval_cd_savedir_root,"eval_ours_1")

    os.makedirs(eval_cd_savedir_vggt,exist_ok=True)
    os.makedirs(eval_cd_savedir,exist_ok=True)
    os.makedirs(eval_cd_savedir_2,exist_ok=True)

    os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path_vggt} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --vis_out_dir {eval_cd_savedir_vggt}")
    os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --vis_out_dir {eval_cd_savedir}")
    os.system(f"python zyb_tools/eval_dtu/eval.py --data {output_path_2} --scan {scan} --mode pcd --dataset_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --vis_out_dir {eval_cd_savedir_2}")







# vis_o3d_pcd_2(np.array(gt_ply.points),input_ply_regmasked_numpy,color1=[1,0,0],color2=[0,1,0],down=10000)



# vis_o3d_pcd_2(input_ply_rescale_numpy,np.array(gt_ply.points),color1=[1,0,0],color2=[0,1,0],down=10000)
# vis_o3d_pcd_3(np.array(colmap_ply.points),input_ply_posealign_numpy,input_ply_2_posealign_numpy,color1=[1,0,0],color2=[0,1,0],color3=[0,0,1],down=10000)



print("end")

