import os
import shutil
import open3d as o3d
import cv2
import numpy as np
import struct
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scene.colmap_loader import *

origin_path = "BMVS_colmap/BlendedMVS-colmap"
output_path = "BMVS_PACG/selected"
# dir_name = ["5aa235f64a17b335eeaf9609","5a69c47d0d5d0a7f3b2e9752","5a6464143d809f1d8208c43c","59338e76772c3e6384afbb15","5947b62af1b45630bd0c2a02"]
# img_diff = [[11,2,25],[23,62,21],[55,61,73],[24,65,169],[11,13,67]]
dir_name = ["5a57542f333d180827dfc132","5b950c71608de421b1e7318f","5b4933abf2b5f44e95de482a","5b22269758e2823a67a3bd03","5ba19a8a360c7c30c1c169df","5bccd6beca24970bce448134","5bce7ac9ca24970bce4934b6","5be47bf9b18881428d8fbc1d"]
img_diff = [[1,2,14],[11,13,15],[9,11,20],[5,7,8],[0,3,11],[4,7,14],[9,23,28],[4,5,29]]
for idx,dir_name1 in enumerate(dir_name):
    img_diff1 = img_diff[idx]
    bin_path = os.path.join(origin_path,dir_name1,"sparse/0/images.bin")
    cameras_bin_path = os.path.join(origin_path,dir_name1,"sparse/0/cameras.bin")
    # bin_path = os.path.join(origin_path,dir_name1,"sparse/0/images.bin")
    if not os.path.exists(os.path.join(output_path,dir_name1,"sparse/0")):
        os.makedirs(os.path.join(output_path,dir_name1,"sparse/0"))
    txt_path = os.path.join(output_path,dir_name1,"sparse/0/images.txt")
    cameras_txt_path = os.path.join(output_path,dir_name1,"sparse/0/cameras.txt")
    points3D_txt_path = os.path.join(output_path,dir_name1,"sparse/0/points3D.txt")

    ixt = read_intrinsics_binary(cameras_bin_path)
    colmap_str_list = ["1","PINHOLE",str(ixt[1].width),str(ixt[1].height),str(ixt[1].params).replace("[","").replace("]","").replace("   "," ").replace("  "," ")]
    colmap_strs = [" ".join(colmap_str_list)]

    try:
        with open(cameras_txt_path, 'w', encoding='utf-8') as file:
            file.write("# Camera list with one line of data per camera:" + '\n')
            file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" + '\n')
            file.write("# Number of cameras: 1" + '\n')

            for line in colmap_strs:
                file.write(line + '\n')
                file.write('\n')
        print(f"colmap相机内参已经写入 {txt_path}")
    except Exception as e:
        print(f"写入文件时出现错误: {e}")

    with open(points3D_txt_path, 'w') as file:
        pass  # 不执行任何写入操作，创建空文件
    print(f"新建了空的: {points3D_txt_path}")




    ext = read_extrinsics_binary(bin_path)
    qs , ts , ids , img_names = [],[],[],[]
    for i in range(len(ext)):
        idx = i + 1
        q = ext[idx].qvec
        qs.append(q)
        t = ext[idx].tvec
        ts.append(t)
        id = ext[idx].id
        ids.append(id)
        img_name = ext[idx].name
        img_names.append(img_name)

    # txt_path = self.opts.save_dir + "/images_inter.txt"
    colmap_strs = []
    # Rs,Ts = camera_traj.R,camera_traj.T
    for ii in range(len(ext)):
        q_colmap_str = " ".join(["{:06f}".format(i) for i in qs[ii]])
        t_colmap_str = " ".join(["{:06f}".format(i) for i in ts[ii]])
        idx_str = str(ids[ii])
        camera_id = "1"
        img_name = img_names[ii]
        if int(img_name.split(".")[0]) not in img_diff1:
            continue
        img_name = "{:04d}".format(img_diff1.index(int(img_name.split(".")[0])))+".png"
        colmap_str_list = [idx_str , q_colmap_str , t_colmap_str , camera_id , img_name]
        colmap_str = " ".join(colmap_str_list)
        # print(colmap_str)
        colmap_strs.append(colmap_str)

    try:
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write("# Image list with two lines of data per image:" + '\n')
            file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" + '\n')
            file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)" + '\n')
            file.write("# Number of images: 49, mean observations per image: 2172.8367346938776" + '\n')

            for line in colmap_strs:
                file.write(line + '\n')
                file.write('\n')
        print(f"colmap相机位姿已经写入 {txt_path}")
    except Exception as e:
        print(f"写入文件时出现错误: {e}")


    # 复制图片
    for ii in range(len(ext)):
        # q_colmap_str = " ".join(["{:06f}".format(i) for i in qs[ii]])
        # t_colmap_str = " ".join(["{:06f}".format(i) for i in ts[ii]])
        idx_str = str(ids[ii])
        # camera_id = "1"
        img_name = img_names[ii]
        if int(img_name.split(".")[0]) not in img_diff1:
            continue
        
        img_path = os.path.join(origin_path,dir_name1,"images",img_name)
        copyed_img_name = "{:04d}".format(img_diff1.index(int(img_name.split(".")[0])))+".png"
        copy_img_path = os.path.join(output_path,dir_name1,"images",copyed_img_name)
        if not os.path.exists(os.path.join(output_path,dir_name1,"images")):
            os.makedirs(os.path.join(output_path,dir_name1,"images"))
        shutil.copy(img_path,copy_img_path)

