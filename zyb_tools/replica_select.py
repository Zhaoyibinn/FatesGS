import os
import cv2

import shutil

# idx_num = [0,762,802] # office0_old



# idx_num = [752,812,840] # office0_old
# idx_num = [1714,1205,1031] #office1
# idx_num = [0,289,422] #office2
# idx_num = [0,174,520] #room0
idx_num = [639,1158,860] #room1

root_path = "Replica/replica_gsicpslam"
scene_name = "room1"

scene_sparse_name = scene_name + "_sparse_vggt"
scene_path = os.path.join(root_path,scene_name)
scene_sparse_path = os.path.join(root_path,scene_sparse_name)


    

colmap_img_path = os.path.join(scene_path,"images_colmap")
colmap_depth_path = os.path.join(scene_path,"depth_images")
colmap_img_sparse_path = os.path.join(scene_sparse_path,"images")
colmap_depth_sparse_path = os.path.join(scene_sparse_path,"depth_images")

if os.path.exists(scene_sparse_path):

    shutil.rmtree(scene_sparse_path)

os.mkdir(scene_sparse_path)
os.mkdir(colmap_img_sparse_path)
os.mkdir(colmap_depth_sparse_path)
os.makedirs(os.path.join(scene_sparse_path,"sparse","gt"), exist_ok=True)

final_idx = 0

for i in idx_num:
    img_name = f"{i}.png"
    depth_name = f"depth{i:06d}.png"
    img_path = os.path.join(colmap_img_path,img_name)
    depth_path = os.path.join(colmap_depth_path,depth_name)

    
    img_sparse_path = os.path.join(colmap_img_sparse_path,f"{final_idx:04d}.png")
    depth_sparse_path = os.path.join(colmap_depth_sparse_path,f"{final_idx:04d}_depth.png")
    final_idx += 1 
    shutil.copy(img_path,img_sparse_path)
    shutil.copy(depth_path,depth_sparse_path)


colmap_img_path_recon = os.path.join(scene_path,"sparse","gt")
colmap_img_sparse_path_recon = os.path.join(scene_sparse_path,"sparse","gt")
colmap_img_path_recon_cameras = os.path.join(colmap_img_path_recon,"cameras.txt")
colmap_img_sparse_path_recon_cameras = os.path.join(colmap_img_sparse_path_recon,"cameras.txt")
shutil.copy(colmap_img_path_recon_cameras,colmap_img_sparse_path_recon_cameras)

colmap_img_path_recon_points3D = os.path.join(colmap_img_path_recon,"points3D.txt")
colmap_img_sparse_path_recon_points3D = os.path.join(colmap_img_sparse_path_recon,"points3D.txt")
shutil.copy(colmap_img_path_recon_points3D,colmap_img_sparse_path_recon_points3D)



colmap_img_path_recon_images = os.path.join(colmap_img_path_recon,"images.txt")
colmap_img_sparse_path_recon_images = os.path.join(colmap_img_sparse_path_recon,"images.txt")

with open(colmap_img_path_recon_images, 'r', encoding='utf-8') as f:
    lines = f.readlines()

modified_lines = []
modified_lines.append("# Image list with two lines of data per image:\n")
modified_lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
modified_lines.append("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

final_idx = 1

for i in idx_num:
    for idx, element in enumerate(lines):
        # 转换为字符串后取最后一位
        if element == "\n":
            continue
        index = int(element.split()[-1].split(".")[0])
        if index == i:
            element = str(final_idx) + " " + element.split(' ', 1)[1]
            element = element.rsplit(' ', 1)[0] + f" {(final_idx-1):04d}.png\n"
            modified_lines.append(element)
            modified_lines.append("\n")

            final_idx += 1

with open(colmap_img_sparse_path_recon_images, 'w', encoding='utf-8') as f:
    f.writelines(modified_lines)










print("end")

