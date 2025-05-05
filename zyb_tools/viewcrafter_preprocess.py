import os
import shutil
import open3d as o3d
import cv2

def copy_folder(source_folder, destination_folder):
    try:
        # 复制文件夹
        shutil.copytree(source_folder, destination_folder,dirs_exist_ok=True)
        # print(f"成功将 {source_folder} 复制到 {destination_folder}")
    except FileExistsError:
        print(f"目标文件夹 {destination_folder} 已存在。")
    except Exception as e:
        print(f"复制文件夹时出现问题: {e}")


def check_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False
    

def MP42PNG(video_path,images_dir):
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
    else:
        frame_count = 0
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 生成存储帧的文件名
                frame_filename = os.path.join(images_dir, f'{frame_count:04d}.png')
                # 保存帧为图像文件
                frame = cv2.resize(frame, (1554, 1162), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            else:
                # 无法读取帧，退出循环
                break

        # 释放视频捕获对象
        cap.release()
        print(f"已保存 {frame_count} 帧到 {images_dir} 文件夹中")

def rename_images(folder_path):

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    image_files.sort()
    for i, file in enumerate(image_files):
        # 生成新的文件名
        new_name = f"{i:04d}{os.path.splitext(file)[1]}"
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        # print(f"已将 {old_file_path} 重命名为 {new_file_path}")

for idx in [83,97,105,106,110,114,118,122]:
    

    view_crafter_path = f"DTU/diff/DTU_3/scan{idx}"
    colmap_path = f"DTU/diff/scan{idx}"
    colmap_moban_path = "DTU/diff/moban"
    ori_colmap_path = f"DTU/set_23_24_33/scan{idx}"
    # check_mkdir(colmap_path)

    copy_folder(colmap_moban_path,colmap_path)

    diff_video_path = os.path.join(view_crafter_path,"diffusion.mp4")
    images_txt_path = os.path.join(view_crafter_path,"images_inter.txt")
    dust3r_pcd_path = os.path.join(view_crafter_path,"pcd.ply")

    colmap_images_path = os.path.join(colmap_path,"images")
    MP42PNG(diff_video_path,colmap_images_path)
    ori_colmap_images_path = os.path.join(ori_colmap_path,"images")

    os.remove(os.path.join(colmap_images_path,"0000.png"))
    os.remove(os.path.join(colmap_images_path,"0024.png"))
    os.remove(os.path.join(colmap_images_path,"0025.png"))
    os.remove(os.path.join(colmap_images_path,"0049.png"))

    shutil.copy(os.path.join(ori_colmap_images_path,"0000.png"),os.path.join(colmap_images_path,"0000.png"))
    shutil.copy(os.path.join(ori_colmap_images_path,"0001.png"),os.path.join(colmap_images_path,"0024.png"))
    shutil.copy(os.path.join(ori_colmap_images_path,"0002.png"),os.path.join(colmap_images_path,"0049.png"))

    rename_images(colmap_images_path)
    print("图片复制并操作完成")

    shutil.copy(images_txt_path,os.path.join(colmap_path,"sparse","model","images.txt"))
    print("colmap位姿复制完成")

    os.chdir(colmap_path)
    if os.path.exists("database.db"):
        os.remove("database.db")
    os.system("bash colmap.bash")
    print("colmap稀疏重建完成")
    os.chdir("../../..")

    shutil.copy(dust3r_pcd_path,os.path.join(colmap_path,"sparse","0","points3D.ply"))
    print("pcd复制完成")

    current_path = os.getcwd()
    current_colmap_path=os.path.join(current_path,colmap_path)
    current_colmap_images_path=os.path.join(current_colmap_path,"images")
    current_colmap_depth_path=os.path.join(current_colmap_path,"depth_npy_all")

    os.chdir("/home/zhaoyibin/3DRE/Marigold")
    py_path = "/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
    check_mkdir(current_colmap_depth_path)
    os.system(f"{py_path} run.py --input_rgb_dir {current_colmap_images_path} --output_dir {current_colmap_depth_path}")
    os.chdir(current_path)

    copy_folder(os.path.join(colmap_path,"depth_npy_all","depth_npy"),os.path.join(colmap_path,"depth_npy"))

