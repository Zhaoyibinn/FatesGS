import os
import cv2
import numpy as np
import pandas as pd

origin_path_root_dir = "pilianghua_out_new_rep"
gt_path = "Replica/replica_gsicpslam"

# 初始化数据结构，以扫描名称为行，场景为列
scans = ["office2_sparse", "room1_sparse", "office0_sparse", "office1_sparse", "room0_sparse"]
scenes = ["3DGS", "2DGS", "origin", "dust3r_abs_trim_splitmix_diff", "origin"]

# 创建一个空的DataFrame来存储平均误差
error_data = pd.DataFrame(index=scans, columns=scenes)

for scene in scenes:
    origin_path_root = os.path.join("pilianghua_out_new_rep", scene)
    for scan_name in scans:
        error_average = 0
        
        # 计算三张图像的平均误差
        for idx in range(3):
            # 读取GT深度图和渲染深度图
            gt_img = cv2.imread(os.path.join(gt_path, scan_name, f'depth/000{idx}_depth.png'), cv2.IMREAD_ANYDEPTH)
            render_img = cv2.imread(os.path.join(origin_path_root, scan_name, f"train/ours_15000/vis/depth_0000{idx}.tiff"), cv2.IMREAD_ANYDEPTH)
            
            # 调整GT图像大小并归一化
            reshape_gt_img = cv2.resize((gt_img / 6553.5), (render_img.shape[1], render_img.shape[0]))
            
            # 计算误差
            error = np.abs(reshape_gt_img - render_img).mean()
            error_average += error
        
        # 计算平均误差并存储到DataFrame
        avg_error = error_average / 3
        error_data.loc[scan_name, scene] = avg_error
        print(f"{scene}: {scan_name} error = {avg_error}")

# 保存到Excel文件
error_data.to_excel("depth_error_results.xlsx", sheet_name="误差汇总")

print("误差数据已成功保存到 depth_error_results.xlsx")