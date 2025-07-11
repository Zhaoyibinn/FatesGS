import os
import cv2
import numpy as np


origin_path_root = "Replica/replica_gsicpslam"
result_path = "DACG_VIS/REPLICA_depth"


for scan_name in ["office2_sparse","room1_sparse","office0_sparse","office1_sparse","room0_sparse"]:

    for image_name in ["0000_depth.png","0001_depth.png","0002_depth.png"]:
        img_origin = cv2.imread(os.path.join(origin_path_root,scan_name,"depth",image_name),cv2.IMREAD_ANYDEPTH)

        img_real = img_origin / 6553.5

        img_color = cv2.applyColorMap(np.uint8(np.clip(img_real * 70,0,255)), cv2.COLORMAP_JET)
        result_img_root = os.path.join(result_path,scan_name)
        result_img_path = os.path.join(result_img_root,"gt_" + image_name)
        cv2.imwrite(result_img_path,img_color)
    # print("end")