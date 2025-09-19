import cv2
import numpy as np

def tiff_to_red_green_png(tiff_path, png_path):
    # 读取tiff深度图
    depth = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    # 归一化到0~255并转为uint8
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    # 构造红到绿的自定义色图（近红远绿）
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        lut[i, 0, 0] = 0      # Blue
        lut[i, 0, 1] = i      # Green（远处更绿）
        lut[i, 0, 2] = 255-i  # Red（近处更红）
    color_img = cv2.LUT(cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR), lut)
    cv2.imwrite(png_path, color_img)

# 用法示例
tiff_to_red_green_png("pilianghua_out/gs_init/gsinit/2dgsok_trim0_extrapose_lrq1e-5t1e-6/scan40/train/ours_1000/vis/depth_00000.tiff", "test.png")