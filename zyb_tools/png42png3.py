import cv2
import os
import argparse

def convert_png_to_three_channels(input_folder, output_folder):
    # 若输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # 检查文件是否为PNG格式
        if not filename.lower().endswith('.png'):
            print(f"跳过非PNG文件: {filename}")
            continue
        
        # 读取PNG文件（以四通道模式读取）
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        # 验证图像是否为四通道
        if img is None:
            print(f"无法读取文件: {filename}")
            continue
        if len(img.shape) != 3 or img.shape[2] != 4:
            print(f"跳过非四通道图像: {filename}")
            continue
        
        # 将四通道图像转为三通道
        # 方法1：舍弃Alpha通道
        # img_rgb = img[:, :, :3]
        
        # 方法2：考虑Alpha通道进行融合（推荐）
        b, g, r, a = cv2.split(img)
        a = a / 255.0
        b = (b * a).astype(img.dtype)
        g = (g * a).astype(img.dtype)
        r = (r * a).astype(img.dtype)
        img_rgb = cv2.merge([b, g, r])
        
        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img_rgb)
        print(f"已转换并保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='将四通道PNG图像转为三通道')
    parser.add_argument('--input', default="/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_23_24_33/scan37_metric/images_4", help='输入文件夹路径')
    parser.add_argument('--output', default="/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_23_24_33/scan37_metric/images", help='输出文件夹路径')
    args = parser.parse_args()
    
    convert_png_to_three_channels(args.input, args.output)

if __name__ == "__main__":
    main()