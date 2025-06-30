import os
from PIL import Image
import argparse
import math

def split_image(input_path, output_dir=None, cols=8, save_individual=True, save_grid=True):
    """
    将RGB图像分割成小方块，每列固定为指定数量
    
    参数:
        input_path: 输入图像路径
        output_dir: 输出目录，默认为None(在原图像所在目录创建split_images文件夹)
        cols: 每列的方块数量，默认为8
        save_individual: 是否保存单个小方块，默认为True
        save_grid: 是否保存重组后的网格图像，默认为True
    """
    # 打开图像
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图像: {e}")
        return
    
    # 确保图像为RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 获取图像尺寸
    width, height = img.size
    
    # 计算每个小方块的尺寸
    block_height = height // cols
    # 根据列数计算行数，确保覆盖整个图像
    rows = math.ceil(width / block_height)
    block_width = width // rows
    
    # 创建输出目录
    if output_dir is None:
        base_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_dir, "split_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有小方块的列表
    blocks = []
    
    # 分割图像
    for row in range(rows):
        for col in range(cols):
            # 计算当前方块的位置
            left = row * block_width
            upper = col * block_height
            right = min((row + 1) * block_width, width)
            lower = min((col + 1) * block_height, height)
            
            # 如果超出图像范围则跳过
            if left >= width or upper >= height:
                continue
            
            # 裁剪方块
            block = img.crop((left, upper, right, lower))
            
            # 保存单个小方块
            if save_individual:
                block_filename = f"block_row{row}_col{col}.jpg"
                block_path = os.path.join(output_dir, block_filename)
                block.save(block_path)
            
            blocks.append(block)
    
    # 保存重组后的网格图像
    if save_grid and blocks:
        # 创建新的网格图像
        grid_width = block_width * rows
        grid_height = block_height * cols
        grid_img = Image.new('RGB', (grid_width, grid_height))
        
        # 填充网格
        index = 0
        for col in range(cols):
            for row in range(rows):
                if index < len(blocks):
                    grid_img.paste(blocks[index], (row * block_width, col * block_height))
                    index += 1
        
        # 保存网格图像
        grid_filename = f"grid_{os.path.basename(input_path)}"
        grid_path = os.path.join(output_dir, grid_filename)
        grid_img.save(grid_path)
    
    print(f"图像分割完成! 共生成 {len(blocks)} 个小方块")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将RGB图像分割成小方块')
    parser.add_argument('--input', default="DTU/origin/set_23_24_33/scan24/images/0001.png",help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出目录', default="vis_patch1")
    parser.add_argument('--cols', '-c', type=int, help='每列方块数量', default=8)
    parser.add_argument('--no-individual', action='store_false', help='不保存单个小方块', dest='save_individual')
    parser.add_argument('--no-grid', action='store_false', help='不保存重组后的网格图像', dest='save_grid')
    
    args = parser.parse_args()
    
    split_image(args.input, args.output, args.cols, args.save_individual, args.save_grid)

if __name__ == "__main__":
    main()    