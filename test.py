import pandas as pd
import json
import os

def json_to_excel(json_file_path, excel_file_path=None):
    """
    读取JSON文件中的图像评估数据，按迭代次数排序后保存为Excel
    
    参数:
        json_file_path: JSON文件的路径
        excel_file_path: 输出Excel文件的路径，默认与JSON同目录同名称
    """
    # 处理输出路径
    if excel_file_path is None:
        # 生成与JSON文件同名的Excel路径
        file_dir, file_name = os.path.split(json_file_path)
        base_name = os.path.splitext(file_name)[0]
        excel_file_path = "汇总结果.xlsx"
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：{json_file_path} 不是有效的JSON文件")
        return
    
    # 解析数据并提取迭代次数
    parsed_data = []
    for key, metrics in data.items():
        # 从"ours_xxx"中提取数字作为迭代次数
        try:
            iteration = int(key.split("_")[1])
            parsed_data.append({
                "迭代次数": iteration,
                "SSIM": metrics["SSIM"],
                "PSNR": metrics["PSNR"],
                "LPIPS": metrics["LPIPS"]
            })
        except (IndexError, ValueError):
            print(f"警告：跳过无效格式的键 {key}")
    
    # 按迭代次数排序
    parsed_data.sort(key=lambda x: x["迭代次数"])
    
    # 转换为DataFrame并保存为Excel
    df = pd.DataFrame(parsed_data)
    df.to_excel(excel_file_path, index=False)
    print(f"数据已成功保存到 {excel_file_path}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_path = "PACG_VIS/iter/origin_office0_sparse/results_psnr.json"  # 可以是绝对路径（如"C:/data/metrics.json"）或相对路径
    json_to_excel(json_path)