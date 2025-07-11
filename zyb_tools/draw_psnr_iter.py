import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter, NullLocator
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

def read_from_json(json_file_path):
        # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：文件 '{json_file_path}' 不存在")
        return
    
    # 读取JSON数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：文件 '{json_file_path}' 不是有效的JSON格式")
        return
    
    # 提取并排序数据
    iterations = []
    psnr_values = []
    iterations_after = []
    psnr_values_after = []
    # iterations_after = []
    # psnr_values_after = []
    for key, metrics in data.items():
        try:
            if int(key.split('_')[1]) == 1:
                continue
            if int(key.split('_')[1]) in [2000,3000,4000,6000,7000,8000,10000,11000,13000,14000]:
                continue
            
            iteration = int(key.split('_')[1])
            psnr = metrics.get('PSNR')
            if psnr is not None:
                if int(key.split('_')[1])==1000:
                    iterations_after.append(iteration)
                    psnr_values_after.append(psnr)
                    iterations.append(iteration)
                    psnr_values.append(psnr)
                
                elif int(key.split('_')[1])==15000:
                   
                    iterations_after.append(iteration)
                    psnr_values_after.append(psnr)
                elif int(key.split('_')[1])<1000:
                    iterations.append(iteration)
                    psnr_values.append(psnr)
        except (ValueError, IndexError):
            continue
    
    if not iterations:
        print("错误：未找到有效数据")
        return
    
    # 排序数据
    sorted_data = sorted(zip(iterations, psnr_values), key=lambda x: x[0])
    iterations, psnr_values = zip(*sorted_data)
    try:
        sorted_data_after = sorted(zip(iterations_after, psnr_values_after), key=lambda x: x[0])
        iterations_after, psnr_values_after = zip(*sorted_data_after)
    except:
        pass
    return iterations, psnr_values,iterations_after,psnr_values_after



def plot_psnr_from_json(json_file_path, json_file_ours_path,save_path="psnr_plot.png"):
    """从JSON文件读取数据并绘制PSNR曲线，确保保存的图片无科学计数法"""
    iterations, psnr_values,iterations_after,psnr_values_after = read_from_json(json_file_path)
    iterations_ours, psnr_values_ours,iterations_after_ours,psnr_values_after_ours = read_from_json(json_file_ours_path)

    # x_smooth = np.linspace(np.array(iterations).min(), np.array(iterations).max(), 300)
    # spl = make_interp_spline(iterations, psnr_values, k=2)
    # y_smooth = spl(x_smooth)

    # 创建图表
    plt.figure(figsize=(8, 6), dpi=100)  # 提高dpi确保清晰度
    ax = plt.gca()  # 获取当前轴对象
    
    # 绘制曲线
    ax.plot(iterations, psnr_values, 'o-', color='red', linewidth=2, markersize=6,label="2DGS")
    ax.plot(iterations_ours, psnr_values_ours, 'o-', color='blue', linewidth=2, markersize=6,label="PACG")

    ax.plot(iterations_after, psnr_values_after, '--', color='gray', linewidth=2, markersize=6)
    ax.plot(iterations_after, psnr_values_after, 'o', color='red', markersize=6)
    ax.plot(iterations_after_ours, psnr_values_after_ours, '--', color='gray', linewidth=2, markersize=6)
    ax.plot(iterations_after_ours, psnr_values_after_ours, 'o', color='blue', markersize=6)
    # ax.plot(x_smooth, y_smooth, color='blue', linewidth=2, markersize=6)
    
    # 设置标题和标签
    # ax.set_title('PSNR', fontsize=16)
    ax.set_xlabel('Training Iteration', fontsize=18)
    ax.set_ylabel('PSNR (dB)', fontsize=18)
    
    # 网格设置
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 关键：彻底禁用科学计数法（针对对数轴和线性轴都有效）
    ax.set_xscale('log')
    ax.xaxis.set_minor_locator(NullLocator())   
    # 使用ScalarFormatter并强制禁用科学计数法
    custom_ticks = [100, 200, 300, 400, 500, 700, 1000, 15000]
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_ticks, rotation=45, ha='right', fontsize=15) 
    formatter = ScalarFormatter()
    formatter.set_scientific(False)  # 禁用科学计数法
    formatter.set_useOffset(False)   # 禁用偏移量


    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)


    
    # 标注数据点（可选，密集时可能重叠）
    for x, y in zip(iterations, psnr_values):
        if not x in [100,400,700,1000,15000]:
            continue
        ax.annotate(f'{y:.2f}', (x, y), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)
        
    for x, y in zip(iterations_after, psnr_values_after):
        if not x in  [100,400,700,1000,15000]:
            continue
        ax.annotate(f'{y:.2f}', (x, y), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)
    for x, y in zip(iterations_after_ours, psnr_values_after_ours):
        if not x in [100,400,700,1000,15000]:
            continue
        ax.annotate(f'{y:.2f}', (x, y), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)
    for x, y in zip(iterations_ours, psnr_values_ours):
        if not x in [100,400,700,1000,15000]:
            continue
        ax.annotate(f'{y:.2f}', (x, y), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)
    ax.legend(loc='lower right',fontsize=20)
    # 调整布局
    plt.tight_layout()
    
    # 保存图像（确保保存时格式正确）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches确保标签不被截断
    print(f"图像已保存至：{save_path}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    json_file_path = "PACG_VIS/iter/2DGS_office2_sparse/results_psnr_origin.json"  # 替换为你的JSON文件路径
    json_file_ours_path = "PACG_VIS/iter/PACG_office2_sparse/results_psnr_origin.json"
    plot_psnr_from_json(json_file_path,json_file_ours_path, save_path="test.png")  # 保存路径可自定义