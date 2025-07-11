import os
import json
import pandas as pd

# 定义根目录
root_dir = "pilianghua_out_new_rep_psnr"

# 创建一个空的DataFrame来存储所有数据
all_data = []

# 获取所有方法文件夹
method_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# 获取所有场景（假设所有方法下的场景是相同的）
first_method = method_folders[0] if method_folders else ""
first_method_path = os.path.join(root_dir, first_method)
scene_folders = [s for s in os.listdir(first_method_path) if os.path.isdir(os.path.join(first_method_path, s))]

# 遍历每个方法文件夹
for method in method_folders:
    method_path = os.path.join(root_dir, method)
    
    # 遍历每个场景文件夹
    for scene in scene_folders:
        scene_path = os.path.join(method_path, scene)
        json_path = os.path.join(scene_path, "results_psnr.json")
        
        # 检查JSON文件是否存在
        if os.path.exists(json_path):
            try:
                # 读取JSON文件
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # 只处理JSON中最后一个键对应的内容
                last_key = list(data.keys())[-1]
                for key, value in data[last_key].items():
                    all_data.append({
                        '算法': method,
                        '指标': key,
                        '场景': scene,
                        '值': value
                    })
            except Exception as e:
                print(f"处理文件 {json_path} 时出错: {e}")
        else:
            print(f"文件不存在: {json_path}")

# 创建DataFrame
df = pd.DataFrame(all_data)

# 创建分层列结构
# 先按算法分组，然后按场景和指标重塑
pivot_df = df.pivot(index='算法', columns=['场景', '指标'], values='值')

# 保存到Excel
excel_path = "汇总结果.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # 写入原始数据
    df.to_excel(writer, sheet_name='原始数据', index=False)
    
    # 写入透视表数据（分层列结构）
    pivot_df.to_excel(writer, sheet_name='分层列结构')

print(f"数据已成功保存到 {excel_path}")    