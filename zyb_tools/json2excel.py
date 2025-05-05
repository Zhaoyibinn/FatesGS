import json
import pandas as pd
import os

def custom_sort_key(x):
    return int(x[3:]) 

excel_file = "test.xlsx"
root_path = "pilianghua_output/origin"
main_columns = ["origin", "ssimdiff","alllossdiff", "ssimdiff_manyrender","ssimdiff_person"]
# main_columns = ["origin", "ssimdiff"]






all_all_data = []

for i in main_columns:
    all_data = []
    root_path = os.path.join("pilianghua_output",i)
    for sense in sorted(os.listdir(root_path), key=custom_sort_key):
        
        json_file = os.path.join(root_path,sense,"results.json")
        # "pilianghua_output/origin/dtu37/results.json"
        
        data_1 = {'data': sense}

        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
        except:
            data = []
        # data = json.load(json_file)
        # data["data"] = json_file.split("/")[-2]
        for item in [data]:
            data_1.update(item)
        del data_1["mean_d2s"]
        del data_1["mean_s2d"]
        all_data.append(data_1)
    all_data_df = pd.DataFrame(all_data)
    all_data_df.columns = pd.MultiIndex.from_product([[i], all_data_df.columns])
    all_all_data.append(all_data_df)


with pd.ExcelWriter(excel_file) as writer:
    # 将四个 DataFrame 按列拼接
    combined_df = pd.concat(all_all_data, axis=1)
    combined_df.to_excel(writer, sheet_name='Sheet1')



# excel_file_path = 'output.xlsx'
# try:
#     combined_df.to_excel(excel_file_path, index=False)
#     print(f"数据已成功写入 {excel_file_path}")
# except Exception as e:
#     print(f"写入Excel文件时出现错误: {e}") 


# print("0")