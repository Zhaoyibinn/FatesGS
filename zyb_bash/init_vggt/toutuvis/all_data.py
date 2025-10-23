import os
import json
import pandas as pd

# scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scans = [63]
folders = ['eval_ours_1', 'eval_ours_100', 'eval_ours_200','eval_ours_300', 'eval_ours_400','eval_ours_500', 'eval_ours_600','eval_ours_700', 'eval_ours_800','eval_ours_900', 'eval_ours_1000']
folder_nums = [int(f.split("_")[-1]) for f in folders]

results = {}

for scan in scans:
    base_dir = f'output/3dgs/scan{scan}/train'
    results[scan] = {}
    for folder in folders:
        json_path = os.path.join(base_dir, folder, 'results.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                overall_value = data.get('overall', None)
                folder_num = int(folder.split("_")[-1])
                results[scan][folder_num] = overall_value
        except:
            continue
df = pd.DataFrame(results, index=folder_nums)
df = df.sort_index()
df.to_excel('test.xlsx')