import shutil
import os
guanfang_depth_dir = "DTU/set_23_24_33"
ours_depth_dir = "DTU/diff/set_23_24_33"



for scan in os.listdir(guanfang_depth_dir):
    
    if not scan in os.listdir(ours_depth_dir):
        print(scan,"continue")
        continue
    
    guanfang_dir = os.path.join(guanfang_depth_dir,scan)
    ours_dir= os.path.join(ours_depth_dir,scan)

    guanfang_dir_sparse = os.path.join(guanfang_dir,"sparse/0")
    ours_dir_sparse = os.path.join(ours_dir,"sparse/origin")

    shutil.copytree(guanfang_dir_sparse, ours_dir_sparse,dirs_exist_ok=True)
    # try:
    #     shutil.rmtree( os.path.join(ours_dir,scan))
    # except:
    #     pass
    print(f"Copied {guanfang_dir_sparse} to {ours_dir_sparse}")

print("0")