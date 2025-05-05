import shutil
import os
guanfang_depth_dir = "DTU/set_23_24_33/depth/set_23_24_33"
ours_depth_dir = "DTU/diff"



for scan in os.listdir(guanfang_depth_dir):
    if not scan in os.listdir(ours_depth_dir):
        continue
    guanfang_dir = os.path.join(guanfang_depth_dir,scan)
    ours_dir= os.path.join(ours_depth_dir,scan)

    os.remove(os.path.join(ours_dir,"depth_npy","0000_pred.npy"))
    os.remove(os.path.join(ours_dir,"depth_npy","0024_pred.npy"))
    os.remove(os.path.join(ours_dir,"depth_npy","0048_pred.npy"))

    shutil.copy(os.path.join(guanfang_dir,"depth_npy","0000_pred.npy"),os.path.join(ours_dir,"depth_npy","0000_pred.npy"))
    shutil.copy(os.path.join(guanfang_dir,"depth_npy","0001_pred.npy"),os.path.join(ours_dir,"depth_npy","0024_pred.npy"))
    shutil.copy(os.path.join(guanfang_dir,"depth_npy","0002_pred.npy"),os.path.join(ours_dir,"depth_npy","0048_pred.npy"))

print("0")