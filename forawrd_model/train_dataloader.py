import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import sys


sys.path.append("submodules/Viewcrafter")
sys.path.append("submodules/Viewcrafter/extern/dust3r")
from Dust3r_class import Dust3r


sys.path.append("submodules/vggt")
from vggt_class import vggt

from tqdm import tqdm




class forward_model_dataset(Dataset):
    def __init__(self, data_root,method="dust3r",val = False):
        # self.Dust3r_model  = Dust3r()
        self.VGGT_model = vggt()
        self.data_root = data_root
        self.scene_names = sorted(os.listdir(data_root))
        self.batch_data = []
        self.dtu_val_idx = [23,24,33]

        dtu_scene_idx = [24,37,40,55,63,65,69,83,97,105,106,110,114,118,122]
        
        # dtu_scene_idx = [24]

        for scene_name in tqdm(self.scene_names):
            scene_path = os.path.join(self.data_root,scene_name,"images")
            tqdm.write(f"Init {scene_path}")


            if int(scene_name.split("scan")[-1]) not in dtu_scene_idx:
                continue
            
              
            
            img_names = sorted(os.listdir(scene_path))
            img_names = [file for file in img_names if file.lower().endswith(".png")]


            first = torch.arange(len(img_names) - 1, dtype=torch.long)
            # first = torch.arange(3 - 1, dtype=torch.long)
            second = first + 1
            pairs = torch.stack([first, second], dim=1)
            img_pairs_idxs = pairs
            img_paths = []
            for img_name in img_names:
                # img = cv2.imread(os.path.join(self.data_root,img_name))
                img_paths.append(os.path.join(scene_path,img_name))
            # self.paired_imgs_paths = []
            

            
            for img_pair in tqdm(img_pairs_idxs, desc="Runing Forward Model", leave=False):
                if val:
                    if not (img_pair[0].item() in self.dtu_val_idx):
                        continue
                else:
                    if img_pair[0].item() in self.dtu_val_idx:
                        continue


                if method=="dust3r":
                    os.makedirs(os.path.join(scene_path,"dust3r"),exist_ok=True)
                    pair_name = f"pair_{img_pair[0].item()}_{img_pair[1].item()}.pth"
                    pair_path = os.path.join(scene_path,"dust3r",pair_name)
                    if os.path.exists(pair_path):
                        
                        load_tensor = torch.load(pair_path)
                        pcd0,pcd1,color0,color1 = load_tensor["pcd0"], load_tensor["pcd1"], load_tensor["color0"], load_tensor["color1"]
                        # pass
                    else:
                        img_1_idx,img_2_idx = img_pair[0].item(),img_pair[1].item()
                        img_1,img_2 = img_paths[img_1_idx],img_paths[img_2_idx]
                        # self.paired_imgs_paths.append([img_1,img_2])
                        full_out,pcd_np,align_model = self.Dust3r_model.run_only_model([img_1,img_2])
                        pcd0,pcd1 = align_model.get_pts3d()[0].detach(),align_model.get_pts3d()[1].detach()
                        color0,color1 = (full_out['view1']['img'].cuda().detach()+1)/2,(full_out['view2']['img'].cuda().detach()+1)/2
                        color0 = color0.permute(2,3,1,0).squeeze()
                        color1 = color1.permute(2,3,1,0).squeeze()
                        tensors_dict = {
                            "pcd0": pcd0,
                            "pcd1": pcd1,
                            "color0": color0,
                            "color1": color1
                            }
                        torch.save(tensors_dict, pair_path)
                    mask0,mask1 = None,None
                    self.batch_data.append([pcd0,pcd1,color0,color1,mask0,mask1])
                elif method=="vggt":
                    pair_name = f"pair_{img_pair[0].item()}_{img_pair[1].item()}.pth"
                    pair_path = os.path.join(scene_path,"vggt",pair_name)
                    os.makedirs(os.path.join(scene_path,"vggt"),exist_ok=True)
                    if os.path.exists(pair_path):
                        
                        load_tensor = torch.load(pair_path)
                        pcd0,pcd1,color0,color1,mask0,mask1,extrinsic_align1,intrinsic = load_tensor["pcd0"], load_tensor["pcd1"], load_tensor["color0"], load_tensor["color1"],load_tensor["mask0"],load_tensor["mask1"],load_tensor["extrinsic"],load_tensor["intrinsic"]
                        # pass
                    else:
                        img_1_idx,img_2_idx = img_pair[0].item(),img_pair[1].item()
                        img_1,img_2 = img_paths[img_1_idx],img_paths[img_2_idx]
                        points_3d,points_rgb,conf_mask,extrinsic_align1,intrinsic = self.VGGT_model.run_only_model([img_1,img_2])
                        points_3d,points_rgb,conf_mask,extrinsic_align1,intrinsic = torch.tensor(points_3d),torch.tensor(points_rgb),torch.tensor(conf_mask),torch.tensor(extrinsic_align1),torch.tensor(intrinsic)
                        pcd0,pcd1 = points_3d[0],points_3d[1]
                        color0,color1 = points_rgb[0]/255,points_rgb[1]/255
                        mask0,mask1 = conf_mask[0],conf_mask[1]

                        tensors_dict = {
                            "pcd0": pcd0,
                            "pcd1": pcd1,
                            "color0": color0,
                            "color1": color1,
                            "mask0":mask0,
                            "mask1":mask1,
                            "extrinsic":extrinsic_align1,
                            "intrinsic":intrinsic,

                            }
                        torch.save(tensors_dict, pair_path)

                    pcd0 = pcd0.permute(2,0,1).unsqueeze(0)
                    pcd1 = pcd1.permute(2,0,1).unsqueeze(0)
                    color0 = color0.permute(2,0,1).unsqueeze(0)
                    color1 = color1.permute(2,0,1).unsqueeze(0)
                    mask0 = mask0.unsqueeze(0).unsqueeze(0)
                    mask1 = mask1.unsqueeze(0).unsqueeze(0)

                    pcd_batch = torch.cat([pcd0, pcd1], dim=0).float()
                    color_batch = torch.cat([color0, color1], dim=0).float()
                    mask_batch = torch.cat([mask0, mask1], dim=0).float()
                    extrinsic_batch = extrinsic_align1
                    intrinsic_batch = intrinsic

                    self.batch_data.append([pcd_batch,color_batch,mask_batch,extrinsic_batch,intrinsic_batch])
        del self.VGGT_model
        torch.cuda.empty_cache()

            



            


        
    def __len__(self):
        # 返回数据集大小
        return len(self.batch_data)
    
    def __getitem__(self, idx):
        # 根据索引返回数据和标签
        # self.paired_imgs
        return self.batch_data[idx]