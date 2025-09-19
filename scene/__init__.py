#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from utils.mvs_depth_consistency_utils import *
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams,OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# real_idx = []
real_idx = [0,24,48]

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        self.args = args
        


        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.train_cameras_diff = {}
        self.test_cameras = {}

        if not os.path.exists(os.path.join(args.source_path, "depth_all")):
            os.system(f"python submodules/Marigold/run.py --input_rgb_dir {os.path.join(args.source_path, 'images')} --output_dir {os.path.join(args.source_path, 'depth_all')} --scale {args.resolution}")
            os.system(f"cp -r {os.path.join(args.source_path, 'depth_all','depth_npy')} {os.path.join(args.source_path,'depth_npy')} ")
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args)
            print("默认采用colmap格式")




        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.train_cameras_diff)  # Multi-res consistent random shuffling

        self.mvs_filter_masks = []
        if self.args.mvs_filter:
            print("进行基于MVS多视角一致性检查")
            self.mvs_filter(scene_info.train_cameras)
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args,mvs_filter_masks = self.mvs_filter_masks)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args,mvs_filter_masks = self.mvs_filter_masks)
            if scene_info.train_cameras_diff:
                print("Loading Diffusion Training Cameras")
                self.train_cameras_diff[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras_diff, resolution_scale, args,mvs_filter_masks = self.mvs_filter_masks)

        train_max = len(self.train_cameras[1])
        test_max = len(self.test_cameras[1])
        cameras_idx_max = max(train_max,test_max)
        self.gaussians.cameras_idx_max = cameras_idx_max


        if args.init == 'vggt_gs':
            self.gaussians.load_ply(os.path.join(args.source_path,"sparse","0","points3D_GS.ply"))
        elif self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            extra_trans_pth = os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"extra_trans.pth")
            if os.path.isfile(extra_trans_pth):
                self.gaussians.init_extra_pose()
                self.gaussians.load_extra_pose(extra_trans_pth)
                print(f"找到优化的pose")
            else:
                self.gaussians.init_extra_pose()
                print(f"没有找到优化的pose 单位初始化")
        else:
            
            if scene_info.ply_path.rsplit('/',1)[1].split('.')[0].endswith("mvsgs"):
                self.gaussians.load_ply(scene_info.ply_path)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def mvs_filter(self,train_cameras):
        
        all_ref_idxs = []
        for ii in range(len(train_cameras)):
            range_num = 2
            ref_tart = max(0, ii - range_num)
            ref_end = min(len(train_cameras)-1, ii + range_num)
            ref_idxs = list(range(ref_tart, ref_end + 1))
            ref_idxs.remove(ii)
            all_ref_idxs.append(ref_idxs)
            dy_range = len(ref_idxs)

        for ii in range(len(train_cameras)):
            current_camera = train_cameras[ii]
            current_ext_mat = self.cameras_trans_for_mvs(current_camera)
            current_in_mat = current_camera.K
            current_in_mat[0][0] = current_in_mat[1][1] = 951.4357
            current_in_mat[0][2] = 256.0
            current_in_mat[1][2] = 144.0
            current_depth = current_camera.dust3r_depth
            ref_idxs = all_ref_idxs[ii]

            geo_mask_sum = 0
            
            for ref_idx in ref_idxs:
                ref_camera = train_cameras[ref_idx]
                ref_ext_mat = self.cameras_trans_for_mvs(ref_camera)
                ref_in_mat = ref_camera.K
                ref_in_mat[0][0] = ref_in_mat[1][1] = 951.4357
                ref_in_mat[0][2] = 256.0
                ref_in_mat[1][2] = 144.0
                ref_depth = ref_camera.dust3r_depth

                masks, masks_per,depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(current_depth,current_in_mat,current_ext_mat,ref_depth,ref_in_mat,ref_ext_mat)
                
                geo_mask_sum += masks[140].astype(np.int32)
            geo_mask = geo_mask_sum >= dy_range * 0.9
            self.mvs_filter_masks.append(geo_mask)
            print(f"相机{ii} 过滤后还剩{np.mean(geo_mask)}的点")
            print("ok")

        return 0
        

    def cameras_trans_for_mvs(self,camera):
        current_R , current_t = camera.R,camera.T
        current_ext_mat = np.eye(4)
        current_ext_mat[:3,:3] = current_R.T
        current_ext_mat[:3,3] = current_t

        return current_ext_mat # C2W
        # return np.linalg.inv(current_ext_mat) #W2C

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_extra_trans(os.path.join(point_cloud_path, "extra_trans.pth"))
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    
    def getTrainCameras_diff(self, scale=1.0):
        return self.train_cameras_diff[scale]

    def getTrainCamerasByIdx(self, idx, scale=1.0):
        cameras = self.train_cameras[scale]
        return [cameras[i] for i in idx]

    def getTrainCamerasSource(self, cam_img_name, scale=1.0):
        cameras = self.train_cameras[scale]
        if self.args.diff:
            ref_camera = cameras[real_idx.index(int(cam_img_name))]
            return [camera for camera in cameras if int(camera.image_name) in ref_camera.pair]
        else:
            ref_camera = cameras[int(cam_img_name)]
            return [camera for camera in cameras if int(camera.image_name) in ref_camera.pair]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]