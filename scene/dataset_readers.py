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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
from utils.feat_utils import FeatExt, load_pair
from utils.general_utils import PILtoTorch


# real_idx = []
real_idx = [0,24,48]

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    feat: list = None
    pair: list = None
    mono_depth: np.array = None
    mask: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    train_cameras_diff: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(path, cam_extrinsics, cam_intrinsics, images_folder, read_mask, args):
    cam_infos = []


    
    depths_folder = os.path.join(path, "depth_npy")
    masks_folder = os.path.join(path, "mask")

    from glob import glob
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs

    image_paths_all = sorted(glob_imgs(images_folder))

    if not args.diff:
        image_paths = image_paths_all
    else:
        image_paths = []
        for idx,image_path in enumerate(image_paths_all):
            if idx in real_idx:
                image_paths.append(image_path)
            

    

    n_images = len(image_paths)

    feats_list = []
    scale_list = [1, 2]
    if args.resolution == 2:
        ori_w, ori_h = 1600, 1200
    else:
        ori_w, ori_h = 768, 576
    for scale in scale_list:
        feat_ext = FeatExt().cuda()
        feat_ext.eval()
        for p in feat_ext.parameters():
            p.requires_grad = False

        size_w = ori_w // scale
        size_h = ori_h // scale

        rgb_2xd = torch.zeros(n_images, 3, size_h, size_w)

        for i in range(n_images):
            image_pil = Image.open(image_paths[i])
            resolution = (int(image_pil.size[0] / scale), int(image_pil.size[1] / scale))
            image = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
            rgb_2xd[i, :, :image.shape[1], :image.shape[2]] = image
            # 这里似乎限制了图片大小,只放进去但是不一定塞满
        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        

        feats = []
        feat_eval_bs = 20
        for start_i in range(0, n_images, feat_eval_bs):
            eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
            feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu() # 通道数32 size减半
            feats.append(feat2)
        feats = torch.cat(feats, dim=0)
        feats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
        feats_list.append(feats)
        # 这里读取图片之后用一个(似乎参考了MVSNET和UNET)特征提取网络

    pairs = load_pair(os.path.join(images_folder, "..", "pair.txt"))

    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if args.diff:
            if int(image_name) not in real_idx:
                continue

        image = Image.open(image_path)

        depth_path = os.path.join(depths_folder, image_name + "_pred.npy")
        depth = None
        # 这边除了常规的还增加了深度图

        if os.path.exists(depth_path):
            depth = np.load(depth_path)

        mask = None
        if read_mask:
            mask_path = os.path.join(masks_folder, "{:0>3}.png".format(int(image_name)))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)



        pair = pairs[str(int(image_name))]['pair'][:2]
        


        if args.diff:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              feat=[feats[real_idx.index(int(image_name))] for feats in feats_list],
                              pair=[int(idx) for idx in pair], mono_depth=depth, mask=mask)
        else:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                feat=[feats[int(image_name)] for feats in feats_list],
                                pair=[int(idx) for idx in pair], mono_depth=depth, mask=mask)
        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       feat=[feats[int(image_name)] for feats in feats_list], mono_depth=depth, mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras_diff(path, cam_extrinsics, cam_intrinsics, images_folder, read_mask, args):
    cam_infos = []




    
    depths_folder = os.path.join(path, "depth_npy")
    masks_folder = os.path.join(path, "mask")

    from glob import glob
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs

    image_paths_all = sorted(glob_imgs(images_folder))

    all_img_num = len(image_paths_all)
    all_img_idx = list(range(all_img_num))
    diff_img_idx = [value for value in all_img_idx if value not in real_idx]

    image_paths = []
    for idx,image_path in enumerate(image_paths_all):
        if idx not in real_idx:
            image_paths.append(image_path)
            

    

    n_images = len(image_paths)

    feats_list = []
    scale_list = [1, 2]
    if args.resolution == 2:
        ori_w, ori_h = 1600, 1200
    else:
        ori_w, ori_h = 768, 576
    for scale in scale_list:
        feat_ext = FeatExt().cuda()
        feat_ext.eval()
        for p in feat_ext.parameters():
            p.requires_grad = False

        size_w = ori_w // scale
        size_h = ori_h // scale

        rgb_2xd = torch.zeros(n_images, 3, size_h, size_w)

        for i in range(n_images):
            image_pil = Image.open(image_paths[i])
            resolution = (int(image_pil.size[0] / scale), int(image_pil.size[1] / scale))
            image = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
            rgb_2xd[i, :, :image.shape[1], :image.shape[2]] = image
            # 这里似乎限制了图片大小,只放进去但是不一定塞满
        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        

        feats = []
        feat_eval_bs = 20
        for start_i in range(0, n_images, feat_eval_bs):
            eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
            feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu() # 通道数32 size减半
            feats.append(feat2)
        feats = torch.cat(feats, dim=0)
        feats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
        feats_list.append(feats)
        # 这里读取图片之后用一个(似乎参考了MVSNET和UNET)特征提取网络

    pairs = load_pair(os.path.join(images_folder, "..", "pair.txt"))

    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if int(image_name) in real_idx:
            continue

        image = Image.open(image_path)

        depth_path = os.path.join(depths_folder, image_name + "_pred.npy")
        depth = None
        # 这边除了常规的还增加了深度图

        if os.path.exists(depth_path):
            depth = np.load(depth_path)

        mask = None
        if read_mask:
            mask_path = os.path.join(masks_folder, "{:0>3}.png".format(int(image_name)))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)

        pair = pairs[str(int(image_name))]['pair'][:2]
        

        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height,
                            feat=[feats[diff_img_idx.index(int(image_name))] for feats in feats_list],
                            pair=[int(idx) for idx in pair], mono_depth=depth, mask=mask)

        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       feat=[feats[int(image_name)] for feats in feats_list], mono_depth=depth, mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort

def readColmapSceneInfo(path, images, eval, args, llffhold=8, n_views=3):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    read_mask = True if os.path.exists(os.path.join(path, "mask")) else False
    cam_infos_unsorted = readColmapCameras(path=path, cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           read_mask=read_mask, args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if args.diff:
        cam_infos_unsorted_diff = readColmapCameras_diff(path=path, cam_extrinsics=cam_extrinsics,
                                            cam_intrinsics=cam_intrinsics,
                                            images_folder=os.path.join(path, reading_dir),
                                            read_mask=read_mask, args=args)
        cam_infos_diff = sorted(cam_infos_unsorted_diff.copy(), key = lambda x : x.image_name)


    

    if eval:
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
        train_idx = train_idx[:n_views]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        if args.diff:
            train_cam_infos_diff = cam_infos_diff

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if args.diff:
        nerf_normalization_diff = getNerfppNorm(train_cam_infos_diff)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if eval:
        ply_path = os.path.join(path, "pixelnerf/dense/fused.ply")
    else:
        if args.origin_data:
            ply_path = os.path.join(path, "sparse/0/points3D_colmap.ply")
            print("直接采用colmap的结果而非其他初始化")
        else:
            ply_path = os.path.join(path, "sparse/0/points3D.ply")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if args.diff:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            train_cameras_diff=train_cam_infos_diff,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    else:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            train_cameras_diff=[],
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}