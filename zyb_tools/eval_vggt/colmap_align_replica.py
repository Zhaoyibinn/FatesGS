import os
import numpy as np
import torch
import roma
import open3d as o3d
import copy
import glob
from cpd import cpd_reg
from teaser import teaser_reg
import render_utils as rend_util
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from skimage.morphology import binary_dilation, disk
import kornia

# scan = 40
# vggt_pose_root = f"DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan{scan}/sparse/vggt"
# gt_pose_root =f"DTU/set_23_24_33/scan{scan}/sparse/0"
# # 用于粗对齐的Colmap相机位姿

# # input_ply = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/ours_1/fuse_post.ply"
# input_ply = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan40/sparse/vggt/points3D.ply"
# output_path = "pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/vggt_align_culled.ply"





def mask_filter(vertices, n_images, pose_all, intrinsics_all, masks):
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()
    W, H = 1600, 1200

    sampled_masks = []

    for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
        # if i not in [23,24,33]:
        #     continue
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)


    sampled_masks = torch.stack(sampled_masks, -1)
    # filter

    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    return mask
    # gt_ply_vdown_numpy_rescale_masked = gt_ply_vdown_numpy_rescale[mask]

def vis_o3d_pcd_2(cloud1,cloud2,color1 = [1,1,1],color2 = [1,1,1],down = 0):
    if down!=0:
        indices1 = np.random.choice(cloud1.shape[0], down, replace=False)
        cloud1 = cloud1[indices1]
        indices2 = np.random.choice(cloud2.shape[0], down, replace=False)
        cloud2 = cloud2[indices2]
    pcd1=o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd1.paint_uniform_color(color1)
    pcd2=o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)
    pcd2.paint_uniform_color(color2)
    pcd_combined = pcd1 + pcd2
    o3d.io.write_point_cloud("test.ply", pcd_combined)

def vis_o3d_pcd_3(cloud1, cloud2, cloud3, color1=[1, 1, 1], color2=[1, 1, 1], color3=[1, 1, 1], down=0):

    if down != 0:
        indices1 = np.random.choice(cloud1.shape[0], down, replace=False)
        cloud1 = cloud1[indices1]
        indices2 = np.random.choice(cloud2.shape[0], down, replace=False)
        cloud2 = cloud2[indices2]
        indices3 = np.random.choice(cloud3.shape[0], down, replace=False)
        cloud3 = cloud3[indices3]

    # 创建点云1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd1.paint_uniform_color(color1)

    # 创建点云2
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)
    pcd2.paint_uniform_color(color2)

    # 创建点云3
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(cloud3)
    pcd3.paint_uniform_color(color3)

    # 合并点云
    pcd_combined = pcd1 + pcd2 + pcd3
    o3d.io.write_point_cloud("test.ply", pcd_combined)


def rotate_points_with_srt(source_points, s, R, t):

    # 1. 处理缩放（先应用缩放）
    if isinstance(s, (float, int)):
        # 标量缩放：所有维度等比例缩放
        scaled_points = source_points * s
    else:
        # 各维度独立缩放（假设s是3维向量）
        scaled_points = source_points * s.reshape(1, -1)  # (N,3) * (1,3)
    
    # 2. 应用旋转变换（矩阵乘法）
    rotated_points = torch.matmul(scaled_points, R.T)  # (N,3) @ (3,3) = (N,3)
    
    # 3. 应用平移变换
    rotated_points = rotated_points + t.reshape(1, -1)  # (N,3) + (1,3)
    
    return rotated_points



def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([p[:3, 3].numpy() for p in poses]))
def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 10
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    # import open3d as o3d

    return s, R, T


def read_colmap_gt(colmap_images_path):
    # colmap_images_path = "sparse_DTU/set_23_24_33/scan40/sparse/0/images.txt"
    # colmap_images_path = "sparse_DTU/wo_pose/scan24/sparse/0/images.txt"
    with open(colmap_images_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = lines[4:]
        poses = torch.zeros(int(len(lines)/2),7)
        for idx,line in enumerate(lines):
            if idx % 2 == 0:
                line_splited = line.split()
                image_idx = int(line_splited[-1][:4])
                pose = torch.tensor([float(line_splited[2]),float(line_splited[3]),float(line_splited[4]),float(line_splited[1]),float(line_splited[5]),float(line_splited[6]),float(line_splited[7])])
                poses[image_idx] = pose

    poses_R = []
    for pose in poses:
        q_x, q_y, q_z,q_w,t_x,t_y,t_z = pose

        R = torch.eye(4)
        R_3 = torch.tensor([
            [1 - 2 * q_y ** 2 - 2 * q_z ** 2, 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)],
            [2 * (q_x * q_y + q_w * q_z), 1 - 2 * q_x ** 2 - 2 * q_z ** 2, 2 * (q_y * q_z - q_w * q_x)],
            [2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * q_x ** 2 - 2 * q_y ** 2]
            ])
        t = torch.tensor([t_x,t_y,t_z])

        R[:3, :3] = R_3
        R[:3, 3] = t

        poses_R.append(R.inverse())
    return torch.stack(poses_R,dim = 0)[:,:3,:]

def read_colmap_camera(colmap_camera_path):
    with open(colmap_camera_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line = lines[3:][0]
        image_W,image_H,focal_x,focal_y,cx,cy = [float(i) for i in line.split()[2:]]

    return image_W,image_H,focal_x,focal_y,cx,cy

colmap_gt_root = "DTU/gt"
def align1_camera_pose(scan,input_ply,focal_scale = None,extra_pose = None):
    vggt_pose_root = f"Replica/replica_vggt/{scan}/sparse/vggt"
    # vggt_pose_root = f"Replica/replica_cfgs/{scan}/sparse/0"
    # vggt_pose_root = f"Replica/replica_freegs/{scan}/sparse/0"



    gt_pose_root =f"Replica/replica_gt/{scan}/sparse/0"


    assert os.path.exists(os.path.join(vggt_pose_root,"images.txt")) or os.path.exists(os.path.join(vggt_pose_root,"images.bin")), "vggt txt和bin文件都不存在 请检查"
    assert os.path.exists(os.path.join(gt_pose_root,"images.txt")) or os.path.exists(os.path.join(gt_pose_root,"images.bin")), "gt txt和bin文件都不存在 请检查"

    assert os.path.exists(os.path.join(vggt_pose_root,"points3D.ply")), "vggt points3D.ply文件不存在 请检查"
    assert os.path.exists(os.path.join(gt_pose_root,"points3D.ply")) or os.path.exists(os.path.join(gt_pose_root,"points3D_colmap.ply")), "colmap points3D.ply文件不存在 请检查"

    vggt_image_txt_path = os.path.join(vggt_pose_root,"images.txt")
    gt_image_txt_path = os.path.join(gt_pose_root,"images.txt")
    vggt_ply_path = os.path.join(vggt_pose_root,"points3D.ply")
    vggt_ply_path = input_ply
    if not os.path.exists(os.path.join(gt_pose_root,"points3D.ply")):
        colmap_ply_path = os.path.join(gt_pose_root,"points3D_colmap.ply")
    else:
        colmap_ply_path = os.path.join(gt_pose_root,"points3D.ply")

    gt_intrinsic = os.path.join(gt_pose_root,"cameras.txt")

    vggt_ply = o3d.io.read_point_cloud(vggt_ply_path)
    colmap_ply = o3d.io.read_point_cloud(colmap_ply_path)


    if not os.path.exists(vggt_image_txt_path):
        os.system(f"colmap model_converter --input_path {vggt_pose_root} --output_path {vggt_pose_root} --output_type TXT")
        print("vggt colmap bin转txt完成")
    if not os.path.exists(gt_image_txt_path):
        os.system(f"colmap model_converter --input_path {gt_pose_root} --output_path {gt_pose_root} --output_type TXT")
        print("gt colmap bin转txt完成")




    image_W,image_H,focal_x,focal_y,cx,cy = read_colmap_camera(gt_intrinsic)

    vggt_pose = read_colmap_gt(vggt_image_txt_path)
    # if focal_scale != None:
    #     vggt_pose[:,:,3] = vggt_pose[:,:,3] * focal_scale
    gt_pose = read_colmap_gt(gt_image_txt_path)

    last_row = torch.tensor([0, 0, 0, 1]).expand(gt_pose.shape[0], 1, 4)
    gt_pose44 = torch.cat([gt_pose, last_row], dim=1)
    vggt_pose = torch.cat([vggt_pose, last_row], dim=1)
    vggt_pose_origin = copy.deepcopy(vggt_pose)

    if extra_pose is not None:
        # extra_pose[:3, :4] 是 wxyz，转换为 xyzw
        q_wxyz = extra_pose[:3, :4]
        q_xyzw = torch.cat([q_wxyz[:, 1:], q_wxyz[:, :1]], dim=1)
        q_normal = torch.nn.functional.normalize(q_xyzw, dim=1)
        rot = kornia.geometry.quaternion_to_rotation_matrix(q_normal)
        t = extra_pose[:3, 4:]
        for i in range(vggt_pose.shape[0]):
            extra_pose_4x4 = torch.eye(4)
            extra_pose_4x4[:3, :3] = rot[i]
            extra_pose_4x4[:3, 3] = t[i].squeeze()
            vggt_pose[i] = torch.matmul(extra_pose_4x4.T, vggt_pose[i])

        # vggt_pose = torch.matmul(extra_pose[None,...],vggt_pose)

    s_2colmap, R_2colmap, T_2colmap = align_multiple_poses(vggt_pose,gt_pose44)
    SRT = torch.eye(4)
    SRT[:3, :3] = s_2colmap * R_2colmap
    SRT[:3, 3] = T_2colmap
    poses_new = torch.matmul(SRT[None, ...], vggt_pose)  # [N, 4, 4]

    return s_2colmap,R_2colmap,T_2colmap,vggt_ply,colmap_ply,poses_new,gt_pose44

def align1_rescale(scan,vggt_ply_numpy_tran2colmap):
    # vggt_ply_numpy_tran2colmap = rotate_points_with_srt(torch.tensor(vggt_ply_numpy).float(),s,R,T).numpy()
    # vis_o3d_pcd_2(colmap_ply_numpy,points_trans,color1=[1,0,0],color2=[0,1,0])
    gt_ply = o3d.io.read_point_cloud(os.path.join(colmap_gt_root,"Points","stl",f"stl{scan:03d}_total.ply"))
    gt_ply_vdown = gt_ply.voxel_down_sample(voxel_size=10)

    gt_ply_numpy = np.array(gt_ply.points)
    gt_ply_vdown_numpy = np.array(gt_ply_vdown.points)

    instance_dir = os.path.join(colmap_gt_root, f'scan{scan}')
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]


    scale_mat = scale_mats[0]
    # colmap_ply_numpy_tran2gt = colmap_ply_numpy * scale_mat[0, 0] + scale_mat[:3, 3][None]
    vggt_plt_numpy_tran2gt = vggt_ply_numpy_tran2colmap * scale_mat[0, 0] + scale_mat[:3, 3][None]

    return vggt_plt_numpy_tran2gt, scale_mat[0, 0], scale_mat[:3, 3][None],gt_ply

    # vis_o3d_pcd_2(vggt_plt_numpy_tran2gt,gt_ply_vdown_numpy,color1=[1,0,0],color2=[0,1,0])

def align1_reg(source,target):
    source_ply = o3d.geometry.PointCloud()
    source_ply.points = o3d.utility.Vector3dVector(source)

    targer_ply = o3d.geometry.PointCloud()
    targer_ply.points = o3d.utility.Vector3dVector(target)

    # source,target,result,res = cpd_reg(source_ply,targer_ply)

    source,target,result,T = teaser_reg(source_ply,targer_ply,VOXEL_SIZE=0.07)

    return T

def mask_dtu(scan,vggt_plt_numpy_tran2gt_align):
    # vggt_plt_numpy_tran2gt_align = (T[:3,:3] @ copy.deepcopy(vggt_plt_numpy_tran2gt).T).T + T[:3,3]
    # # vis_o3d_pcd_2(gt_ply_vdown_numpy,vggt_plt_numpy_tran2gt_align,color1=[1,0,0],color2=[0,1,0])


    # # gt_ply_vdown_numpy_rescale = (gt_ply_vdown_numpy - scale_mat[:3, 3][None]) / scale_mat[0, 0]
    


    intrinsics_all = []
    pose_all = []
    instance_dir = os.path.join(colmap_gt_root, f'scan{scan}')
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mat = scale_mats[0]

    vggt_plt_numpy_tran2gt_align_rescale = (vggt_plt_numpy_tran2gt_align - scale_mat[:3, 3][None]) / scale_mat[0, 0]

    for scale_mat, world_mat in zip(scale_mats, world_mats):
        
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)


    # project and filter

    # vertices = torch.from_numpy(gt_ply_vdown_numpy_rescale).cuda()
    # mask = mask_filter(vertices, n_images, pose_all, intrinsics_all, masks)
    # gt_ply_vdown_numpy_rescale_masked = gt_ply_vdown_numpy_rescale[mask]
    vertices = torch.from_numpy(vggt_plt_numpy_tran2gt_align_rescale).cuda()
    mask = mask_filter(vertices, n_images, pose_all, intrinsics_all, masks)
    vggt_plt_numpy_tran2gt_align_rescale_masked = vggt_plt_numpy_tran2gt_align_rescale[mask]
    # vggt_plt_numpy_tran2gt_align_rescale_masked = vggt_plt_numpy_tran2gt_align_rescale

    vggt_plt_numpy_tran2gt_align_masked = vggt_plt_numpy_tran2gt_align_rescale_masked * scale_mat[0, 0] + scale_mat[:3, 3][None]

    return vggt_plt_numpy_tran2gt_align_masked


    # vis_o3d_pcd_2(vggt_plt_numpy_tran2gt_align_rescale_masked,gt_ply_numpy,color1=[1,0,0],color2=[0,1,0])

    # o3d.io.write_point_cloud(output_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vggt_plt_numpy_tran2gt_align_masked)))


print("end")
    


# colmap model_converter --input_path sparse/origin --output_path sparse/origin --output_type TXT