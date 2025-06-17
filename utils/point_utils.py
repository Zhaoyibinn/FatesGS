import torch
import numpy as np
import open3d as o3d
import cv2
def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrinsic[:3, :3]
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)


    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depth_to_normal_dust3r(view, depth,conf):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)

    u, v = np.meshgrid(np.arange(depth.shape[2]), np.arange(depth.shape[1]))
    u = u.flatten()
    v = v.flatten()

    depth_flatten = depth.flatten().cpu().detach()
    
    valid_mask = torch.logical_and(depth_flatten > 0,torch.tensor(conf.flatten()) > 5 )
    u_filtered = u[valid_mask]
    v_filtered = v[valid_mask]

    points_filtered = points.reshape(-1, 3)[valid_mask]
    points_filtered_numpy = points_filtered.cpu().detach().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered_numpy)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1,max_nn=30)
        )
    
    normal_map = np.zeros((depth.shape[1], depth.shape[2], 3), dtype=np.float32)
    normal_map[v_filtered, u_filtered] = np.asarray(pcd.normals)
    # normal_map_vis = ((normal_map + 1) / 2 * 255).astype(np.uint8)
    # dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    # dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    # output[1:-1, 1:-1, :] = normal_map
    return torch.tensor(normal_map)
