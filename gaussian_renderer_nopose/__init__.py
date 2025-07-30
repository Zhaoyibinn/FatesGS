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

import torch
from torch import nn
import numpy as np
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh,RGB2SH
from utils.point_utils import depth_to_normal
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal,getProjectionMatrix
from simple_knn._C import distCUDA2
# from fun import *



def inverse_opacity_activation(x):
    return torch.log(x/(1-x))

    # return xyz,features_dc,features_rest,scaling,rotation,opacity,max_radii2D

def render(color0,pc ,color,tcnn_pred,pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,drop=False,iteration=None,record_transmittance=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # pc = pc[:100000]
    if True:
        sh_degree = 0
        fused_point_cloud = torch.tensor(np.asarray(pc.cpu())).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(color.cpu())).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pc.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")


        xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        scaling = nn.Parameter(scales.requires_grad_(True))
        rotation = nn.Parameter(rots.requires_grad_(True))
        opacity = nn.Parameter(opacities.requires_grad_(True))
        max_radii2D = torch.zeros((pc.shape[0]), device="cuda")








    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.shape[0], 4), dtype=pc.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    img_height = color0.shape[-2]
    img_width = color0.shape[-1]

    # Set up rasterization configuration

    beishu = 777*2 / img_width
    focal = 2892.33/beishu
    # focal = 2892.33
    # focal = align_model.im_focals[0].item()
    FoVy = focal2fov(focal, img_height)
    FoVx = focal2fov(focal, img_width)

    cam_K= torch.tensor([[focal,0.0,img_width/2],[0.0,focal,img_height/2],[0.0,0.0,1.0]]).cuda()

    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    viewmatrix_zero =torch.tensor([  [ 1.000,  0.000,  0.000,  0.000],
                                     [ 0.000,  1.000,  0.000,  0.000],
                                     [ 0.000,  0.000,  1.000,  0.000],
                                     [ 0.000,  0.000,  0.000,  1.000]]).cuda()
    camera_center_zero =torch.tensor([0.00,0.00,0.00]).cuda()
    zfar = 100.0
    znear = 0.01
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()

    full_proj_transform_zero = (viewmatrix_zero.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(img_height),
            image_width=int(img_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix_zero,
            projmatrix=full_proj_transform_zero,
            sh_degree=0,
            campos=camera_center_zero,
            prefiltered=False,
            record_transmittance=record_transmittance,
            debug=False,
            # pipe.debug
        )
        trimgs = True
    except:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(img_height),
            image_width=int(img_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix_zero,
            projmatrix=full_proj_transform_zero,
            sh_degree=0,
            campos=camera_center_zero,
            prefiltered=False,
            debug=False,
            # pipe.debug
        )
        trimgs = False

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc
    means2D = screenspace_points

    opacities = tcnn_pred[:,0].unsqueeze(-1) - 2
    scaling = tcnn_pred[:,1:3]-10
    rotation = tcnn_pred[:,3:7]

    opacity = torch.sigmoid(opacities)
    scales = torch.exp(scaling)
    rotations = torch.nn.functional.normalize(rotation)
    
    cov3D_precomp = None
    pipe.convert_SHs_python = False


    features_dc = torch.reshape(tcnn_pred[:,7:10],(tcnn_pred[:,7:10].shape[0],-1,3))
    features_rest = torch.reshape(tcnn_pred[:,10:],(tcnn_pred[:,10:].shape[0],-1,3))
    shs = torch.cat((features_dc, features_rest), dim=1)

    # colors_precomp = tcnn_pred[:,1:4]
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    try:
        means3D.retain_grad()
    except:
        pass

    # if drop:
    #     # Create initial compensation factor (1 for each Gaussian)
    #     compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

    #     # Apply DropGaussian with compensation
    #     drop_rate = 0.2 * (iteration/7500)
    #     # print(f"dropping{drop_rate}")
    #     d = torch.nn.Dropout(p=drop_rate)
    #     compensation = d(compensation)

    #     # Apply to opacity
    #     opacity = opacity * compensation[:, None]

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    # if trimgs:
    #     if record_transmittance:
    #         transmittance_sum, num_covered_pixels, radii = rendered_image, radii, allmap
    #         transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
    #         return transmittance
    #     else:
    #         rendered_image, radii, allmap = rendered_image, radii, allmap

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = surf_normal.permute(2,0,1)
    # # remember to multiply with accum_alpha since render_normal is unnormalized.
    # surf_normal = surf_normal * (render_alpha).detach()
    surf_normal = None

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets