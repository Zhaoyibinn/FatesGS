import torch
import cv2
import numpy as np
import torch.nn as nn
from utils.loss_utils import l1_loss, ssim, TVLoss, get_depth_ranking_loss,local_pearson_loss,pearson_depth_loss
from fused_ssim import fused_ssim
import torch.nn.functional as F
from utils.point_utils import depths_to_points
from utils.feat_utils import get_feat_loss


class LossFactory():
    def __init__(self, opt,args, scene,dataset):
        self.opt = opt
        self.scene = scene
        self.args = args
        self.dataset = dataset

    def ranking_loss(self,error, penalize_ratio=0.7, extra_weights=None , type='mean'):
        error, indices = torch.sort(error)
        # only sum relatively small errors
        s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        if extra_weights is not None:
            weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
            s_error = s_error * weights

        if type == 'mean':
            return torch.mean(s_error)
        elif type == 'sum':
            return torch.sum(s_error)


    def get_loss(self,render_pkg,render_scale,viewpoint_cam,iteration):
        
        total_loss_diff = 0
        gt_image = viewpoint_cam.original_image.cuda()
        mono_depth = viewpoint_cam.mono_depth
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                       recompute_scale_factor=True, antialias=True)[0]
            mono_depth = torch.nn.functional.interpolate(mono_depth[None, None], scale_factor=1/render_scale, mode="bilinear", 
                                                       recompute_scale_factor=True, antialias=True)[0,0]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

        # regularization
        # lambda_normal = self.opt.lambda_normal if iteration > 7000 else 0.0
        # lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0
        lambda_normal = self.opt.lambda_normal
        lambda_dist = self.opt.lambda_dist
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        if self.opt.lambda_normal_est!=0:
            # est_normal_loss =(1 - ( rend_normal * torch.tensor(viewpoint_cam.normal).cuda()).sum(dim=0))[None].mean()
            est_normal_loss = (1 - F.cosine_similarity(torch.tensor(viewpoint_cam.normal).cuda(), rend_normal, dim=0))
            est_normal_loss = self.ranking_loss(est_normal_loss.flatten(), penalize_ratio=1.0, type='mean')
        else:
            est_normal_loss = 0


        
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        surf_depth = render_pkg["surf_depth"]
        

        dsmooth_loss = TVLoss(surf_depth, mono_depth.unsqueeze(0))

        if self.opt.lambda_local_pearson !=0:
            Local_pearson_loss = local_pearson_loss(mono_depth,surf_depth.squeeze(0),64,0.5)
        else:
            Local_pearson_loss = 0

        if self.opt.lambda_pearson !=0:
            pearson_loss = pearson_depth_loss(mono_depth,surf_depth.squeeze(0))
        else:
            pearson_loss = 0

        mask = (surf_depth.view(-1) > 0)

        if self.args.use_mask:
            object_mask = viewpoint_cam.gt_alpha_mask > 0.5
            mask = mask & object_mask.view(-1)
            depth_rank_loss = get_depth_ranking_loss(surf_depth, mono_depth, object_mask)
        else:
            depth_rank_loss = get_depth_ranking_loss(surf_depth, mono_depth, None)

        # Feature loss
        surf_points = depths_to_points(viewpoint_cam, surf_depth)
        src_viewpoint_stack = self.scene.getTrainCamerasSource(viewpoint_cam.image_name).copy()
        feat_loss = get_feat_loss(surf_points, viewpoint_cam, src_viewpoint_stack, mask, resolution=self.dataset.resolution)

        # loss
        total_loss = loss + dist_loss + normal_loss + self.opt.lambda_dsmooth * dsmooth_loss  + \
            self.opt.lambda_feat * feat_loss + \
            self.opt.lambda_depth * depth_rank_loss + 0.5 * total_loss_diff + self.opt.lambda_normal_est * est_normal_loss
        
        loss_dict = {
            "loss": loss,
            "Ll1": Ll1,
            "dist_loss": dist_loss,
            "normal_loss": normal_loss,
            "dsmooth_loss": dsmooth_loss,
            "local_pearson_loss": Local_pearson_loss,
            "pearson_loss": pearson_loss,
            "feat_loss": feat_loss,
            "depth_rank_loss": depth_rank_loss,
            "est_normal_loss": est_normal_loss
        }
        return total_loss,loss_dict

