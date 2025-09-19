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
import swanlab
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, TVLoss, get_depth_ranking_loss,local_pearson_loss,pearson_depth_loss
from utils.feat_utils import get_feat_loss
from utils.point_utils import depth_to_normal,depth_to_normal_dust3r
from utils.schedule_utils import TrainingScheduler
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import depths_to_points
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from fused_ssim import fused_ssim

from extra_model.lowpass_pt import create_lowpass_filter,apply_lowpass_filter
import cv2
import numpy as np

from utils.loss_factory import LossFactory
# import matplotlib
# matplotlib.use('TkAgg')


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
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
    
def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.1):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans = render(cam, gaussians, pipe, bg, record_transmittance=True)
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any(): #如果有任何一个比之前大的贡献
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]
            # 第一轮复制五个一样的trans
    # 其实就是记录了每个点的在不同相机中的最大贡献度，并且从大到小排（属于哪个相机不重要）

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()

def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_z / 2
    z_max = scene_center[2] + span_z / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    train_config = {
        "lambda_diff_l1": opt.lambda_diff_l1,
        "lambda_diff_ssim": opt.lambda_diff_ssim,
        "lambda_diff_rend_dist": opt.lambda_diff_rend_dist,
        "lambda_diff_normal": opt.lambda_diff_normal,
        "lambda_diff_dsmooth": opt.lambda_diff_dsmooth,
        "lambda_diff_depth": opt.lambda_diff_depth,
        "lambda_local_pearson": opt.lambda_local_pearson,
        # "origin_fatesgs":opt.origin_train,
    }

    config_save_path = os.path.join(dataset.model_path,"train_conf.json")

    # if not opt.not_record:
    #     swanlab.init(
    #     # 设置项目名
    #     project="Diff_FatesGS_train",
    #     # 设置超参数
    #     config=train_config,
    #     experiment_name=dataset.source_path.split("/")[-1]
    #     )



    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # dataset.origin_train = opt.origin_train
    dataset.init = opt.init
    dataset.mvs_filter = opt.mvs_filter
    # dataset.vggt = opt.vggt
    # dataset.normals_est = opt.normals_est
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_diff = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    print(f"检测到原本训练视角 {len(scene.getTrainCameras())}")
    if args.diff:
        print(f"检测到扩散模型训练视角 {len(scene.getTrainCameras_diff())}")
    with open(config_save_path, 'w') as f:
        json.dump(train_config, f)
    all_cameras = scene.getTrainCameras().copy()


    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, 
                                  [cam.original_image for cam in scene.getTrainCameras()],max_scale = 3)
    # max scale就是dash开始时候的下采样率 注意会取int 所以本质上是-1的
    render_scale = scheduler.get_res_scale(1)


    loss_factory = LossFactory(opt, args,scene,dataset)

    for iteration in range(first_iter, opt.iterations + 1):
        # print("render scale", render_scale)
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if not viewpoint_stack_diff:
            if args.diff:
                viewpoint_stack_diff = scene.getTrainCameras_diff().copy()
                
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if args.diff:
        # if False:
            viewpoint_cam_diff = viewpoint_stack_diff.pop(randint(0, len(viewpoint_stack_diff)-1))
            render_pkg_diff = render(viewpoint_cam_diff, gaussians, pipe, background)
            image_diff, viewspace_point_tensor_diff, visibility_filter_diff, radii_diff = render_pkg_diff["render"], \
                render_pkg_diff["viewspace_points"], render_pkg_diff["visibility_filter"], render_pkg_diff["radii"]
            gt_image_diff = viewpoint_cam_diff.original_image.cuda()
            

            filter_tensor = create_lowpass_filter((image_diff.shape[1], image_diff.shape[2]), cutoff=0.6, device="cuda")
            
            low_pass_image_diff = apply_lowpass_filter(image_diff.unsqueeze(0), filter_tensor).squeeze()
            low_pass_gt_image_diff = apply_lowpass_filter(gt_image_diff.unsqueeze(0), filter_tensor).squeeze()
            
            # Ll1_diff = l1_loss(image_diff, gt_image_diff)
            # SSIM_diff = 1.0 - ssim(image_diff, gt_image_diff)
            Ll1_diff = l1_loss(low_pass_image_diff, low_pass_gt_image_diff)
            SSIM_diff = 1.0 - fused_ssim(low_pass_image_diff.unsqueeze(0), low_pass_gt_image_diff.unsqueeze(0))

            # loss_diff = opt.lambda_dssim * SSIM_diff + (1.0 - opt.lambda_dssim) * Ll1_diff
            loss_diff = opt.lambda_dssim * SSIM_diff 

            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist_diff = render_pkg_diff["rend_dist"]
            rend_normal_diff  = render_pkg_diff['rend_normal']
            surf_normal_diff = render_pkg_diff['surf_normal']


            normal_error_diff = (1 - (rend_normal_diff * surf_normal_diff).sum(dim=0))[None]
            normal_loss_diff = lambda_normal * (normal_error_diff).mean()
            dist_loss_diff = lambda_dist * (rend_dist_diff).mean()

            surf_depth_diff = render_pkg_diff["surf_depth"]
            mono_depth_diff = viewpoint_cam_diff.mono_depth

            dsmooth_loss_diff = TVLoss(surf_depth_diff, mono_depth_diff.unsqueeze(0))
            
            mask_diff = (surf_depth_diff.view(-1) > 0)

            if args.use_mask:
                object_mask_diff = viewpoint_cam_diff.gt_alpha_mask > 0.5
                mask_diff = mask_diff & object_mask_diff.view(-1)
                depth_rank_loss_diff = get_depth_ranking_loss(surf_depth_diff, mono_depth_diff, object_mask_diff)
            else:
                depth_rank_loss_diff = get_depth_ranking_loss(surf_depth_diff, mono_depth_diff, None)

            # Feature loss
            # surf_points = depths_to_points(viewpoint_cam_diff, surf_depth_diff)
            # src_viewpoint_stack = scene.getTrainCamerasSource(viewpoint_cam.image_name).copy()
            # feat_loss = get_feat_loss(surf_points, viewpoint_cam, src_viewpoint_stack, mask, resolution=dataset.resolution)
        
            # total_loss_diff = loss_diff + dist_loss_diff + normal_loss_diff + dsmooth_loss_diff + opt.lambda_depth * depth_rank_loss_diff
            total_loss_diff = opt.lambda_diff_l1*Ll1_diff+opt.lambda_diff_ssim *SSIM_diff + opt.lambda_diff_rend_dist * dist_loss_diff + opt.lambda_diff_normal * normal_loss_diff + opt.lambda_diff_dsmooth * dsmooth_loss_diff + opt.lambda_diff_depth * depth_rank_loss_diff
            # print(opt.lambda_diff_l1,opt.lambda_diff_ssim)


        gt_image = viewpoint_cam.original_image.cuda()
        
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                       recompute_scale_factor=True, antialias=True)[0]
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,drop=opt.drop,iteration=iteration, render_size=gt_image.shape[-2:])


        total_loss,loss_dict = loss_factory.get_loss(render_pkg,render_scale,viewpoint_cam,iteration)
        
        
        total_loss.backward()

        iter_end.record()

        loss = loss_dict["loss"]
        dist_loss = loss_dict["dist_loss"]
        normal_loss = loss_dict["normal_loss"]
        Ll1 = loss_dict["Ll1"]
        est_normal_loss = loss_dict["est_normal_loss"]
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                # loss_dict = {
                #     "Loss": f"{ema_loss_for_log:.{5}f}",
                #     "distort": f"{ema_dist_for_log:.{5}f}",
                #     "normal": f"{ema_normal_for_log:.{5}f}",
                #     "Points": f"{len(gaussians.get_xyz)}"
                # }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/est_normal_loss', est_normal_loss, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # if iteration%100 == 0 and not opt.not_record:
            #     swanlab.log({"L1_loss": Ll1})
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            viewspace_point_tensor, visibility_filter, radii =  render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # if iteration % opt.densification_interval == 0:
                    # print("正在debug 没有开始densify的iter")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if opt.split == "ordinary":
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold,opt.absgs)
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    elif opt.split == "scale":
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        gaussians.densify_and_scale_split(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, opt.max_screen_size, opt.densify_scale_factor, scene_mask, N=3, no_grad=True)
                    elif opt.split == "mix":
                        # grads = gaussians.xyz_gradient_accum / gaussians.denom
                        # grads[grads.isnan()] = 0.0
                        # grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
                        # grads_abs[grads_abs.isnan()] = 0.0
                        # gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene.cameras_extent)

                        # Apply DashGaussian primitive scheduler to control densification.
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        # 这个主要是用来确定可视区域的 但是对于DTU来说z轴的可视太奇怪了 所以暂时在后面是弃用的
                        densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                        momentum_add = gaussians.densify_and_mix_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold,opt,scene,scene_mask,opt.absgs,densify_rate = densify_rate)
                        scheduler.update_momentum(momentum_add)
                        render_scale = scheduler.get_res_scale(iteration)
                        print("render scale", render_scale)
                        

                    
                    # # gaussians.prune_large_and_transparent(0.005, 10.0)

                # TrimGS
                if opt.trim and iteration == 1:
                    origin_num = len(gaussians.get_xyz)
                    prune_low_contribution_gaussians(gaussians, all_cameras, pipe, background, K=5, prune_ratio=opt.contribution_prune_ratio)
                    pruned_num = len(gaussians.get_xyz)
                    print(f'修建到{round(pruned_num/origin_num * 100, 2)}%的点 从{origin_num}到{pruned_num}')
                    print(f'在初始帧修剪')
                if opt.trim and iteration > opt.contribution_prune_from_iter and iteration % opt.contribution_prune_interval == 0:
                    origin_num = len(gaussians.get_xyz)
                    prune_low_contribution_gaussians(gaussians, all_cameras, pipe, background, K=5, prune_ratio=opt.contribution_prune_ratio)
                    pruned_num = len(gaussians.get_xyz)
                    print(f'修建到{round(pruned_num/origin_num * 100, 2)}%的点 从{origin_num}到{pruned_num}')
                    

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # if viewpoint.gt_alpha_mask is not None:
                    #     object_mask = (viewpoint.gt_alpha_mask > 0.5).view(-1)
                    #     image = image.view(3, -1)[:, object_mask]
                    #     gt_image = gt_image.view(3, -1)[:, object_mask]
                    if viewpoint.gt_alpha_mask is not None:
                        object_mask = (viewpoint.gt_alpha_mask > 0.5)
                        object_mask = object_mask.expand_as(image) 
                        image = image * object_mask
                        gt_image = gt_image * object_mask

                    gt_alpha_mask = viewpoint.gt_alpha_mask

                    masked_image = image * gt_alpha_mask
                    masked_gt_image = gt_image * gt_alpha_mask

                    # l1_test += l1_loss(image, gt_image).mean().double()
                    # psnr_test += psnr(image, gt_image).mean().double()
                    # ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    l1_test += l1_loss(masked_image, masked_gt_image).mean().double()
                    psnr_test += psnr(masked_image, masked_gt_image).mean().double()
                    ssim_test += fused_ssim(masked_image.unsqueeze(0), masked_gt_image.unsqueeze(0)).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                save_path = os.path.join(scene.model_path,"iter_test_result.json")
                if iteration == 1:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                        print(f"已删除文件: {save_path}")
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f, ensure_ascii=False, indent=2)

                with open(save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data[iteration] = {}
                data[iteration][f"psnr"] = psnr_test.item()
                data[iteration][f"ssim"] = ssim_test.item()
                data[iteration][f"l1"] = l1_test.item()

                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,500,1000,3000,5000,10000, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,500,1000,3_000,5000,10000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--diff", type=bool, default = False)
    # 是否采用扩散模型

    args = parser.parse_args(sys.argv[1:])
    # lp.use_diff = args.diff
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")