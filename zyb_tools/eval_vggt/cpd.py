



from timeit import default_timer as timer
import numpy as np
import open3d as o3
# import utils
from probreg import cpd
from probreg import l2dist_regs
from probreg import gmmtree
from probreg import filterreg
import copy

threshold = 0.001
max_iteration = 10000

# source, target = utils.prepare_source_and_target_rigid_3d('/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_output_gsinit/scan40/train/ours_1/fuse_post.ply',  n_random=0,
#                                                           orientation=np.deg2rad([0.0, 0.0, 10.0]),voxel_size=0.008)

# source = o3.io.read_point_cloud('pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/align_ours.ply')



# target = o3.io.read_point_cloud('pilianghua_out/gs_init/pilianghua_output_gsinit/scan40/train/gt.ply')



# source.points = o3.utility.Vector3dVector(np.array(source.points))
# source = source.voxel_down_sample(voxel_size=20)
# # source.remove_non_finite_points()
# target = target.voxel_down_sample(voxel_size=20)
# # target.remove_non_finite_points()



# start = timer()
# res = o3.pipelines.registration.registration_icp(source, target, 0.5,
#                                                  np.identity(4), o3.pipelines.registration.TransformationEstimationPointToPoint(),
#                                                  o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
# end = timer()
# print('ICP(Open3D): ', end - start)

# start = timer()
# res = cpd.registration_cpd(source, target, maxiter=max_iteration, tol=threshold)
# end = timer()
# print('CPD: ', end - start)

# result = copy.deepcopy(source)
# result.points = res.transformation.transform(result.points)

# # draw result
# source.paint_uniform_color([1, 0, 0])
# target.paint_uniform_color([0, 1, 0])
# result.paint_uniform_color([0, 0, 1])
# # o3.visualization.draw_geometries([source, target, result])

# combined = source + target + result
# # combined = target + result
# o3.io.write_point_cloud("test.ply", combined)


# start = timer()
# res = l2dist_regs.registration_svr(source, target, opt_maxiter=max_iteration, opt_tol=threshold)
# end = timer()
# print('SVR: ', end - start)

# start = timer()
# res = gmmtree.registration_gmmtree(source, target, maxiter=max_iteration, tol=threshold)
# end = timer()
# print('GMMTree: ', end - start)

# start = timer()
# res = filterreg.registration_filterreg(source, target,
#                                        sigma2=None, maxiter=max_iteration, tol=threshold)
# end = timer()
# print('FilterReg: ', end - start)

def cpd_reg(source,target):
    source = source.voxel_down_sample(voxel_size=10)
    target = target.voxel_down_sample(voxel_size=10)
    start = timer()
    res = cpd.registration_cpd(source, target, maxiter=max_iteration, tol=threshold)
    end = timer()
    print('CPD: ', end - start)

    result = copy.deepcopy(source)
    result.points = res.transformation.transform(result.points)

    return source,target,result,res
