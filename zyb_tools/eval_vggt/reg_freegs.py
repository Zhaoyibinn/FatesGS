
import os
import open3d as o3d
from teaser import teaser_reg
import numpy as np
import copy
from colmap_align import *

target_path = "DTU/gt/Points/stl/stl024_total.ply"
source_path = "/home/zhaoyibin/3DRE/3DGS/FreeSplatter/outputs/scan24/mesh.obj"
source_path_root = os.path.dirname(source_path)

source_mesh = o3d.io.read_triangle_mesh(source_path)
source_ply = source_mesh.sample_points_uniformly(number_of_points=100000)
target_ply = o3d.io.read_point_cloud(target_path)

source_ply_scale = copy.deepcopy(source_ply)
source_ply_scale.points = o3d.utility.Vector3dVector(np.array(source_ply.points)*200)

source_center = np.mean(np.asarray(source_ply_scale.points), axis=0)
target_center = np.mean(np.asarray(target_ply.points), axis=0)
aligned_points = np.asarray(source_ply_scale.points) - source_center + target_center
source_ply_scale.points = o3d.utility.Vector3dVector(aligned_points)


# source_ply_scale_down = source_ply_scale.voxel_down_sample(voxel_size = 10)
# target_ply_down = target_ply.voxel_down_sample(voxel_size = 10)

o3d.io.write_point_cloud(f"{source_path_root}/freegs_scale.ply",source_ply_scale,write_ascii=True)
o3d.io.write_point_cloud(f"{source_path_root}/gt.ply",target_ply,write_ascii=True)

T_mannual_align = np.array( [[0.258314341307, 0.470988810062, -0.843470931053, 551.601623535156],
                            [-0.881378531456, 0.472370624542, -0.006154851057, -47.947738647461],
                            [0.395532041788, 0.745007038116, 0.537139534950, 268.333679199219],
                            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])

source_ply_scale_mannual_align = copy.deepcopy(source_ply_scale)
source_ply_scale_mannual_align.points = o3d.utility.Vector3dVector(np.dot(T_mannual_align[:3,:3],np.array(source_ply_scale.points).T).T + T_mannual_align[:3,3])
o3d.io.write_point_cloud(f"{source_path_root}/freegs_scale_mannual_align.ply",source_ply_scale_mannual_align,write_ascii=True)


source,target,result,T = teaser_reg(source_ply_scale_mannual_align,target_ply,VOXEL_SIZE = 10)




vis_o3d_pcd_3(np.array(result.points),np.array(source.points),np.array(target.points),color1=[1,0,0],color2=[0,1,0],color3=[0,0,1])


print("end")
