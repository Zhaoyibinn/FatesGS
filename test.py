import open3d as o3d

# 读取点云文件
point_cloud_file = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan40/sparse/vggt/points3D.ply"  # 替换为你的点云文件路径
point_cloud = o3d.io.read_point_cloud(point_cloud_file)

# 检查点云是否加载成功
if not point_cloud.has_points():
    print("点云加载失败，请检查文件路径或格式。")
    exit()

# 使用 Alpha Shape 方法将点云转换为网格
alpha = 0.005  # 调整 alpha 值以控制网格的细节程度
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)

# 或者使用 Ball Pivoting 方法
# radii = [0.005, 0.01, 0.02]  # 调整半径以控制网格生成
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     point_cloud, o3d.utility.DoubleVector(radii)
# )

# 平滑网格
mesh.compute_vertex_normals()

# 保存网格为文件
mesh_file = "DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan40/sparse/vggt/points3D_vggt_mesh.ply"  # 替换为你想保存的文件路径
o3d.io.write_triangle_mesh(mesh_file, mesh)

print(f"网格已保存到: {mesh_file}")