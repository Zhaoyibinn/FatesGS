import numpy as np
import gradio as gr
import cv2


import torch
import cv2
import os
import open3d as o3d

from utils import image_utils
import trimesh

from gradio_train import gradio_train_input
from gradio_render import gradio_render_input

# def read_ply(path):
#     if path==None:
#         return None
#     ply_path = os.path.join(path, "sparse/vggt/points3D.ply")
#     if not os.path.exists(ply_path):
#         return None
#     point_cloud = o3d.io.read_point_cloud(ply_path)
#     file_dir, file_name = os.path.split(ply_path)
#     file_base, _ = os.path.splitext(file_name)
#     output_file = os.path.join(file_dir, f"{file_base}.glb")
    
#     scene_3d = trimesh.Scene()

#     # Add point cloud data to the scene
#     point_cloud_data = trimesh.PointCloud(vertices=np.array(point_cloud.points), colors=np.array(np.array(point_cloud.colors)))
#     scene_3d.add_geometry(point_cloud_data)
#     scene_3d.export(file_obj=output_file)
#     # test_path = "/home/zhaoyibin/3DRE/MVS/vggt/input_images_20250720_125041_452644/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"
#     return output_file


with gr.Blocks() as demo:
    gr.Markdown("# ZYB: VGGT + Fatesgs; 一个简单的少视角重建DEMO")
    gr.Markdown("请输入三张图片 横屏 然后点击Train按钮开始训练")
    gr.Markdown("点击Train之后需要一段时间做预处理 包含深度估计、VGGT等 请耐心等待")
    gr.Markdown("在*迭代次数*不再变化之后 GS已经优化完毕 点击Render Mesh开始合成表面")
    gr.Markdown("一般来说 会在迭代200轮左右就呈现比较良好的视图合成结果")
    model_path = gr.State(value=None)
    source_path = gr.State(value=None)

    with gr.Row():
        # path = gr.Textbox(label="路径", value="DTU/wo_pose/scan24")
        input_image = gr.File(file_count="multiple", label="Upload Images", interactive=True)
        iteration = gr.Textbox(label="迭代次数")
        pcd_view = gr.Model3D(label="训练过程中的点云",display_mode="point_cloud")
        output = gr.Image(label="render",width="30vw")
    
    greet_btn = gr.Button("Train")

    greet_btn.click(fn=gradio_train_input, inputs=input_image, outputs=[iteration, output,pcd_view,model_path,source_path])

    with gr.Row():
        mesh_view = gr.Model3D(label="最终的表面训练结果")
    render_btn = gr.Button("Render Mesh")

    render_btn.click(fn=gradio_render_input, inputs=[model_path,source_path], outputs=mesh_view)

    


    # time_gr = gr.Timer(value=1 # 每秒触发一次
    # )
    # time_gr.tick(fn=read_ply,inputs=path,outputs=pcd_view)






demo.launch(allowed_paths=["/"],share=True)
