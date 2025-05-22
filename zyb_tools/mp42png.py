import cv2
import os

# 视频文件路径
video_path = '/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/diff/DTU_3_2/scan63/render.mp4'
# 保存帧的文件夹路径
save_folder = 'jiaojie_render'

def MP42PNG(video_path,images_dir):
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
    else:
        frame_count = 0
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 生成存储帧的文件名
                frame_filename = os.path.join(images_dir, f'{frame_count:04d}.png')
                # 保存帧为图像文件
                frame = cv2.resize(frame, (1554, 1162), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            else:
                # 无法读取帧，退出循环
                break

        # 释放视频捕获对象
        cap.release()
        print(f"已保存 {frame_count} 帧到 {images_dir} 文件夹中")
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
MP42PNG(video_path,save_folder)

