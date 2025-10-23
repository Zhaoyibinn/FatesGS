scans=(63)
source ~/.bashrc
# for scan in "${scans[@]}"
# do  
#     # cd /home/zhaoyibin/3DRE/3DGS/zancun/FatesGS
#     # conda activate fatesgs
#     data_dir="DTU/set_23_24_33_vggt_pose_colmap/dtu_3_images_vggt/scan$scan"
#     output_dir="output/3dgs/scan$scan"

#     # cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     # conda activate 3DGS
#     # python train.py -s $data_dir -m $output_dir --iterations 15000 -r 2 --test_iterations 1 100 200 300 400 500 600 700 800 900 1000 --save_iterations 1 100 200 300 400 500 600 700 800 900 1000


#     # cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     # conda activate 3DGSD
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 1
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 100
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 200
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 300
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 400
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 500
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 600
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 700
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 800
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 900
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 1000
#     # python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.001 --depth_trunc 10.0 --skip_test --iteration 15000


#     cd /home/zhaoyibin/3DRE/3DGS/FatesGS


#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 1 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 100 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 200 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 300 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 400 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 500 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 600 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 700 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 800 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 900 --scan $scan
#     python zyb_tools/eval_vggt/align_muti.py --input_root output/3dgs --iteration 1000 --scan $scan
    
# done
for scan in "${scans[@]}"
do  

    cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
    conda activate 2dgs_kd
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_vggt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/output/fatesgs/scan$scan
    python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/output/fatesgs/scan$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/output/3dgs/scan$scan
done
