
#!/bin/bash
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
# scans=(room1_sparse)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 
# cd /home/zhaoyibin/3DRE/3DGS/FatesGS

# for scan in "${scans[@]}"
# do  
#     data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

#     output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_extrapose/scan$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim -r 2

#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1  -r 2
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1000  -r 2
# done

# for scan in "${scans[@]}"
# do  
#     data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

#     output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_depth/scan$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_pose --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 0.0 -r 2

#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1  -r 2
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1000  -r 2
# done

# for scan in "${scans[@]}"
# do  
#     data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

#     output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_feat/scan$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_pose --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_feat 0.0 -r 2

#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1   -r 2
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1000   -r 2
# done


# for scan in "${scans[@]}"
# do  
#     data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

#     output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_trim/scan$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1 --absgs -r 2

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1  -r 2

# done

# for scan in "${scans[@]}"
# do  
#     data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

#     output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_denseGS/scan$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init colmap --iterations 1 --absgs --trim -r 2

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1  -r 2

# done

for scan in "${scans[@]}"
do  
    data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

    output_dir="pilianghua_out/gs_init/DTU/xiaorong/no_smooth/scan$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_pose --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_dsmooth 0.0 -r 2

    # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1  -r 2
    python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1000  -r 2
done