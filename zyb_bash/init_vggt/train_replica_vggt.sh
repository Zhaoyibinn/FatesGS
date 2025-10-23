
#!/bin/bash
scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 
# cd /home/zhaoyibin/3DRE/3DGS/FatesGS





for scan in "${scans[@]}"
do  
   

    data_dir="Replica/replica_vggt_more/$scan"

    output_dir="pilianghua_out/gs_init/Replica/vggt_more/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim 
    python train.py -s $data_dir -m $output_dir --split mix  --init colmap --iterations 1

    python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1 --extra_pose
    # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 500 
    # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose


    # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.001 --iteration 1000 --extra_pose
    # # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env scripts/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done
