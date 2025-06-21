
#!/bin/bash
scans=(5947b62af1b45630bd0c2a02)
# scans_less=(83 97 105 106 110 114 118 122)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 




for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
    conda activate fatesgs
    data_dir="BMVS_colmap/diff/$scan"

    output_dir="BMVS_OUT/diff_ssim/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 1 --diff --not_record
    $conda_env render.py -s $data_dir -m $output_dir -r 1 --diff 

    # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
done

for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
    conda activate fatesgs
    data_dir="BMVS_colmap/diff/$scan"

    output_dir="BMVS_OUT/origin/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 1 --diff --origin_train --not_record
    $conda_env render.py -s $data_dir -m $output_dir -r 1 --diff 

    # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
done




