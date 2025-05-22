
#!/bin/bash
scans=(59338e76772c3e6384afbb15)
# scans=(37)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 


for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
    conda_env="/home/zhaoyibin/anaconda3/envs/3DGS/bin/python"
    conda activate 3DGS
    data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap/ok_3dgs/$scan"

    output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_OUT/3dgs/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir 
    # $conda_env render.py -s $data_dir -m $output_dir 

    cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
    conda_env="/home/zhaoyibin/anaconda3/envs/3DGSD/bin/python"
    conda activate 3DGSD
    $conda_env render_2dgs.py -s $data_dir -m $output_dir --skip_train --skip_test

    # cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    # conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
    # conda activate fatesgs
    # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
done

