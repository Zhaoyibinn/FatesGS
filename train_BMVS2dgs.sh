
#!/bin/bash
scans=(59338e76772c3e6384afbb15)
scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 

cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting

for scan in "${scans[@]}"
do  
   
    conda_env="/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python"
    conda activate 2dgs_kd
    data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap/ok_3dgs/$scan"

    output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_OUT/2dgs/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir
    $conda_env render.py -s $data_dir -m $output_dir --depth_trunc 5.0

    # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env scripts/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
done
