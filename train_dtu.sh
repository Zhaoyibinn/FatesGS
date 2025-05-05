
#!/bin/bash
scans=(24 37 63 65 69 83 97 105 106 110 114 118 122)
# scans_less=(118 122)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 


for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
    conda activate fatesgs
    data_dir="DTU/diff/scan$scan"

    output_dir="pilianghua_output/ssim_l1_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2 --diff --lambda_local_pearson 0  --lambda_diff_ssim 0.2 --lambda_diff_l1 0.2
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    # conda_env="/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python"
    # conda activate 2dgs_kd
    # cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
done

# for scan in "${scans[@]}"
# do  
#     cd /home/zhaoyibin/3DRE/3DGS/FatesGS
#     conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
#     conda activate fatesgs
#     data_dir="DTU/diff/scan$scan"

#     output_dir="pilianghua_output/ssimdiff_person/dtu$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     $conda_env train.py -s $data_dir -m $output_dir -r 2 --diff --lambda_local_pearson 0.15
#     $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff



#     result_dir=$output_dir
#     DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
#     culled_mesh=$result_dir/culled_mesh.ply

#     $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
# done





# for scan in "${scans[@]}"
# do  
#     cd /home/zhaoyibin/3DRE/3DGS/FatesGS
#     conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
#     conda activate fatesgs
#     data_dir="DTU/diff/scan$scan"

#     output_dir="pilianghua_output/origin/dtu$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     $conda_env train.py -s $data_dir -m $output_dir -r 2 --diff --lambda_local_pearson 0 --origin_train
#     $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

#     result_dir=$output_dir
#     DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
#     culled_mesh=$result_dir/culled_mesh.ply

#     $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
# done

# for scan in "${scans[@]}"
# do  
#     cd /home/zhaoyibin/3DRE/3DGS/FatesGS
#     conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
#     conda activate fatesgs
#     data_dir="DTU/diff/scan$scan"

#     output_dir="pilianghua_output/alllossdiff/dtu$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     $conda_env train.py -s $data_dir -m $output_dir -r 2 --diff --lambda_local_pearson 0 --lambda_diff_rgb 1 --lambda_diff_rend_dist 1 --lambda_diff_normal 1 --lambda_diff_dsmooth 1 --lambda_diff_depth 10
#     $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff 



#     result_dir=$output_dir
#     DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
#     culled_mesh=$result_dir/culled_mesh.ply

#     $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
# done



# for scan in "${scans[@]}"
# do  
#     cd /home/zhaoyibin/3DRE/3DGS/FatesGS
#     conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"
#     conda activate fatesgs
#     data_dir="DTU/diff/scan$scan"

#     output_dir="pilianghua_output/ssimdiff_manyrender/dtu$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     $conda_env train.py -s $data_dir -m $output_dir -r 2 --diff --lambda_local_pearson 0
#     $conda_env render.py -s $data_dir -m $output_dir -r 2 

#     result_dir=$output_dir
#     DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
#     culled_mesh=$result_dir/culled_mesh.ply

#     $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
# done


