source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 

# scan=37
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
scans_less=(40)



for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="DTU/diff/set_23_24_33/scan$scan "


    output_dir="pilianghua_out_new/xiaorong/noabs/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --mvs_filter
    # $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --absgs --mvs_filter
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="DTU/diff/set_23_24_33/scan$scan "


    output_dir="pilianghua_out_new/xiaorong/notrim/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --dust3r --absgs --mvs_filter
    # $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --absgs
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="DTU/diff/set_23_24_33/scan$scan "


    output_dir="pilianghua_out_new/xiaorong/nodust3r/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim  --absgs 
    # $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --mvs_filter --absgs 
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done



for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="DTU/diff/set_23_24_33/scan$scan "


    output_dir="pilianghua_out_new/xiaorong/nofatesgs/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --mvs_filter --absgs --lambda_feat 0.0 --lambda_depth 0.0
    # $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --diff --trim --dust3r --mvs_filter --absgs 
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done

