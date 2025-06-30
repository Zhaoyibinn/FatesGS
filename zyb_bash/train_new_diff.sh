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


    output_dir="pilianghua_out_new/diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/dust3r_abs_trim_splitmix_mvs_filter_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --dust3r --absgs --trim --mvs_filter --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/abs_trim_splitmix_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --absgs --trim --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/abs_trim_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --absgs --trim --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/abs_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --absgs --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/trim_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --trim --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/dust3r_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/dust3r_abs_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r --absgs --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/dust3r_abs_trim_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r --absgs --trim --diff
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
    data_dir="DTU/diff/set_23_24_33/scan$scan"


    output_dir="pilianghua_out_new/dust3r_abs_trim_splitmix_diff/dtu$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --dust3r --absgs --trim --diff
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done