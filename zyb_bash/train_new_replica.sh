source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 

# scan=37
scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
scans_less=(office2_sparse room1_sparse)

for scan in "${scans_less[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/origin/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary
    $conda_env render.py -s $data_dir -m $output_dir -r 2 --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans_less[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/dust3r/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans_less[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/dust3r_abs/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r --absgs
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans_less[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/dust3r_abs_trim/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --dust3r --absgs --trim
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans_less[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/dust3r_abs_trim_splitmix/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --dust3r --absgs --trim
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/dust3r_abs_trim_splitmix_mvs_filter/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --dust3r --absgs --trim --mvs_filter
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/abs_trim_splitmix/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split mix --absgs --trim
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/abs_trim/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --absgs --trim
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/abs/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --absgs
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done

for scan in "${scans[@]}"
do  
   
    cd /home/zhaoyibin/3DRE/3DGS/FatesGS
    conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrimabs/bin/python"
    conda activate fatesgstrimabs
    data_dir="Replica/replica_gsicpslam/$scan "


    output_dir="pilianghua_out_new_rep/trim/$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    $conda_env train.py -s $data_dir -m $output_dir -r 2  --split ordinary --trim
    $conda_env render.py -s $data_dir -m $output_dir -r 2  --depth_trunc 10.0

    result_dir=$output_dir
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    $conda_env zyb_tools/eval_replica/eval.py --data $result_dir/train/ours_15000/fuse_post.ply --mode mesh --dataset_dir $data_dir --vis_out_dir $output_dir

done