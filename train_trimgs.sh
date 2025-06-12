source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 

# scan=37
scan=37

cd /home/zhaoyibin/3DRE/3DGS/FatesGS
conda_env="/home/zhaoyibin/anaconda3/envs/fatesgstrim/bin/python"
conda activate fatesgstrim
data_dir="DTU/set_23_24_33/scan$scan"


output_dir="output/test_dust$scan"
# output_dir="output/ours_nodust_dtu$scan"
# output_dir="output/set_23_24_33_dtu$scan"
echo -e "${RED}$output_dir${NC}"

$conda_env train.py -s $data_dir -m $output_dir -r 2
$conda_env render.py -s $data_dir -m $output_dir -r 2 

result_dir=$output_dir
DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
culled_mesh=$result_dir/culled_mesh.ply

$conda_env zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU


