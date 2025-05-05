scans=(114)
# conda_env="/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python"
conda_env="/home/zhaoyibin/anaconda3/envs/fatesgs/bin/python"


for scan in "${scans[@]}"
do
    # data_dir="DTU/diff/scan$scan"
    # output_dir="output/ours_dtu$scan"
    result_dir=output/test
    DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env zybtools/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    $conda_env zyb_tools/eval_dtu/eval.py --data $culled_mesh --scan $scan --mode mesh --dataset_dir $DTU_dir --vis_out_dir $result_dir

    # $conda_env render.py -s $data_dir -m $output_dir -r 2 --diff
done