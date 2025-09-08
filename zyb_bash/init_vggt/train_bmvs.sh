
#!/bin/bash
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 
# cd /home/zhaoyibin/3DRE/3DGS/FatesGS

for scan in "${scans[@]}"
do  
   

    data_dir="DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan"

    output_dir="pilianghua_output_gsinit/scan$scan"
    # output_dir="output/ours_nodust_dtu$scan"
    # output_dir="output/set_23_24_33_dtu$scan"
    echo -e "${RED}$output_dir${NC}"

    # python train.py -s $data_dir -m $output_dir -r 2 --split mix --trim --init vggt_gs --absgs --resolution_mode freq --densify_mode freq
    python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.0006 --iteration 500

    # result_dir=$output_dir
    # DTU_dir=/home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # culled_mesh=$result_dir/culled_mesh.ply

    # $conda_env scripts/eval_dtu/evaluate_single_scene.py --input_mesh $result_dir/train/ours_15000/fuse_post.ply --scan_id $scan --output_dir $result_dir --mask_dir $DTU_dir --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done
