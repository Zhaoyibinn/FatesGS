
#!/bin/bash
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(69 83 97 105 106 110 114 118 122)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 
# cd /home/zhaoyibin/3DRE/3DGS/FatesGS

for scan in "${scans[@]}"
do  
   


    python zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh pilianghua_out/gs_init/fatesgs/scan$scan/train/ours_1000/fuse.ply --scan_id $scan --output_dir pilianghua_out/gs_init/fatesgs/scan$scan/train/ours_1000 --mask_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # python zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh pilianghua_out/gs_init/pgsr_vggt/scan$scan/train/ours_1000/tsdf_fusion.ply --scan_id $scan --output_dir pilianghua_out/gs_init/pgsr_vggt/scan$scan/train/ours_1000 --mask_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU
    # python zyb_tools/eval_dtu/evaluate_single_scene.py --input_mesh pilianghua_out/gs_init/pgsr_vggt/scan$scan/train/ours_15000/tsdf_fusion.ply --scan_id $scan --output_dir pilianghua_out/gs_init/pgsr_vggt/scan$scan/train/ours_15000 --mask_dir /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU --DTU /home/zhaoyibin/3DRE/3DGS/GSDF/data/DTU

done

