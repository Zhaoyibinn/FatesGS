
#!/bin/bash
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(55 83 114)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 
# cd /home/zhaoyibin/3DRE/3DGS/FatesGS


   


python zyb_tools/eval_vggt/align_muti_freegs.py --input_root pilianghua_out/gs_init/DTU/xiaorong/no_denseGS --iteration 1

python zyb_tools/eval_vggt/align_muti_freegs.py --input_root pilianghua_out/gs_init/DTU/xiaorong/no_depth --iteration 1000

python zyb_tools/eval_vggt/align_muti_freegs.py --input_root pilianghua_out/gs_init/DTU/xiaorong/no_extrapose --iteration 1000

python zyb_tools/eval_vggt/align_muti_freegs.py --input_root pilianghua_out/gs_init/DTU/xiaorong/no_feat --iteration 1000

python zyb_tools/eval_vggt/align_muti_freegs.py --input_root pilianghua_out/gs_init/DTU/xiaorong/no_trim --iteration 1

