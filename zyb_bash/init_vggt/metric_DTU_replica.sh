# scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 





for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
    conda activate 2dgs_kd
    python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/DTU/2dgs_vggt/scan$scan


done
