scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
# scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
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
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/2dgs_vggt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/3dgs_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/3dgs_vggt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/cfgs/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/cfgs_ok/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/fategs_vggt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/freegs/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/gsinit_more/2dgsok_trim0_extrapose_nodensify_q1e-5t1e-8_nofeat/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/pgsr_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/pgsr_vggt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/vggt_more/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/fategs_gt/$scan 
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/2dgs_gt/$scan 
done
