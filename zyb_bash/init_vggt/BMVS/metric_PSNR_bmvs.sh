scans=(5a48d4b2c7dab83a7d7b9851 5a69c47d0d5d0a7f3b2e9752 5a618c72784780334bc1972d 5b908d3dc6ab78485f3d24a9 59338e76772c3e6384afbb15)
scans_less=(5a48d4b2c7dab83a7d7b9851)

# scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
RED='\033[0;31m'
# 重置颜色的ANSI转义序列
NC='\033[0m' 





# for scan in "${scans_less[@]}"
# do  
#     cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
#     conda activate 2dgs_kd
#     # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_gt/$scan
#     python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_vggt/$scan
#     # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/3dgs_gt/$scan
#     python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/3dgs_vggt/$scan
#     # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/gsinit/2dgsok_trim0_extrapose_q1e-6t1e-8_depth20/$scan
#     # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/pgsr_gt/$scan
#     python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/pgsr_vggt/$scan
#     # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/vggt/$scan

# done


for scan in "${scans[@]}"
do  
    cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
    conda activate 2dgs_kd
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_gt/$scan
    # python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_vggt/$scan
    python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/cfgs/$scan
    python  metrics.py -m /home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/freegs/$scan
done
