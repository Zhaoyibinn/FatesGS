
allscans=(5b4933abf2b5f44e95de482a 5b22269758e2823a67a3bd03 5ba19a8a360c7c30c1c169df 5bccd6beca24970bce448134 5be47bf9b18881428d8fbc1d)
scans=(5b950c71608de421b1e7318f)
# python render.py -s BMVS_PACG/3DGS/5aa515e613d42d091d29d300 -m BMVS_OUT_PACG/fatesgs/5aa515e613d42d091d29d300 --depth_trunc 1000.0

# for scan in "${scans[@]}"
# do  

#     python train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/PACG/$scan -r 1 --split mix --trim --dust3r --absgs
#     python render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/PACG/$scan --depth_trunc 10.0

#     python train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/fatesgs/$scan -r 1 --split ordinary
#     python render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/fatesgs/$scan --depth_trunc 10.0

#     /home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting/train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/2DGS/$scan
#     /home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting/render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/2DGS/$scan
# done

# for scan in "${scans[@]}"
# do  

#     python train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/PACG/$scan -r 1 --split mix --trim --dust3r --absgs
#     python render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/PACG/$scan --depth_trunc 10.0

#     python train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/fatesgs/$scan -r 1 --split ordinary
#     python render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/fatesgs/$scan --depth_trunc 10.0
# done

for scan in "${scans[@]}"
do  



    /home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting/train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/2DGS/$scan
    /home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting/render.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/2DGS/$scan --depth_trunc 10.0

    /home/zhaoyibin/anaconda3/envs/3DGS/bin/python /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin/train.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/3dgs/$scan
    /home/zhaoyibin/anaconda3/envs/3DGSD/bin/python /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth/render_2dgs.py -s BMVS_PACG/3DGS/$scan -m BMVS_OUT_PACG/3dgs/$scan --depth_trunc 10.0 
done


