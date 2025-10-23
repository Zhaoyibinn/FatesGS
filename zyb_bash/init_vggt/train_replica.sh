
#!/bin/bash

source ~/.bashrc
scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)

conda activate fategsvggt




# for scan in "${scans[@]}"
# do  
#     data_dir="Replica/replica_vggt_more/$scan"

#     output_dir="pilianghua_out/gs_init/Replica/gsinit_more/2dgsok_trim0_extrapose_q1e-6t1e-8_depth20/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 20.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1  --extra_pose
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done

# for scan in "${scans[@]}"
# do  
#     data_dir="Replica/replica_vggt_more/$scan"

#     output_dir="pilianghua_out/gs_init/Replica/gsinit_more/2dgsok_trim0_extrapose_q1e-6t1e-8_depth20_smooth2.0/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 20.0 --lambda_dsmooth 2.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1  --extra_pose
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done

# for scan in "${scans[@]}"
# do  
#     data_dir="Replica/replica_vggt_more/$scan"

#     output_dir="pilianghua_out/gs_init/Replica/gsinit_more/2dgsok_trim0_extrapose_q1e-6t1e-8_smooth2.0/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_dsmooth 2.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1  --extra_pose
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done








# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     conda activate 3DGS
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_gt/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/3dgs_gt/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 30000


#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     conda activate 3DGSD
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --skip_test --iteration 1000
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --skip_test --iteration 30000
# done


# for scan in "${scans[@]}"
# do  
#     data_dir="Replica/replica_vggt_more/$scan"

#     output_dir="pilianghua_out/gs_init/Replica/gsinit_more/2dgsok_trim0_extrapose_q1e-6t1e-8_depth20/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 20.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1  --extra_pose
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done



# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/zancun/FatesGS
#     conda activate fatesgs
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_vggt_more_camerapose/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/fategs_vggt/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 30000 

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --iteration 30000
# done


for scan in "${scans[@]}"
do  

    cd /home/zhaoyibin/3DRE/3DGS/PGSR
    conda activate pgsr
    data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_gt/$scan"
    output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/pgsr_gt/$scan"
    python train.py -s $data_dir -m $output_dir --quiet -r2 --ncc_scale \ 0.5 --iterations 30000

    python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 1000
    python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 30000 
done


# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
#     conda activate 2dgs_kd
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_freegs/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/freegs/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 1 --sh_degree 0

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 1 --sh_degree 0
#     # python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 30000 
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     conda activate 3DGS
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_cfgs/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/cfgs/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 1


#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     conda activate 3DGSD
#     python render_2dgs.py -s $data_dir -m $output_dir  --depth_trunc 10.0 --skip_test --iteration 1
    
# done


# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     conda activate 3DGS
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/Replica/replica_cfgs_ok/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/Replica/cfgs_ok/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 1


#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     conda activate 3DGSD
#     python render_2dgs.py -s $data_dir -m $output_dir  --depth_trunc 10.0 --skip_test --iteration 1
    
# done