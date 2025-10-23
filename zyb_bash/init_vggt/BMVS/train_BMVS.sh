
#!/bin/bash

source ~/.bashrc
# scans=(5a48d4b2c7dab83a7d7b9851 5a69c47d0d5d0a7f3b2e9752 5a588a8193ac3d233f77fbca 5a618c72784780334bc1972d 5a6464143d809f1d8208c43c 5aa235f64a17b335eeaf9609 5b908d3dc6ab78485f3d24a9 5947b62af1b45630bd0c2a02 59338e76772c3e6384afbb15)
# scans=(5a48d4b2c7dab83a7d7b9851 5a69c47d0d5d0a7f3b2e9752 5a618c72784780334bc1972d 5a6464143d809f1d8208c43c 5b908d3dc6ab78485f3d24a9 59338e76772c3e6384afbb15)
scans=(5a48d4b2c7dab83a7d7b9851 5a69c47d0d5d0a7f3b2e9752 5a618c72784780334bc1972d 5b908d3dc6ab78485f3d24a9 59338e76772c3e6384afbb15)

conda activate fategsvggt









# for scan in "${scans[@]}"
# do  
#     data_dir="BMVS_colmap_new/BMVS_origin_vggt_3/$scan"

#     output_dir="pilianghua_out/gs_init/BMVS/gsinit/2dgsok_trim0_extrapose_q1e-6t1e-8_depth20/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init vggt_gs --iterations 1000 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 20.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.005 --iteration 1  --extra_pose
#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.005 --iteration 1000  --extra_pose
# done

# for scan in "${scans[@]}"
# do  
#     data_dir="BMVS_colmap_new/BMVS_origin_vggt_3/$scan"

#     output_dir="pilianghua_out/gs_init/BMVS/vggt/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --split mix  --init colmap --iterations 1 --absgs --trim --extra_q_lr 1e-6 --extra_t_lr 1e-8 --lambda_depth 20.0

#     python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.005 --iteration 1  --extra_pose
#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done


# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     conda activate 3DGS
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_gt_3/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/3dgs_gt/$scan"
#     # python train.py -s $data_dir -m $output_dir --iterations 30000


#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     conda activate 3DGSD
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --skip_test --iteration 1000
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.01 --depth_trunc 10.0 --skip_test --iteration 30000
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/PGSR
#     conda activate pgsr
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_gt_3/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/psgr_gt/$scan"
#     # python train.py -s $data_dir -m $output_dir --quiet -r2 --ncc_scale \ 0.5 --iterations 30000

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 30000 
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
#     conda activate 2dgs_kd
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_gt_3/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_gt/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --iterations 30000 --sh_degree 0

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 30000
    
#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/zancun/FatesGS
#     conda activate fatesgs
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_gt_3/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_gt/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 30000 

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --iteration 30000
# done



# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
#     conda activate 3DGS
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_vggt_3_camerapose/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/3dgs_vggt/$scan"
#     # python train.py -s $data_dir -m $output_dir --iterations 30000


#     cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
#     conda activate 3DGSD
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --skip_test --iteration 1000
#     python render_2dgs.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --skip_test --iteration 30000
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/PGSR
#     conda activate pgsr
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_vggt_3_camerapose/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/psgr_vggt/$scan"
#     # python train.py -s $data_dir -m $output_dir --quiet -r2 --ncc_scale \ 0.5 --iterations 30000

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --iteration 30000 
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
#     conda activate 2dgs_kd
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_vggt_3_camerapose/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/2dgs_vggt/$scan"
#     # output_dir="output/ours_nodust_dtu$scan"
#     # output_dir="output/set_23_24_33_dtu$scan"
#     echo -e "${RED}$output_dir${NC}"

#     # python train.py -s $data_dir -m $output_dir --iterations 30000 --sh_degree 0

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --iteration 30000
    
#     # python render.py -s $data_dir -m $output_dir --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000  --extra_pose
# done

# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/zancun/FatesGS
#     conda activate fatesgs
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_origin_vggt_3_camerapose/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/fategs_vggt/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 30000 

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --iteration 1000
#     python render.py -s $data_dir -m $output_dir --voxel_size 0.005 --depth_trunc 10.0 --iteration 30000
# done



# for scan in "${scans[@]}"
# do  

#     cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting-origin/2d-gaussian-splatting
#     conda activate 2dgs_kd
#     data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_freegs/$scan"
#     output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/freegs/$scan"
#     python train.py -s $data_dir -m $output_dir --iterations 1 --sh_degree 0

#     python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 1 --sh_degree 0
#     # python render.py -s $data_dir -m $output_dir --voxel_size 0.01 --iteration 30000 
# done

for scan in "${scans[@]}"
do  

    cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_origin
    conda activate 3DGS
    data_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/BMVS_colmap_new/BMVS_cfgs/$scan"
    output_dir="/home/zhaoyibin/3DRE/3DGS/FatesGS/pilianghua_out/gs_init/BMVS/cfgs/$scan"
    python train.py -s $data_dir -m $output_dir --iterations 1


    cd /home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth
    conda activate 3DGSD
    python render_2dgs.py -s $data_dir -m $output_dir  --depth_trunc 10.0 --skip_test --iteration 1
done
    