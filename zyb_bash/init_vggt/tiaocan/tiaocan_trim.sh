scans=(office0_sparse office1_sparse office2_sparse room0_sparse room1_sparse)
# scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(97)
# scans_less=(55)
source ~/.bashrc
for scan in "${scans[@]}"
do  
    python train.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/10_500/$scan --split mix --trim --init vggt_gs --absgs --iterations 1000 --extra_pose --extra_q_lr 1e-7 --extra_t_lr 1e-8 --contribution_prune_ratio 0.1 --contribution_prune_interval 500 --resolution_mode freq --densify_mode freq
    python render.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/10_500/$scan --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose
done

for scan in "${scans[@]}"
do  
    python train.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/20_500/$scan --split mix --trim --init vggt_gs --absgs --iterations 1000 --extra_pose --extra_q_lr 1e-7 --extra_t_lr 1e-8 --contribution_prune_ratio 0.2 --contribution_prune_interval 500 --resolution_mode freq --densify_mode freq
    python render.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/20_500/$scan --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose
done

for scan in "${scans[@]}"
do  
    python train.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/10_300/$scan --split mix --trim --init vggt_gs --absgs --iterations 1000 --extra_pose --extra_q_lr 1e-7 --extra_t_lr 1e-8 --contribution_prune_ratio 0.1 --contribution_prune_interval 300 --resolution_mode freq --densify_mode freq
    python render.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/10_300/$scan --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose
done

for scan in "${scans[@]}"
do  
    python train.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/20_300/$scan --split mix --trim --init vggt_gs --absgs --iterations 1000 --extra_pose --extra_q_lr 1e-7 --extra_t_lr 1e-8 --contribution_prune_ratio 0.2 --contribution_prune_interval 300 --resolution_mode freq --densify_mode freq
    python render.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/20_300/$scan --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose
done

for scan in "${scans[@]}"
do  
    python train.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/5_300/$scan --split mix --trim --init vggt_gs --absgs --iterations 1000 --extra_pose --extra_q_lr 1e-7 --extra_t_lr 1e-8 --contribution_prune_ratio 0.05 --contribution_prune_interval 300 --resolution_mode freq --densify_mode freq
    python render.py -s Replica/replica_vggt_more/$scan -m pilianghua_out/gs_init/Replica/tiaocan_trim/5_300/$scan --depth_trunc 10.0 --voxel_size 0.01 --iteration 1000 --extra_pose
done
