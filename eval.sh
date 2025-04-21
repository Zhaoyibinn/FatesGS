CUDA_VISIBLE_DEVICES=$1 python eval/eval_dtu/dtu_eval.py \
    --input_mesh output/DTU/set_23_24_33/$2/SCAN_NAME/train/ours_15000/fuse_post.ply \
    --output_dir output/DTU/set_23_24_33/$2/SCAN_NAME/train/ours_15000 \
    --mask /mnt/ssd_1/dc/gaussian-surfels/dtu-pixelnerf \
    --DTU /mnt/ssd_0/wyl/dtu_gt
