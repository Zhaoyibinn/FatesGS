
scans=(office2_sparse room1_sparse office0_sparse office1_sparse room0_sparse)


for scan in "${scans[@]}"
do  
    python render.py -s Replica/diff/$scan -m pilianghua_out_new_rep/dust3r_abs_trim_splitmix_diff/$scan --depth_trunc 10.0 -r 2 --diff
done

