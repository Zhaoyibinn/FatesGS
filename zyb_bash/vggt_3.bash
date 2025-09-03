scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
# scans=(37 40 55 63 65 69 83 97 105 106 110 114 118 122)



for scan in "${scans[@]}"
do
    python submodules/vggt/main_many23.py --scene_dir DTU/dtu_colmap_3_many/scan$scan
done