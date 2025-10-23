scan=83

python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 100 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 200 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 300 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 400 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 500 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 600 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 700 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 800 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 900 --voxel_size 0.001

python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 1 --voxel_size 0.001
python render.py -s DTU/set_23_24_33_vggt_initok/dtu_3_images_vggt/scan$scan -m output/gsinit/scan$scan --extra_pose --depth_trunc 10.0 --iteration 1000 --voxel_size 0.001


python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 1 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 100 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 200 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 300 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 400 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 500 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 600 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 700 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 800 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 900 --scan $scan
python zyb_tools/eval_vggt/align_muti.py --input_root output/gsinit --iteration 1000 --scan $scan