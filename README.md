<h2 align="center">
  😇FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with Depth-Feature Consistency
</h2>
<h4 align="center">AAAI 2025 Oral</h4>
<div align="center">
  <a href='https://github.com/alvin528' target='_blank'>Han Huang</a>*&emsp;
  <a href='https://yulunwu0108.github.io/' target='_blank'>Yulun Wu</a>*&emsp;
  Chao Deng&emsp;
  Ge Gao&dagger;&emsp;
  Ming Gu&emsp;
  <a href='https://yushen-liu.github.io/' target='_blank'>Yu-Shen Liu</a><br>
  Tsinghua University<br>
  <small>*Equal contribution.&emsp;&dagger;Corresponding author.</small>
</div>
<p align="center">
  <a href="https://arxiv.org/abs/2501.04628" target='_blank'><img src="http://img.shields.io/badge/cs.CV-arXiv%3A2501.04628-b31b1b"></a>
  <a href="https://alvin528.github.io/FatesGS/" target='_blank'><img src="http://img.shields.io/badge/Project_Page-😇-lightblue"></a>
</p>

## Overview

<div align="center"><img src="./media/overview.png" width=100%></div>

We propose FatesGS for sparse-view surface reconstruction, taking full advantage of the Gaussian Splatting pipeline. Compared with previous methods, our approach neither requires long-term per-scene optimization nor costly pre-training.

## Installation

```
conda create -n fatesgs python=3.8
conda activate fatesgs
pip install -r requirements.txt
```

## Dataset

### DTU dataset

1. Download the processed DTU dataset from [this link](https://drive.google.com/drive/folders/143jIT9DJN17gigp3uxBtqBv6UBdcO7Lm?usp=drive_link). The data structure should be like:

```
|-- DTU
    |-- <set_name, e.g. set_23_24_33>
        |-- <scan_name, e.g. scan24>
            |-- pair.txt
            |-- images
                |-- 0000.png
                |-- 0001.png
                ...
            |-- sparse <COLMAP sparse reconstruction>
                |-- 0
                    |-- cameras.txt
                    |-- images.txt
                    |-- points3D.txt
            |-- dense <COLMAP dense reconstruction>
                |-- fused.ply
                ...
            |-- depth_npy <monocular depth maps (to be generated)>
                |-- 0000_pred.npy
                |-- 0001_pred.npy
                ...
        ...
    ...
```

2. Following [Marigold](https://github.com/prs-eth/Marigold) to generate the estimated monocular depth maps. Put the `.npy` format depth maps under the `depth_npy` folder. You may also use more advanced depth estimation models for better performance. (P.S. The size of the depth maps used as priors ought to be consistent with those of the rendered color images during the Gaussian Splatting training process.)

## Running

- Training

```
CUDA_VISIBLE_DEVICES=0
python train.py -s <source_path> -m <model_path> -r 2
```

- Extract mesh

```
CUDA_VISIBLE_DEVICES=0
python render.py -s <source_path> -m <model_path> -r 2 --skip_test --skip_train
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{huang2025fatesgs,
    title={FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with Depth-Feature Consistency},
    author={Han Huang and Yulun Wu and Chao Deng and Ge Gao and Ming Gu and Yu-Shen Liu},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```

## Acknowledgement

This implementation is based on [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [MVSDF](https://github.com/jzhangbs/MVSDF). Thanks to the authors for their great work.