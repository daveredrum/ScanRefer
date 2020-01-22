# ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

<p align="center"><img src="demo/ScanRefer.gif"/></p>

## Introduction

We introduce the new task of 3D object localization in RGB-D scans using natural language descriptions. As input, we assume a point cloud of a scanned 3D scene along with a free-form description of a specified target object. To address this task, we propose ScanRefer, where the core idea is to learn a fused descriptor from 3D object proposals and encoded sentence embeddings. This learned descriptor then correlates the language expressions with the underlying geometric features of the 3D scan and facilitates the regression of the 3D bounding box of the target object. In order to train and benchmark our method, we introduce a new ScanRefer dataset, containing 46,173 descriptions of 9,943 objects from 703 [ScanNet](http://www.scan-net.org/) scenes. ScanRefer is the first large-scale effort to perform object localization via natural language expression directly in 3D.

Please also check out the project video [here](https://youtu.be/T9J5t-UEcNA).

For additional detail, please see the ScanRefer paper:  
"[ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/abs/1912.08830)"  
by [Dave Zhenyu Chen](https://www.niessnerlab.org/members/zhenyu_chen/profile.html), [Angel X. Chang](https://angelxuanchang.github.io/) and [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)  
from [Technical University of Munich](https://www.tum.de/en/) and [Simon Fraser University](https://www.sfu.ca/).

## Dataset

If you would like to access to the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet project page](https://github.com/ScanNet/ScanNet) for more details.

Download the dataset by simply executing the wget command:
```shell
wget <download_link>
```

### Data format
```
"scene_id": [ScanNet scene id, e.g. "scene0000_00"],
"object_id": [ScanNet object id (corresponds to "objectId" in ScanNet aggregation file), e.g. "34"],
"object_name": [ScanNet object name (corresponds to "label" in ScanNet aggregation file), e.g. "coffee_table"],
"ann_id": [description id, e.g. "1"],
"description": [...],
"token": [a list of tokens from the tokenized description] 
```

## Usage
### Data preparation
1. Download the ScanRefer dataset and the preprocessed [GLoVE embeddings](http://kaldir.vc.in.tum.de/glove.p), put them under `data/`
2. Download the [ScanNetV2 dataset](https://github.com/ScanNet/ScanNet) and put (or link) `scans` under (or to) `data/scannet/scans`
3. Pre-process ScanNet data:
```shell
cd data/scannet/
python batch_load_scannet_data.py
```
4. Download the preprocessed [multiview features](http://kaldir.vc.in.tum.de/enet_feats.hdf5) and put it under `data/scannet_data`

### Training
To train the ScanRefer model with multiview features:
```shell
python scripts/train.py --use_multiview
```
For more training options, please run `scripts/train.py -h`.

### Evaluation
To evaluate the trained ScanRefer models, please find the folder under `outputs/` with the current timestamp and run:
```shell
python scripts/eval.py --folder <folder_name> --use_multiview
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`

### Visualize
To predict the localization results predicted by the trained ScanRefer model in a specific scene, please find the corresponding folder under `outputs/` with the current timestamp and run:
```shell
python scripts/visualize.py --folder <folder_name> --scene_id <scene_id> --use_multiview
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`. The output `.ply` files will be stored under `vis/<scene_id>/`

## Citation

If you use the ScanRefer data or code in your work, please kindly cite our work and the original ScanNet paper:

```
@misc{chen2019scanrefer,
    title={ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language},
    author={Dave Zhenyu Chen and Angel X. Chang and Matthias Nießner},
    year={2019},
    eprint={1912.08830},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{dai2017scannet,
    title={Scannet: Richly-annotated 3d reconstructions of indoor scenes},
    author={Dai, Angela and Chang, Angel X and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={5828--5839},
    year={2017}
}
```

## Acknowledgement
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection codebase and [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) for the CUDA accelerated PointNet++ implementation.

## License
ScanRefer is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2020 Dave Zhenyu Chen, Angel X. Chang, Matthias Nießner
