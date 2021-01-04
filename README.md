# ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

<p align="center"><img src="demo/ScanRefer.gif" width="600px"/></p>

## Introduction

We introduce the new task of 3D object localization in RGB-D scans using natural language descriptions. As input, we assume a point cloud of a scanned 3D scene along with a free-form description of a specified target object. To address this task, we propose ScanRefer, where the core idea is to learn a fused descriptor from 3D object proposals and encoded sentence embeddings. This learned descriptor then correlates the language expressions with the underlying geometric features of the 3D scan and facilitates the regression of the 3D bounding box of the target object. In order to train and benchmark our method, we introduce a new ScanRefer dataset, containing 51,583 descriptions of 11,046 objects from 800 [ScanNet](http://www.scan-net.org/) scenes. ScanRefer is the first large-scale effort to perform object localization via natural language expression directly in 3D.

Please also check out the project website [here](https://daveredrum.github.io/ScanRefer/).

For additional detail, please see the ScanRefer paper:  
"[ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/abs/1912.08830)"  
by [Dave Zhenyu Chen](https://www.niessnerlab.org/members/zhenyu_chen/profile.html), [Angel X. Chang](https://angelxuanchang.github.io/) and [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)  
from [Technical University of Munich](https://www.tum.de/en/) and [Simon Fraser University](https://www.sfu.ca/).

## :star2: Benchmark Challenge :star2:
We provide the ScanRefer Benchmark Challenge for benchmarking your model automatically on the hidden test set! Learn more at our [benchmark challenge website](http://kaldir.vc.in.tum.de/scanrefer_benchmark/).
After finishing training the model, please download [the benchmark data](http://kaldir.vc.in.tum.de/scanrefer_benchmark_data.zip) and put the unzipped `ScanRefer_filtered_test.json` under `data/`. Then, you can run the following script the generate predictions:
```shell
python scripts/predict.py --folder <folder_name> --use_color
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`. The generated predictions are stored in `outputs/<folder_name>/pred.json`.
For submitting the predictions, please compress the `pred.json` as a .zip or .7z file and follow the [instructions](http://kaldir.vc.in.tum.de/scanrefer_benchmark/documentation) to upload your results.

## Dataset

If you would like to access to the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet Instructions](data/scannet/README.md) for more details.

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

## Setup
~~The code is tested on Ubuntu 16.04 LTS & 18.04 LTS with PyTorch 1.2.0 CUDA 10.0 installed. There are some issues with the newer version (>=1.3.0) of PyTorch. You might want to make sure you have installed the correct version. Otherwise, please execute the following command to install PyTorch:~~

The code is now compatiable with PyTorch 1.6! Please execute the following command to install PyTorch

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```
After all packages are properly installed, please run the following commands to compile the CUDA modules for the PointNet++ backbone:
```shell
cd lib/pointnet2
python setup.py install
```
__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__

### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:
```shell
cd data/scannet/
python batch_load_scannet_data.py
```
> After this step, you can check if the processed scene data is valid by running:
> ```shell
> python visualize.py --scene_id scene0000_00
> ```
<!-- 5. (Optional) Download the preprocessed [multiview features (~36GB)](http://kaldir.vc.in.tum.de/enet_feats.hdf5) and put it under `data/scannet/scannet_data/`. -->
5. (Optional) Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and decompress [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip).

    c. Change the data paths in `config.py` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    python script/compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds; you need ~36GB to store the generated HDF5 database:
    ```shell
    python script/project_multiview_features.py --maxpool
    ```
    > You can check if the projections make sense by projecting the semantic labels from image to the target point cloud by:
    > ```shell
    > python script/project_multiview_labels.py --scene_id scene0000_00 --maxpool
    > ```

## Usage
### Training
To train the ScanRefer model with RGB values:
```shell
python scripts/train.py --use_color
```
For more training options (like using preprocessed multiview features), please run `scripts/train.py -h`.

### Evaluation
To evaluate the trained ScanRefer models, please find the folder under `outputs/` with the current timestamp and run:
```shell
python scripts/eval.py --folder <folder_name> --reference --use_color --no_nms --force --repeat 5
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`

### Visualization
To predict the localization results predicted by the trained ScanRefer model in a specific scene, please find the corresponding folder under `outputs/` with the current timestamp and run:
```shell
python scripts/visualize.py --folder <folder_name> --scene_id <scene_id> --use_color
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`. The output `.ply` files will be stored under `outputs/<folder_name>/vis/<scene_id>/`

## Models
For reproducing our results in the paper, we provide the following training commands and the corresponding pre-trained models:

<table>
    <col>
    <col>
    <colgroup span="2"></colgroup>
    <colgroup span="2"></colgroup>
    <colgroup span="2"></colgroup>
    <col>
    <tr>
        <th rowspan=2>Name</th>
        <th rowspan=2>Command</th>
        <th colspan=2 scope="colgroup">Unique</th>
        <th colspan=2 scope="colgroup">Multiple</th>
        <th colspan=2 scope="colgroup">Overall</th>
        <th rowspan=2>Weights</th>
    </tr>
    <tr>
        <td>Acc<!-- -->@<!-- -->0.25IoU</td>
        <td>Acc<!-- -->@<!-- -->0.5IoU</td>
        <td>Acc<!-- -->@<!-- -->0.25IoU</td>
        <td>Acc<!-- -->@<!-- -->0.5IoU</td>
        <td>Acc<!-- -->@<!-- -->0.25IoU</td>
        <td>Acc<!-- -->@<!-- -->0.5IoU</td>
    </tr>
    <tr>
        <td>xyz</td>
        <td><pre lang="shell">python script/train.py --no_lang_cls</pre></td>
        <td>63.98</td>					
        <td>43.57</td>
        <td>29.28</td>
        <td>18.99</td>
        <td>36.01</td>
        <td>23.76</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+rgb</td>
        <td><pre lang="shell">python script/train.py --use_color --no_lang_cls</pre></td>
        <td>63.24</td>					
        <td>41.78</td>
        <td>30.06</td>
        <td>19.23</td>
        <td>36.5</td>
        <td>23.61</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_COLOR.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+rgb+normals</td>
        <td><pre lang="shell">python script/train.py --use_color --use_normal --no_lang_cls</pre></td>
        <td>64.63</td>					
        <td>43.65</td>
        <td>31.89</td>
        <td>20.77</td>
        <td>38.24</td>
        <td>25.21</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_COLOR_NORMAL.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+multiview</td>
        <td><pre lang="shell">python script/train.py --use_multiview --no_lang_cls</pre></td>
        <td>77.2</td>					
        <td>52.69</td>
        <td>32.08</td>
        <td>19.86</td>
        <td>40.84</td>
        <td>26.23</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_MULTIVIEW.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+multiview+normals</td>
        <td><pre lang="shell">python script/train.py --use_multiview --use_normal --no_lang_cls</pre></td>
        <td>78.22</td>					
        <td>52.38</td>
        <td>33.61</td>
        <td>20.77</td>
        <td>42.27</td>
        <td>26.9</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_MULTIVIEW_NORMAL.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+lobjcls</td>
        <td><pre lang="shell">python script/train.py</pre></td>
        <td>64.31</td>										
        <td>44.04</td>
        <td>30.77</td>
        <td>19.44</td>
        <td>37.28</td>
        <td>24.22</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_LANGCLS.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+rgb+lobjcls</td>
        <td><pre lang="shell">python script/train.py --use_color</pre></td>
        <td>65.00</td>										
        <td>43.31</td>
        <td>30.63</td>
        <td>19.75</td>
        <td>37.30</td>
        <td>24.32</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_COLOR_LANGCLS.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+rgb+normals+lobjcls</td>
        <td><pre lang="shell">python script/train.py --use_color --use_normal</pre></td>
        <td>67.64</td>					
        <td>46.19</td>
        <td>32.06</td>
        <td>21.26</td>
        <td>38.97</td>
        <td>26.10</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_COLOR_LANGCLS.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+multiview+lobjcls</td>
        <td><pre lang="shell">python script/train.py --use_multiview</pre></td>
        <td>76.00</td>															
        <td>50.40</td>
        <td>34.05</td>
        <td>20.73</td>
        <td>42.19</td>
        <td>26.50</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_MULTIVIEW_LANGCLS.zip>weights</a></td>
    </tr>
    <tr>
        <td>xyz+multiview+normals+lobjcls</td>
        <td><pre lang="shell">python script/train.py --use_multiview --use_normal</pre></td>
        <td>76.33</td>										
        <td>53.51</td>
        <td>32.73</td>
        <td>21.11</td>
        <td>41.19</td>
        <td>27.40</td>
        <td><a href=http://kaldir.vc.in.tum.de/scanrefer_pretrained_XYZ_MULTIVIEW_NORMAL_LANGCLS.zip>weights</a></td>
    </tr>
    
</table>

If you would like to try out the pre-trained models, please download the model weights and extract the folder to `outputs/`. Note that the results are higher than before because of a few iterations of code refactoring and bug fixing.

## Changelog
11/11/2020: Updated paper with the improved results due to bug fixing.

11/05/2020: Released pre-trained weights.

08/08/2020: Fixed the issue with `lib/box_util.py`.

08/03/2020: Fixed the issue with `lib/solver.py` and `script/eval.py`.

06/16/2020: Fixed the issue with multiview features.

01/31/2020: Fixed the issue with bad tokens.

01/21/2020: Released the ScanRefer dataset.

## Citation

If you use the ScanRefer data or code in your work, please kindly cite our work and the original ScanNet paper:

```
@article{chen2020scanrefer,
    title={ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language},
    author={Chen, Dave Zhenyu and Chang, Angel X and Nie{\ss}ner, Matthias},
    journal={16th European Conference on Computer Vision (ECCV)},
    year={2020}
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
