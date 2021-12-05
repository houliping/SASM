# Source Code of AAAI22-2171

## Introduction

The source code includes training and inference procedures for the proposed method of the paper submitted to the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI 2022) with title "**Shape-Adaptive Selection and Measurement for Oriented Object Detection**" (ID: 2171).

This part of the code illustrates the effectiveness of the proposed method choosing [S$^2$A-Net](https://github.com/csuhan/s2anet) as the baseline. Please refer to the publications of S$^2$A-Net in arXiv  ([arXiv:2008.09397](https://arxiv.org/abs/2008.09397)) and IEEE TGRS (https://ieeexplore.ieee.org/document/9377550) for details. 

The implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection) framework, just as S$^2$A-Net does. 

It is recommended to get the MMDetection framework and some useful tools by installing [S$^2$A-Net](https://github.com/csuhan/s2anet) directly and copy the source code of the proposed method  to its source code tree for usage.


## Dependencies 
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.3+
- CUDA 9.0+
- NCCL 2
- GCC (G++) 4.9+
- [mmdetection](https://github.com/open-mmlab/mmdetection) 1.1.0 
- [mmcv](https://github.com/open-mmlab/mmcv) 0.2.14
- [s2anet](https://github.com/csuhan/s2anet)

Note that some CUDA extensions, e.g. ```box_iou_rotated``` and ```nms_rotated```, require PyTorch>=1.3 and GCC>=4.9.

We have tested the code on the following versions of OS and softwares:

- OS:  Ubuntu 16.04 LTS
- Python: 3.7 (installed along with Anaconda 3)
- CUDA: 10.0/10.1
- NCCL: 2.3.7
- GCC (G++): 7.5
- PyTorch: 1.3.1

## Instructions for Usage

### Setp1: Create Environment
a. Create a virtual environment and activate it in Anaconda:

```shell
conda create -n s2anet python=3.7 -y
conda activate s2anet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/):

```shell
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```
### Setp2:  Install S$^2$A-Net with MMDetection
a. Clone s2anet to current path.
```shell
git clone https://github.com/csuhan/s2anet.git
cd s2anet
```
b. Install other dependencies and setup s2anet.
```shell
pip install -r requirements.txt
python setup.py develop
```

### Setp3:  Prepare datasets
It is recommended to make a symbolic link of the dataset root to ``data`` and install [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) in s2anet root path.

The following instructions are for converting annotations of each dataset to the format of MMDetection. They are provided just for reference. You can achieve the same goal in whatever way convenient to you.

**Note**: the corresponding dataset path and class name should be changed in the python scripts according to actual position and setting of the data, and current path for executing the instructions is ``s2anet``.

- **DOTA**
	Please refer to [DOTA](https://captain-whu.github.io/DOTA/index.html) to get the training, validation and test set.
	Before training, the image-splitting process must be carried out.  See  [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for details. 
```bash
mv ../data_prepare/dota/convert_dota_to_mmdet.py ./DOTA_devkit/
python ./DOTA_devkit/convert_dota_to_mmdet.py 
```

- **HRSC2016**
	Please refer to [HRSC2016](https://sites.google.com/site/hrsc2016/) to get the training, validation and test set.
	
-  **UCAS-AOD**
	Please refer to [UCAS-AOD](https://hyper.ai/datasets/5419) to get the training, validation and test set.
	Please run the following files in sequence, and note that the corresponding dataset paths should be changed.
```bash
python ../data_prepare/ucas-aod/data_prepare.py
python ../data_prepare/ucas-aod/prepare_ucas.py
python ./DOTA_devkit/convert_dota_to_mmdet.py 
```
- **ICDAR2015**
	
	Please refer to [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) to get the training, validation and test set. 
```
python ../data_prepare/icdar2015/prepare_ic15.py
python ./DOTA_devkit/convert_dota_to_mmdet.py 
```


### Setp4: Install the source codes of SA-S and SA-M to s2anet
**(a)** Add source code files to s2anet (execute following commands in the sub-directory ``s2anet``)

```shell
 mv ../sam_anchor_target.py ./mmdet/core/anchor/
 mv ../sas_assigner.py ./mmdet/core/bbox/assigners/
 mv ../sa_max_iou_assigner.py ./mmdet/core/bbox/assigners/
 mv ../sam_s2anet_head.py ./mmdet/models/anchor_heads_rotated/
 mv ../icdar2015.py ./mmdet/datasets/
 mv ../ucasaod.py ./mmdet/datasets/
 mv ../icdar_nms.py ./DOTA_devkit/
 mv ../ucasaod_nms.py ./DOTA_devkit/
```
**(b)** Import new core classes and functions in \__init__.py 

- Import core classes and functions in ``./mmdet/core/anchor/__init__.py``
```shell
from .sam_anchor_target import sam_anchor_inside_flags, sam_anchor_target, sam_unmap, sam_images_to_levels

__all__ = ['sam_anchor_target',
    'sam_anchor_inside_flags', 'sam_unmap', 'sam_images_to_levels']
```

- Import core classes and functions in ``./mmdet/core/bbox/assigners/__init__.py`` 
```shell
from .sa_max_iou_assigner import SAMaxIoUAssigner
from .sas_assigner import SASAssigner

__all__ = ['SAMaxIoUAssigner', 'SASAssigner']
```

- Import core classes and functions to ``./mmdet/models/anchor_heads_rotated/__init__.py`` 
```shell
from .sam_s2anet_head import SAMS2ANetHead
__all__ = ['SAMS2ANetHead']
```
### Setp5: Write config files
For each dataset, we provide sample configures for both the baseline and our method in the files under the sub-directory ``configs``. For example, ``configs/dota/sasm_s2anet_rx101_2x_dota.py`` provides the configures for the proposed method running on DOTA dataset.

### Setp6: Train and test
Take the same instructions as s2anet. See [getting_started.md](https://github.com/csuhan/s2anet/blob/master/docs/GETTING_STARTED.md) for details.





