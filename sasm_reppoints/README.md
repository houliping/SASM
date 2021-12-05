# Source Code of AAAI22-2171

## Introduction

The source code includes training and inference procedures for the proposed method of the paper submitted to the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI 2022) with title "**Shape-Adaptive Selection and Measurement for Oriented Object Detection**" (ID: 2171).

This part of the code illustrates the effectiveness of the proposed method choosing [RepPoints](https://ieeexplore.ieee.org/document/9009032) as the baseline. The implementation of the baseline method for comparison, RepPoints with oriented bounding boxes, comes from [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/models/anchor_heads/cfa_head.py), which also adopts [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

It is recommended to get the MMDetection framework and some necessary CUDA functions  by installing [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox) directly and copy the  source code of the proposed method  to its source code tree for usage.

## Dependencies

- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.2
- CUDA 9.0+
- NCCL 2
- GCC (G++) 4.9+
- [mmdetection](https://github.com/open-mmlab/mmdetection) 1.1.0 
- [mmcv](https://github.com/open-mmlab/mmcv) 0.2.14
- [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox)


We have tested the code on the following versions of OS and softwares:

- OS:  Ubuntu 16.04 LTS
- Python: 3.7 (installed along with Anaconda 3)
- CUDA: 10.1
- NCCL: 2.3.7
- GCC (G++): 7.5
- PyTorch: 1.2

## Instructions for Usage
### Setp1: Create Environment
a. Create a virtual environment and activate it in Anaconda:

```bash
conda create -n reppoints python=3.7 -y
conda activate reppoints
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/):

```bash
conda install pytorch=1.2 torchvision cudatoolkit=10.0 -c pytorch
```
### Setp2: Install BeyondBoundingBox with MMDetection
a. Clone BeyondBoundingBox to current path.
```bash
git clone https://github.com/SDL-GuoZonghao/BeyondBoundingBox.git
cd BeyondBoundingBox
```
b. Install other dependencies and setup BeyondBoundingBox.
```bash
pip install -r requirements.txt
python setup.py develop
```

### Setp3: Prepare datasets
It is recommended to make a symbolic link of the dataset root to ``data`` and install [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) in BeyondBoundingBox root path.

The following instructions are for converting annotations of each dataset to the format of MMDetection. They are provided just for reference. You can achieve the same goal in whatever way convenient to you.

**Note**: the corresponding dataset path and class name should be changed in the python scripts according to actual position and setting of the data, and current path for executing the instructions is ``BeyondBoundingBox``.

- **DOTA**

  Please refer to [DOTA](https://captain-whu.github.io/DOTA/index.html) to get the training, validation and test set.
  Before training, the image-splitting process must be carried out.  See [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for details. 
```
python ./DOTA_devkit/DOTA2COCO.py 
```

- **HRSC2016**
	Please refer to [HRSC2016](https://sites.google.com/site/hrsc2016/) to get the training, validation and test set.
```bash
mv ../data_prepare/hrsc2016/HRSC2DOTA.py ./DOTA_devkit/
mv ../data_prepare/hrsc2016/HRSC2JSON.py ./DOTA_devkit/	
mv ../data_prepare/hrsc2016/prepare_hrsc2016.py ./DOTA_devkit/
python ./DOTA_devkit/prepare_hrsc2016.py 
```


-  **UCAS-AOD**
	Please refer to [UCAS-AOD](https://hyper.ai/datasets/5419 ) to get the training, validation and test set.
	Please run the following scripts in sequence, and note that the corresponding dataset paths should be changed in these scripts accordingly:
```bash
python ../data_prepare/ucas-aod/data_prepare.py
python ../data_prepare/ucas-aod/prepare_ucas.py
python ./DOTA_devkit/DOTA2COCO.py
```
- **ICDAR2015**
	Please refer to [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) to get the training, validation and test set. 
```bash
python ../data_prepare/icdar2015/prepare_ic15.py
python ./DOTA_devkit/DOTA2COCO.py
```


### Setp4: Install the source codes of SA-S and SA-M
**(a)** Add source files of the proposed method to BeyondBoundingBox (execute following commands in the sub-directory ``BeyondBoundingBox``)

```shell
 mv ../transforms_points.py ./mmdet/core/bbox/
 mv ../sas_assigner.py ./mmdet/core/bbox/assigners/
 mv ../sa_max_convex_iou_assigner.py ./mmdet/core/bbox/assigners/
 mv ../rreppoints_detector.py ./mmdet/models/detectors/
 mv ../sam_reppoints_head.py ./mmdet/models/anchor_heads/
 mv ../rreppoints_head.py ./mmdet/models/anchor_heads/
 mv ../atss_cfa_head.py ./mmdet/models/anchor_heads/
 rm ./mmdet/models/losses/iou_loss.py
 mv ../iou_loss.py ./mmdet/models/losses/
 mv ../icdar2015.py ./mmdet/datasets/
 mv ../ucas_aod.py ./mmdet/datasets/
 mv ../dota_beyond_read_pkl.py ./tools/
 mv -r ../data_prepare/icdar2015/icdar2015_evalution ./tools/ # see readme.txt for detail
 mv ../data_prepare/hrsc2016/hrsc2016_evaluation.py ./DOTA_devkit/
 mv ../data_prepare/ucas-aod/ucas_aod_evaluation.py ./DOTA_devkit/
```

**(b)** Import new core classes and functions in \__init__.py 

- Import core classes and functions in ``/mmdet/models/detectors/__init__.py``
```shell
from .rreppoints_detector import RRepPoints

__all__ = ['RRepPoints']
```

- Import core classes and functions in ``./mmdet/core/bbox/assigners/__init__.py`` 
```shell
from .sa_max_convex_iou_assigner import SAMaxConvexIoUAssigner
from .sas_assigner import SASAssigner

__all__ = ['SAMaxConvexIoUAssigner', 'SASAssigner']
```

- Import core classes and functions to ``./mmdet/models/anchor_heads/__init__.py`` 
```shell
from .rreppoints_head import RRepPointsHead
from .atss_cfa_head import ATSSCFAHead
from .sam_reppoints_head import SAMRepPointsHead
__all__ = ['RRepPointsHead', 'SAMRepPointsHead', 'ATSSCFAHead']
```

- Import core classes and functions to ``./mmdet/models/losses/__init__.py`` 
```shell
from .iou_loss import BCConvexGIoULoss
__all__ = ['BCConvexGIoULoss']
```


### Setp5: Prepare config files
For each dataset, we provide sample configures for both the baseline and our method in the files under the sub-directory ``configs``. For example, ``sasm_RepPoints_r101_3x_3ms_dota.py`` provides the configures for the proposed method running on DOTA dataset.

### Setp6: Train and test
Take the same instructions as BeyondBoundingBox. See [Training and Inference](https://github.com/SDL-GuoZonghao/BeyondBoundingBox) for details.





