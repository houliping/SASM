# Source Code of AAAI22-2171

## Introduction

The source code includes training and inference procedures for the proposed method of the paper submitted to the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI 2022) with title "**Shape-Adaptive Selection and Measurement for Oriented Object Detection**" (ID: 2171).

The the effectiveness of the proposed method is verified on two baseline methods. Corresponding source code and configurations reside in following two sub-directories:

* ``sasm_reppoints``: the implementation and verification using [RepPoints](https://ieeexplore.ieee.org/document/9009032) as baseline;
* ``sasm_s2anet``: the implementation and verification using [S$^2$A-Net](https://ieeexplore.ieee.org/document/9377550) as baseline.

We provide only the source code related to the proposed method in the sub-directories so that reviewers can check them quickly and conveniently.

Please refer to the ``README.md`` file in each sub-directory for the detailed instructions of usage.

### Introduction



|   Method   | Assignment |   Reg. Loss   | **Tricks**  |  mAP  |
| :--------: | :--------: | :-----------: | :---------: | :---: |
| RepPoints  |   MaxIoU   |     GIoU      |      -      | 70.46 |
| RepPoints  |    SASM    | BCLoss + GIoU |      -      | 74.27 |
| RepPoints  |    SASM    | BCLoss + GIoU | MS training | 77.19 |
| S$^2$A-Net |    SASM    |   Smooth L1   | MS training | 79.17 |



### Reference

1、https://github.com/open-mmlab/mmdetection

2、https://github.com/SDL-GuoZonghao/BeyondBoundingBox

3、https://github.com/csuhan/s2anet

4、https://github.com/sfzhang15/ATSS
