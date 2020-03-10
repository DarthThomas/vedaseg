## Introduction
vedaseg is an open source semantic segmentation toolbox based on PyTorch. This branch aims to construct a minimal inference-only version
## Features

- **Modular Design**

  We decompose the semantic segmentation framework into different components. The flexible and extensible design make it easy to implement a customized semantic segmentation project by combining different modules like building Lego.

- **Support of several popular frameworks**

  The toolbox supports several popular and semantic segmentation frameworks out of box, *e.g.* DeepLabv3+, DeepLabv3, U-Net, PSPNet, FPN, etc.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Note: All models are trained only on PASCAL VOC 2012 trainaug dataset and evaluated on PASCAL VOC 2012 val dataset.

| Architecture | backbone | OS | MS & Flip | mIOU|
|:---:|:---:|:---:|:---:|:---:|
| DeepLabv3plus | ResNet-101 | 16 | True | 79.80% |
| DeepLabv3plus | ResNet-101 | 16 | False | 78.19% |
| DeepLabv3 | ResNet-101 | 16 | True | 78.94% |
| DeepLabv3 | ResNet101 | 16 | False | 77.07% |
| FPN | ResNet-101 | 2 | True | 75.42% |
| FPN | ResNet-101 | 2 | False | 73.65% |
| PSPNet | ResNet-101 | 8 | True | 74.68% |
| PSPNet | ResNet-101 | 8 | False | 73.71% |
| U-Net | ResNet-101 | 1 | True | 73.09% |
| U-Net | ResNet-101 | 1 | False | 70.98% |

OS: Output stride used during evaluation\
MS: Multi-scale inputs during evaluation\
Flip: Adding left-right flipped inputs during evaluation

Models above are available in the [GoogleDrive](https://drive.google.com/drive/folders/1ooIOX5Aeu-0aHJYT1eZgzkSnZUvPi2by).

## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.1.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 9.0
- Python 3.7.3

### Install vedaseg

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedaseg python=3.7 -y
conda activate vedaseg
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedaseg repository.

```shell
git clone https://github.com/Media-Smart/vedaseg.git
cd vedaseg
vedaseg_root=${PWD}
```

d. Install dependencies.

```shell
pip install -r requirements.txt
```


## Test

a. Config

Modify some configuration accordingly in the like `configs/horc_1_50_d3p_481.py`

b. Run

```shell
python infer_template.py configs/horc_1_50_d3p_481.py path_to_deeplabv3plus_weights path_to_image
```

## Contact

This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).
