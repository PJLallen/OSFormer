# OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers (ECCV 2022)

![OSFormer](docs/OSFormer.png)

Official Implementation of "[OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers](https://arxiv.org/abs/2207.02255)"

[Jialun Pei*](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en), [Tianyang Cheng*](https://github.com/Patrickctyyx), [Deng-Ping Fan](https://dengpingfan.github.io/), [He Tang](https://scholar.google.com/citations?hl=en&user=70XLFUsAAAAJ), Chuanbo Chen, and [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html)

[[Paper]](https://arxiv.org/abs/2207.02255); [[Chinese Version]](https://dengpingfan.github.io/papers/[2022][ECCV]OSFormer_Chinese.pdf); [[Official Version]](https://link.springer.com/content/pdf/10.1007/978-3-031-19797-0_2.pdf); [[Project Page]](https://blog.patrickcty.cc/OSFormer-Homepage/)

**Contact:** dengpfan@gmail.com, peijl@hust.edu.cn

|            *Sample 1*             |             *Sample 2*             |             *Sample 3*             |             *Sample 4*             |
| :------------------------------: | :-------------------------------: | :-------------------------------: | :-------------------------------: |
| <img src="docs/COD10K-CAM-3-Flying-53-Bird-3024.gif"  height=125 width=170> | <img src="docs/COD10K-CAM-3-Flying-65-Owl-4620.gif" height=125 width=170> | <img src="docs/488.gif" height=125 width=170> |  <img src="docs/4126.gif" height=125 width=170> |

## Environment preparation

The code is tested on CUDA 11.1 and pytorch 1.9.0, change the versions below to your desired ones.

```shell
git clone https://github.com/PJLallen/OSFormer.git
cd OSFormer
conda create -n osformer python=3.8 -y
conda activate osformer
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```

## Dataset preparation

### Download the datasets

- **COD10K**: [Baidu](https://pan.baidu.com/s/1IPcPjdg1EJ-h9HPoU42nHA) (password:hust) / [Google](https://drive.google.com/file/d/1YGa3v-MiXy-3MMJDkidLXPt0KQwygt-Z/view?usp=sharing) / [Quark](https://pan.quark.cn/s/07ba3258b777); **Json files:** [Baidu](https://pan.baidu.com/s/1kRawj-hzBDycCkZZfQjFhg) (password:hust) / [Google](https://drive.google.com/drive/folders/1Yvz63C8c7LOHFRgm06viUM9XupARRPif?usp=sharing)
- **NC4K**: [Baidu](https://pan.baidu.com/s/1li4INx4klQ_j8ftODyw2Zg) (password:hust) / [Google](https://drive.google.com/file/d/1eK_oi-N4Rmo6IIxUNbYHBiNWuDDLGr_k/view?usp=sharing); **Json files:** [Baidu](https://pan.baidu.com/s/1DBPFtAL2iEjefwiqXE_GWA) (password:hust) / [Google](https://drive.google.com/drive/folders/1LyK7tl2QVZBFiNaWI_n0ZVa0QiwF2B8e?usp=sharing)

### Register datasets

1. generate coco annotation files, you may refer to [the tutorial of mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/2_new_data_model.md) for some help
2. change the path of the datasets as well as annotations in `adet/data/datasets/cis.py`, please refer to [the docs of detectron2](https://detectron2.readthedocs.io/en/latest/) for more help

```python
# adet/data/datasets/cis.py
# change the paths 
DATASET_ROOT = 'COD10K-v3'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train/Image')
TEST_PATH = os.path.join(DATASET_ROOT, 'Test/Image')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train_instance.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test2026.json')

NC4K_ROOT = 'NC4K'
NC4K_PATH = os.path.join(NC4K_ROOT, 'Imgs')
NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')
```

## Pre-trained models

Model weights: [Baidu](https://pan.baidu.com/s/1Ao3Myqa6xiA9ymAkZgZOeQ) (password:l6vn) / [Google](https://drive.google.com/drive/folders/1pl9iM1NAfN5N6Voc03oPmlbKJ-YNldMF?usp=sharing) / [Quark](https://pan.quark.cn/s/6676592ff08b)

| Model         | Config                                           | COD10K-test AP | NC4K-test AP |
|:--------------|:------------------------------------------------ |:---------------|:-------------|
| R50-550       | [configs/CIS_RT.yaml](configs/CIS_RT.yaml)       | 36.0           | 41.4         |
| R50           | [configs/CIS_R50.yaml](configs/CIS_R50.yaml)     | 41.0           | 42.5         |
| R101          | [configs/CIS_R101.yaml](configs/CIS_R101.yaml)   | 42.0           | 44.4         |
| PVTv2-B2-Li   | [configs/CIS_PVTv2B2Li](configs/CIS_PVTv2B2Li)   | 47.2           | 50.5         |
| SWIN-T        | [configs/CIS_SWINT.yaml](configs/CIS_SWINT.yaml) | 47.7           | 50.2         |

## Visualization results

The visual results are achieved by our OSFormer with ResNet-50 trained on the COD10K training set.

- Results on the COD10K test set: [Baidu](https://pan.baidu.com/s/16xH7coaGoOGiB5x1AXgy5w) (password:hust) /
[Google](https://drive.google.com/open?id=16XMw6NaiCQdHG1By-1a7s8SmnyEqlmYD)
- Results on the NC4K test set: [Baidu](https://pan.baidu.com/s/15Y-7fNcHRhu38Vjybx1HMg) (password:hust) /
[Google](https://drive.google.com/file/d/1cRcwbD3Y3fMO3n7eTtA6VGZWKCWwJSU0/view?usp=sharing)

## Frequently asked questions

[FAQ](https://github.com/PJLallen/OSFormer/blob/main/docs/faq.md)

## Usage

### Train

```shell
python tools/train_net.py --config-file configs/CIS_R50.yaml --num-gpus 1 \
  OUTPUT_DIR {PATH_TO_OUTPUT_DIR}
```

Please replace `{PATH_TO_OUTPUT_DIR}` to your own output dir

### Inference

```shell
python tools/train_net.py --config-file configs/CIS_R50.yaml --eval-only \
  MODEL.WEIGHTS {PATH_TO_PRE_TRAINED_WEIGHTS}
```

Please replace `{PATH_TO_PRE_TRAINED_WEIGHTS}` to the pre-trained weights

### Eval

```shell
python demo/demo.py --config-file configs/CIS_R50.yaml \
  --input {PATH_TO_THE_IMG_DIR_OR_FIRE} \
  --output {PATH_TO_SAVE_DIR_OR_IMAGE_FILE} \
  --opts MODEL.WEIGHTS {PATH_TO_PRE_TRAINED_WEIGHTS}
```

- `{PATH_TO_THE_IMG_DIR_OR_FIRE}`: you can put image dir or image paths here
- `{PATH_TO_SAVE_DIR_OR_IMAGE_FILE}`: the place where the visualizations will be saved
- `{PATH_TO_PRE_TRAINED_WEIGHTS}`: please put the pre-trained weights here


## Acknowledgement

This work is based on:
- [detectron2](https://github.com/facebookresearch/detectron2)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- [DETR](https://github.com/facebookresearch/detr)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)

We also get help from [mmdetection](https://github.com/open-mmlab/mmdetection). Thanks them for their great work!

## Citation

If this helps you, please cite this work:

```
@inproceedings{pei2022osformer,
  title={OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers},
  author={Pei, Jialun and Cheng, Tianyang and Fan, Deng-Ping and Tang, He and Chen, Chuanbo and Van Gool, Luc},
  booktitle={European conference on computer vision},
  year={2022},
  organization={Springer}
}
```
