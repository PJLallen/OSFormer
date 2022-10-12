# FAQ

## Dataset settings

Following [SINet](https://github.com/DengPingFan/SINet) and other previous COD works, the original COD10K contains both camouflaged and non-camouflaged images. We only use camouflaged images with instance-level labels for training (3,040) and testing (2,026).
We have uploded the 3,040 training and 2,026 testing COD10K dataset Baidu/[Google](https://drive.google.com/file/d/1YGa3v-MiXy-3MMJDkidLXPt0KQwygt-Z/view?usp=sharing)/[Quark](https://pan.quark.cn/s/07ba3258b777).

| Dataset      |  CAM (instance-level) | NonCAM | Total |
|  ----        |         ----          | ----   | ----  |
| COD10K-Train |         3040          | 2960   | 6000  |
| COD10K-Test  |         2026          | 1974   | 4000  |

## Paper link

Arxiv: https://arxiv.org/abs/2207.02255

## Initial weights for PVT and Swin

To fit detectron2 framework, we add prefix to the key of pth. See https://github.com/PJLallen/OSFormer/issues/4 for details.
