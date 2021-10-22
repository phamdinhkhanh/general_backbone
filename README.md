# Introduction

facilitates implementing deep neural-network backbones, data augmentations, optimizers and learning schedulers.

- [x] backbones
- loss functions
- [x] augumentation styles
- optimizers
- schedulers
- data types

# Installation

Refer to [docs/installation.md](docs/installation.md) for installion of `general_backbone` package.

# Train model

## Model backone

Currently, `general_backbone` supports more than 70 type of **resnet** models such as: `resnet18, resnet34, resnet50, resnet101, resnet152, resnext50`.

All models is supported can be found in `general_backbone.list_models()` function:

```
import general_backbone
general_backbone.list_models()
```

Results

```
{'resnet': ['resnet18', 'resnet18d', 'resnet34', 'resnet34d', 'resnet26', 'resnet26d', 'resnet26t', 'resnet50', 'resnet50d', 'resnet50t', 'resnet101', 'resnet101d', 'resnet152', 'resnet152d', 'resnet200', 'resnet200d', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'tv_resnext50_32x4d', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet50t', 'seresnet101', 'seresnet152', 'seresnet152d', 'seresnet200d', 'seresnet269d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_32x8d', 'senet154', 'ecaresnet26t', 'ecaresnetlight', 'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet50t', 'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnet200d', 'ecaresnet269d', 'ecaresnext26t_32x4d', 'ecaresnext50t_32x4d', 'resnetblur18', 'resnetblur50', 'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420']}
```

To select your backbone type, you set model=`resnet50` in train_config of your config file. An example config file [general_backbone/configs/image_clf_config.py](general_backbone/configs/image_clf_config.py).

## Dataset

A toy dataset is provided at `toydata` for your test training. It has a structure organized as below:

```
toydata/
└── image_classification
    ├── test
    │   ├── cat
    │   └── dog
    └── train
        ├── cat
        └── dog
```

Inside each folder cat and dog is the images. If you want to add a new class, you just need to create a new folder with the folder's name is label name inside `train` and `test` folder.

## Data Augmentation

`general_backbone` package support many augmentations style for training. It is efficient and important to improve model accuracy. Some of common augumentations is below:

| Augumentation Style     | Parameters                  |  Description                                                 |
| ------------------------- | ----------------------------- | ---------------------------------- |
| **Pixel-level transforms** |
| Blur | {'blur_limit':7, 'always_apply':False, 'p':0.5} | Blur the input image using a random-sized kernel | 
| GaussNoise | | {'var_limit':(10.0, 50.0), 'mean':0, 'per_channel':True, 'always_apply':False, 'p':0.5} | Apply gaussian noise to the input image |
| GaussianBlur | {'blur_limit':(3, 7), 'sigma_limit':0, 'always_apply':False, 'p':0.5} | Blur the input image using a Gaussian filter with a random kernel size |
| GlassBlur | {'sigma': 0.7, 'max_delta':4, 'iterations':2, 'always_apply':False, 'mode':'fast', 'p':0.5} | Apply glass noise to the input image |
| HueSaturationValue | {'hue_shift_limit':20, 'sat_shift_limit':30, 'val_shift_limit':20, 'always_apply':False, 'p':0.5 | Randomly change hue, saturation and value of the input image |
| MedianBlur | {'blur_limit':7, 'always_apply':False, 'p':0.5} | Blur the input image using a median filter with a random aperture linear size |
| RGBShift | {'r_shift_limit': 15, 'g_shift_limit': 15, 'b_shift_limit': 15, 'p': 0.5} | Randomly shift values for each channel of the input RGB image. |
| Normalize | {'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)} | Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)` |
| **Spatial-level transforms** |
| RandomCrop | {'height':128, 'width':128} | Crop a random part of the input |
| VerticalFlip | {'p': 0.5} | Flip the input vertically around the x-axis |
| ShiftScaleRotate | {'shift_limit':0.05, 'scale_limit':0.05, 'rotate_limit':15, 'p':0.5} | Randomly apply affine transforms: translate, scale and rotate the input |
| RandomBrightnessContrast | {'brightness_limit':0.2, 'contrast_limit':0.2, 'brightness_by_max':True, 'always_apply':False,'p': 0.5} | Randomly change brightness and contrast of the input image |

Augumentation is configured in the configuration file [general_backbone/configs/image_clf_config.py](general_backbone/configs/image_clf_config.py):

```
data_conf = dict(
    dict_transform = dict(
        SmallestMaxSize={'max_size': 160},
        ShiftScaleRotate={'shift_limit':0.05, 'scale_limit':0.05, 'rotate_limit':15, 'p':0.5},
        RandomCrop={'height':128, 'width':128},
        RGBShift={'r_shift_limit': 15, 'g_shift_limit': 15, 'b_shift_limit': 15, 'p': 0.5},
        RandomBrightnessContrast={'p': 0.5},
        Normalize={'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
        ToTensorV2={'always_apply':True}
    )
)
```
You can add a new transformation step in `data_conf['dict_transform']` and they are transformed in order from top-down. You can also debug your transformation by setup `debug=True`:

```
from general_backbone.data import AugmentationDataset
augdataset = AugmentationDataset(data_dir='toydata/image_classification',
                            name_split='train',
                            config_file = 'general_backbone/configs/image_clf_config.py', 
                            dict_transform=None, 
                            input_size=(256, 256), 
                            debug=True, 
                            dir_debug = 'tmp/alb_img_debug', 
                            class_2_idx=None)

for i in range(50):
    img, label = augdataset.__getitem__(i)
```

In default, the augmentation images output is saved in `tmp/alb_img_debug` to you review before train your models. the code tests augmentation image is available in [debug/transform_debug.py](tools/transform_debug.py):

```
conda activate gen_backbone
python tools/transform_debug.py
```

## Train model


To train model, you run file `tools/train.py`. There are variaty of config for your training such as `--model, --batch_size, --opt, --loss, --sched`.

```
python3 tools/train.py --config general_backbone/configs/image_clf_config.py
```

Results:
```
Model resnet50 created, param count:25557032
Train: 0 [   0/33 (  0%)]  Loss: 8.863 (8.86)  Time: 1.663s,    9.62/s  (1.663s,    9.62/s)  LR: 5.000e-04  Data: 0.460 (0.460)
Train: 0 [  32/33 (100%)]  Loss: 1.336 (4.00)  Time: 0.934s,    8.57/s  (0.218s,   36.68/s)  LR: 5.000e-04  Data: 0.000 (0.014)
Test: [   0/29]  Time: 0.560 (0.560)  Loss:  0.6912 (0.6912)  Acc@1: 87.5000 (87.5000)  Acc@5: 100.0000 (100.0000)
Test: [  29/29]  Time: 0.041 (0.064)  Loss:  0.5951 (0.5882)  Acc@1: 81.2500 (87.5000)  Acc@5: 100.0000 (99.3750)
Train: 1 [   0/33 (  0%)]  Loss: 0.5741 (0.574)  Time: 0.645s,   24.82/s  (0.645s,   24.82/s)  LR: 5.000e-04  Data: 0.477 (0.477)
Train: 1 [  32/33 (100%)]  Loss: 0.5411 (0.313)  Time: 0.089s,   90.32/s  (0.166s,   48.17/s)  LR: 5.000e-04  Data: 0.000 (0.016)
Test: [   0/29]  Time: 0.537 (0.537)  Loss:  0.3071 (0.3071)  Acc@1: 87.5000 (87.5000)  Acc@5: 100.0000 (100.0000)
Test: [  29/29]  Time: 0.043 (0.066)  Loss:  0.1036 (0.1876)  Acc@1: 100.0000 (93.9583)  Acc@5: 100.0000 (100.0000)
```

You can study about config parameters in [training](docs/training.md)

# Inference



# TODO

- [x] code setup.py
- [x] conda virtual environment setup
- [x] Introduce group of CNN models support
- [] Table ranking model performances.
- [x] Support new type of Datasets: You can change the augmentation styles:
    - references: https://albumentations.ai/docs/examples/pytorch_classification/
- [] New loss function: 
    - Focal Loss function; KL divergence.
    - references: https://github.com/pytorch/pytorch/blob/3097755e7a88333c945a14ee44fda055ba276138/torch/nn/modules/loss.py; https://pytorch.org/docs/stable/nn.html#loss-functions

# Package reference:

There are many open sources package we refered to build up `general_backbone`:

- [timm](https://github.com/rwightman/pytorch-image-models): PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

- [albumentations](https://github.com/albumentations-team/albumentations): is a Python library for image augmentation.

- [mmcv](https://github.com/open-mmlab/mmcv): MMCV is a foundational library for computer vision research and supports many research projects.

# Citation

If you find this project is useful in your reasearch, kindly consider cite:

```
@article{genearal_ocr,
    title={GeneralOCR:  A Comprehensive package for DeepLearning Backbone models},
    author={khanhphamdinh},
    email= {phamdinhkhanh.tkt53.neu@gmail.com},
    year={2021}
}
```