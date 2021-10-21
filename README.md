# Introduction

facilitates implementing deep neural-network backbones, data augmentations, optimizers and learning schedulers.

- backbones :
- loss functions :
- augumentation styles :
- optimizers :
- schedulers :
- data types :

# Installation

# Train model

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
- [] Introduce about group of CNN models.
- [] Table ranking model performance.
- [] Guideline find list of all supported models.
- [] Support new type of Datasets: You can change the augmentation styles:
    - references: https://albumentations.ai/docs/examples/pytorch_classification/
- [] New loss function: 
    - Focal Loss function; KL divergence.
    - references: https://github.com/pytorch/pytorch/blob/3097755e7a88333c945a14ee44fda055ba276138/torch/nn/modules/loss.py; https://pytorch.org/docs/stable/nn.html#loss-functions


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