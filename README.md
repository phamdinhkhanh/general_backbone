# Introduction

facilitates applied backbones, implements data augumentation and optimizes train models.

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
python3 tools/train.py --model resnet50 --data_dir toydata --batch-size 8 --output checkpoint/resnet50
```

You can study about config parameters in doc [](docs/training.md)

# Inference

# TODO

- [] code setup.py
- [] conda virtual environment setup
- [] Introduce about group of CNN models.
- [] Table ranking model performance.
- [] Guideline find list of all supported models.


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