# Copyright (c) general_backbone. All rights reserved.
#!/usr/bin/env python
import argparse
import os
import os.path as osp
import cv2
import torch
import datetime
from general_backbone import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import yaml
import logging

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data.dataloader import DataLoader
import torchvision
from general_backbone.data import create_dataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from general_backbone.models import create_model, safe_model_name 
from general_backbone.utils import resume_checkpoint, load_checkpoint, model_parameters
from general_backbone.layers import convert_splitbn_model
from general_backbone.utils import *
from general_backbone.loss import *
from general_backbone.optim import create_optimizer_v2, optimizer_kwargs
from general_backbone.scheduler import create_scheduler
from general_backbone.utils import ApexScaler, NativeScaler
from general_backbone.data.loader import create_loader_aug
from general_backbone.utils import ConfigDict, Config
from general_backbone.data import AlbImageDataset, AugmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


parser = argparse.ArgumentParser(description='General backbone model training', add_help=False)
parser.add_argument('--img', default='demo/cat0.jpg', type=str, metavar='PATH',
                    help='link to image file')

parser.add_argument('-c', '--config', default='general_backbone/configs/image_clf_config.py', type=str, metavar='FILE',
                    help='python config file specifying default arguments')

parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')

args, remaining = parser.parse_known_args()
print(args)
cfg = Config.fromfile(args.config)

def main():
    
    # initialize model
    model = create_model(
        cfg.train_conf.model,
        pretrained=False,
        num_classes=cfg.train_conf.num_classes,
        checkpoint_path=args.initial_checkpoint
        )

    print('************model configuration************')
    print('model: ', cfg.train_conf.model)
    print('pretrained: ', False)
    print('num_classes: ', cfg.train_conf.num_classes)
    print('checkpoint_path: ', args.initial_checkpoint)
    
    # move model to GPU
    model.cuda()
    print('************model prediction************')
    img = cv2.imread(args.img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = torch.unsqueeze(torch.Tensor(img).permute(2, 0, 1), 0).cuda()
    prob = torch.exp(model(img))/torch.sum(torch.exp(model(img)))
    argpred = torch.argmax(prob, 1)
    print('label: {}, prob: {}'.format(argpred[0].detach().cpu().numpy(), prob[0][argpred[0]].detach().cpu().numpy()))
    return prob
if __name__ == '__main__':
    main()
