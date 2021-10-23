# Copyright (c) general_backbone. All rights reserved.
from general_backbone.models.resnet import resnet18, default_cfgs
import torch
from torchsummary import summary


if __name__ == '__main__':
    print('hello world!')
    device = 'cuda:0'
    model = resnet18(num_classes=2, pretrained=True).to(device)
    summary(model, (3, 224, 224))
    print(default_cfgs.keys())
