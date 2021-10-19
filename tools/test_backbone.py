from general_backbone.models.resnet import resnet18, default_cfgs
import torch
from torchsummary import summary


if __name__ == '__main__':
    print('hello world!')
    device = 'cuda:0'
    model = resnet18(pretrained=True).to(device)

    # x = torch.randn(4, 3, 224, 224)
    # pred = model(x)


    # print(pred)
    summary(model, (3, 224, 224))
    print(default_cfgs.keys())
