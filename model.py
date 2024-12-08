# import torch
# import torch.nn as nn
# from torchvision import models

# def get_pretrained_model(num_classes):
#     # 加载预训练的 ResNeXt101_32x8d 模型
#     # model = models.resnext101_32x8d(pretrained=True)
#     model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
#     # 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False
#     # 替换最后的全连接层
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
#     return model

import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes):
    # 加载预训练的 ResNeSt200 模型
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻倒数第一层
    # 假设倒数第一层是 layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 替换最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 重新设置最后一层的参数为可训练
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
