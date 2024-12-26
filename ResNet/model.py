import torch.nn as nn
from torchvision import models

def resnet(num_classes: int, scale: str):
    match scale:
        case "s":
            model = models.resnet18(pretrained=False)
        case "m":
            model = models.resnet50(pretrained=False)
        case "l":
            model = models.resnet101(pretrained=False)
        case _:
            raise ValueError("Invalid scale")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 修改第一个卷积层
    model.conv1 = nn.Conv2d(
        in_channels=1,  # 单通道输入
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    return model
