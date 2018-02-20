import math

import torch
import torch.nn as nn

from .resnext2 import ResNeXt
from .se_module import SELayer

__all__ = ['se_resnext50', 'se_resnext101', 'se_resnext101_64', 'se_resnext152']


class SEBottleneck(nn.Module):
    """
    SE-RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None,
                 reduction=16):
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def load_weights_without_fc(model, path):
    # skip = ['fc.weight', 'fc.bias']
    pretrained_weights = torch.load(path)['state_dict']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'fc' in key:
            continue
        state_dict[key] = pretrained_weights[key]
    model.load_state_dict(state_dict)
    return model


def se_resnext50(num_classes=1000, pretrained=True):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    model = nn.DataParallel(model)
    if pretrained:
        model = load_weights_without_fc(model, '../../imagenet_pretrain/se_resnext50.pth')
    return model


def se_resnext101(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext101_64(num_classes=1000):
    """Constructs a SE-ResNeXt-101 (64x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext152(num_classes=1000):
    """Constructs a SE-ResNeXt-152 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model
