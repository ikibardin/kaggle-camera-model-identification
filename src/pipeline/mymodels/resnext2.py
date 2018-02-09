import math

import torch
import torch.nn as nn

from .common import rotate_channels

__all__ = ['resnext50', 'resnext101', 'resnext101_64']


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, baseWidth, cardinality, layers, num_classes):
        super(ResNeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion + 1, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth,
                            self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x, aug):
        x = rotate_channels(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(torch.cat((x, aug), 1))
        return x


def resnext50(num_classes=1000):
    """Constructs a ResNeXt-50 model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 6, 3], num_classes=num_classes)
    return model


def resnext101(num_classes=1000):
    """Constructs a ResNeXt-101 (32x4d) model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 4, 23, 3], num_classes=num_classes)
    return model


def resnext101_64(num_classes=1000):
    """Constructs a ResNeXt-101 (64x4d) model."""
    model = ResNeXt(Bottleneck, 4, 64, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnext151(num_classes=1000):
    """Constructs a ResNeXt-151 (32x4d) model."""
    model = ResNeXt(Bottleneck, 4, 32, [3, 8, 36, 3], num_classes=num_classes)
    return model
