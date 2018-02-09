import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models


class AvgPool(nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, (x.size(2), x.size(3)))

def rotate_channels(x):
    out = torch.transpose(x, 1, 3)  # 0, 3, 2, 1
    out = torch.transpose(out, 2, 3)  # 0, 3, 1, 2
    return out

class ResNet(nn.Module):
    def __init__(self, num_classes, net_cls=models.resnet50, pretrained=False):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        self.net.avgpool = AvgPool()

        self.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features + 1, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.net.fc = nn.Dropout(0.0)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x, O):  # 0, 1, 2, 3 -> (0, 3, 1, 2)
        out = torch.transpose(x, 1, 3)  # 0, 3, 2, 1
        out = torch.transpose(out, 2, 3)  # 0, 3, 1, 2
        out = self.net(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, O], 1)
        return self.fc(out)


class ConcatenatedAvgMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(ConcatenatedAvgMaxPool, self).__init__()
        self._maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self._avgpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        avg_x = self._avgpool(x)
        max_x = self._maxpool(x)
        out = torch.cat((avg_x, max_x), dim=1)
        return out


class VGG11NoPooling(nn.Module):
    def __init__(self, num_classes):
        super(VGG11NoPooling, self).__init__()
        filters_list = 4 * [64] + 2 * [128]
        layers = []
        in_channels = 3
        for filters in filters_list:
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            in_channels = filters
        self.features = nn.Sequential(*layers)
        self.global_pool = ConcatenatedAvgMaxPool(kernel_size=512, stride=1,
                                                  padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 1, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, aug):
        x = rotate_channels(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, aug), 1)
        x = self.classifier(x)
        return x
