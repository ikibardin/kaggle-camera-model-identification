import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['VGG', 'vgg11_bn']

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}


def rotate_channels(x):
    out = torch.transpose(x, 1, 3)  # 0, 3, 2, 1
    out = torch.transpose(out, 2, 3)  # 0, 3, 1, 2
    return out


class MyAvgMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MyAvgMaxPool2d, self).__init__()
        self._maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self._avgpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return 0.5 * torch.sum(torch.stack(
            [self._maxpool(x), self._avgpool(x)]), 0).squeeze(dim=0)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.global_pool = MyAvgMaxPool2d(kernel_size=30, stride=1, padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1, 320),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(320, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, aug):
        # print(x.size())
        out = rotate_channels(x)
        out = self.features(out)
        out = self.global_pool(out)
        out = out.view(x.size(0), -1)
        # int(x.size(), aug.size())
        out = self.classifier(torch.cat((out, aug), 1))
        return out

    def get_feats(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
}


def _load_pretrained_weights(model, model_name):
    pretrained_weights = model_zoo.load_url(model_urls[model_name])
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'classifier' in key:
            continue
        state_dict[key] = pretrained_weights[key]
    model.load_state_dict(state_dict)
    return model


def vgg11_bn_modified(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, 'vgg11_bn')
    return model
