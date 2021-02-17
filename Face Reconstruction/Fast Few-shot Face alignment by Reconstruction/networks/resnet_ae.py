import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'conv3x3']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.layer_norm =  layer_normalization
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LandmarkHead(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_size=256, input_channels=3, layer_normalization='batch'):
        super(LandmarkHead, self).__init__()
        self.input_size = input_size
        self.layer_norm = layer_normalization
         # always use name bnX so model weights can be found when loading pretrained models
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x2 = None
        self.inplanes = 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                norm2d(self.layer_norm)(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_normalization=self.layer_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(layer_normalization)(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(layer_normalization)(planes * 4)
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

def norm2d(type):
    if type == 'batch':
        return nn.BatchNorm2d
    elif type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise ValueError("Invalid normalization type: ", type)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, input_size=256, input_channels=3, layer_normalization='batch'):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer_norm = layer_normalization
         # always use name bnX so model weights can be found when loading pretrained models
        self.bn1 = norm2d(layer_normalization)(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x2 = None

        # self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

        self.with_additional_layers = True
        if input_size == 256 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        elif input_size == 512 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer01 = self._make_layer(block, 64, layers[0], stride=2)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                # torch.nn.utils.spectral_norm(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),#),
                norm2d(self.layer_norm)(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_normalization=self.layer_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        if self.input_size > 128 and self.with_additional_layers:
            x = self.layer0(x)
            if self.input_size > 256:
                x = self.layer01(x)

        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)
        x = self.layer3(self.x2)
        x = self.layer4(x)
        self.ft = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # x =  torch.nn.functional.normalize(x, dim=1)
        return x


class DiscriminatorHead(nn.Module):
    def __init__(self, **params):
        super(DiscriminatorHead, self).__init__()
        conv = conv3x3
        self.t1 = conv(64, 64)
        self.t2 = conv(128, 128)
        self.t3 = conv(256, 256)
        self.t4 = conv(512, 512)
        self.lin = nn.Linear(512 * 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, Q, qx1):
        # gen_layers = {
        #     'l2': Q.layer2,
        #     'l3': Q.layer3,
        #     'l4': Q.layer4,
        #     # 'l5': P.layer5
        # }

        x = qx1

        x = self.t1(x)
        # x = self.t1(x)

        # x = gen_layers['l2'](x)
        x = self.t2(x)

        # x = gen_layers['l3'](x)
        x = self.t3(x)

        # x = gen_layers['l4'](x)
        x = self.t4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.sigmoid(self.lin(x))

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except RuntimeError or KeyError as e:
            print(e)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        except RuntimeError as e:
            print(e)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        except RuntimeError as e:
            print(e)
    return model
