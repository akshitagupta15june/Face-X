import math
# import matplotlib.pyplot as plt
# import numpy as np
from torch import nn as nn
from networks.resnet_ae import conv3x3
import torch


def deconv4x4(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)


def norm2d(type):
    if type == 'batch':
        return nn.BatchNorm2d
    elif type == 'instance':
        return nn.InstanceNorm2d
    elif type == 'none':
        return nn.Identity
    else:
        raise ValueError("Invalid normalization type: ", type)


class InvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch',
                 with_spectral_norm=False):
        super(InvBasicBlock, self).__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            self.conv1 = deconv4x4(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.stride = stride
        if with_spectral_norm:
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.layer_normalization == 'batch':
            out = self.bn1(out)
        elif self.layer_normalization == 'instance':
            out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.layer_normalization == 'batch':
            out = self.bn2(out)
        elif self.layer_normalization == 'instance':
            out = self.in2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return self.relu(out)


class LandmarkHead(nn.Module):

    def __init__(self, block, layers, output_size=128, output_channels=68, layer_normalization='none', start_layer=2):
        super(LandmarkHead, self).__init__()
        self.layer_normalization = layer_normalization
        self.lin_landmarks = None
        self.output_size = output_size
        self.output_channels = output_channels
        self.start_layer = start_layer
        self.lin = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, layer_normalization=self.layer_normalization))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class LandmarkHeadV2(LandmarkHead):
    def __init__(self,block, layers, **params):
        super(LandmarkHeadV2, self).__init__(block, layers, **params)
        conv = conv3x3
        self.t1 = conv(256, 256)
        self.t2 = conv(128, 128)
        self.t3 = conv(64, 64)
        self.t4 = conv(64, 64)
        # self.t5 = conv(64, 64)

    def forward(self, P):
        x = P.x1
        x = self.t1(x)

        x = P.layer2(x)
        x = self.t2(x)

        x = P.layer3(x)
        x = self.t3(x)

        x = P.layer4(x)
        x = self.t4(x)

        # if self.output_size > 128:
        #     x = self.layer5(x)
        #     x = self.t5(x)
        #     if self.output_size > 256:
        #         x = self.layer6(x)
        return self.lin(x)


class SemanticFeatureHead(LandmarkHead):
    def __init__(self,block, layers, **params):
        super(SemanticFeatureHead, self).__init__(block, layers, **params)
        conv = conv3x3
        self.t1 = conv(256, 256)
        self.t2 = conv(128, 128)
        self.t3 = conv(64, 64)
        self.t4 = conv(64, 64)
        self.t5 = conv(64, 64)

    def forward(self, P):
        inputs = {
            'x1': P.x1,
            'x2': P.x2.detach(),
            'x3': P.x3.detach(),
        }
        gen_layers = {
            'l1': P.layer1,
            'l2': P.layer2,
            'l3': P.layer3,
            'l4': P.layer4,
            # 'l5': P.layer5
        }

        x = inputs['x1']
        x = self.t1(x)

        x = gen_layers['l2'](x)
        x = self.t2(x)

        x = gen_layers['l3'](x)
        x = self.t3(x)

        x = gen_layers['l4'](x)
        x = self.t4(x)

        # if self.output_size > 128:
        #     # x = self.layer5(x)
        #     x = gen_layers['l5'](x)
        #     x = self.t5(x)
        #     if self.output_size > 256:
        #         x = self.layer6(x)

        x = self.lin(x)
        return torch.nn.functional.normalize(x, dim=1)



class InvResNet(nn.Module):

    def __init__(self, block, layers, output_size=256, output_channels=3, input_dims=99,
                 layer_normalization='none', spectral_norm=False):
        super(InvResNet, self).__init__()
        self.layer_normalization = layer_normalization
        self.with_spectral_norm = spectral_norm
        if self.with_spectral_norm:
            self.sn = torch.nn.utils.spectral_norm
        else:
            self.sn = lambda x: x

        self.lin_landmarks = None
        self.inplanes = 512
        self.output_size = output_size
        self.output_channels = output_channels
        self.fc = nn.Linear(input_dims, 512)
        self.conv1 = self.sn(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False))
        self.add_in_tensor = None

        self.norm = norm2d(layer_normalization)
        self.bn1 = self.norm(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        inplanes_after_layer1 = self.inplanes # self.inplanes gets changed in _make_layers
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        inplanes_after_layer2 = self.inplanes # self.inplanes gets changed in _make_layers
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.tanh = nn.Tanh()
        self.x2 = None
        if self.output_size == 256:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
        elif self.output_size == 512:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
            self.layer6 = self._make_layer(block,  64, layers[3], stride=2)

        self.lin = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_finetuning(self, batchsize):
        if self.add_in_tensor is None or self.add_in_tensor.shape[0] != batchsize:
            self._create_finetune_layers(batchsize)
        else:
            self._reset_finetune_layers()

    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                self.sn(nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=4, stride=stride, padding=1, bias=False)),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,
                            layer_normalization=self.layer_normalization,
                            with_spectral_norm=self.with_spectral_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.fc(x)
        x = x.view(x.size(0), -1, 1,1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.x0 = x

        x1 = self.layer1(x)
        self.x1 = x1
        self.x2 = self.layer2(x1)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)

        if self.output_size == 128:
            x = self.x4
        elif self.output_size == 256:
            x = self.layer5(self.x4)
            self.x5 = x
        elif self.output_size == 512:
            x = self.layer5(self.x4)
            x = self.layer6(x)

        x = self.lin(x)
        x = self.tanh(x)
        return x

