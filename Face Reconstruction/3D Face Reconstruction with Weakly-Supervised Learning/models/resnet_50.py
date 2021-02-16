import torch
import torch.nn as nn
import math
import pickle


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            # raise KeyError('unexpected key "{}" in state_dict'.format(name))
            print('unexpected key "{}" in state_dict'.format(name))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=-1, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # CHJ_ADD task use
        self.fc_dims = {
            "id": 80,
            "ex": 64,
            "tex": 80,
            "angles": 3,
            "gamma": 27,
            "XY": 2,
            "Z": 1}

        # self.fc_dims_arr=[0] * (1+len(self.fc_dims))
        # for i, (k, v) in enumerate(self.fc_dims.items()):
        #    self.fc_dims_arr[i+1] = v + self.fc_dims_arr[i]

        _outdim = 512 * block.expansion
        '''
        self.fcid = nn.Linear(_outdim, 80)
        self.fcex = nn.Linear(_outdim, 64)
        self.fctex = nn.Linear(_outdim, 80)
        self.fcangles = nn.Linear(_outdim, 3)
        self.fcgamma = nn.Linear(_outdim, 27)
        self.fcXY = nn.Linear(_outdim, 2)
        self.fcZ = nn.Linear(_outdim, 1)
        '''
        self.fcid = conv1x1(_outdim, 80)
        self.fcex = conv1x1(_outdim, 64)
        self.fctex = conv1x1(_outdim, 80)
        self.fcangles = conv1x1(_outdim, 3)
        self.fcgamma = conv1x1(_outdim, 27)
        self.fcXY = conv1x1(_outdim, 2)
        self.fcZ = conv1x1(_outdim, 1)

        self.arr_fc = [self.fcid, self.fcex, self.fctex,
                       self.fcangles, self.fcgamma, self.fcXY, self.fcZ]

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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # 这里不需要view
        n_b = x.size(0)
        outs = []
        for fc in self.arr_fc:
            outs.append(fc(x).view(n_b, -1))

        return outs


def resnet50_use():
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model
