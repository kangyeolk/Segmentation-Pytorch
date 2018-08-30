import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        #NOTE: return 5 result
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        return h1, h2, h3, h4


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
    return model


class ChAttnBlock(nn.Module):
    """ Channel Attention Block """
    def __init__(self, in_dim):
        super(ChAttnBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_1 = nn.Conv2d(in_dim*2, in_dim, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        concat = torch.cat((low, high), dim=1)
        out = self.global_pool(concat)
        out = self.conv1x1_1(out)
        out = self.relu(out)
        out = self.conv1x1_2(out)
        scale = self.sigmoid(out)

        scale = scale.expand_as(low)
        low *= scale

        out = low + high
        return out


class RefnResBlock(nn.Module):
    """ Refinement Residual Block """
    def __init__(self, in_dim):
        super(RefnResBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_dim, 512, kernel_size=1)
        self.conv3x3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        h = self.conv3x3(x)
        h = self.batchnorm(h)
        h = self.relu(h)
        h = self.conv3x3(h)

        out = x + h
        out = self.relu(out)

        return out

""" Smooth Network """
class SmoothNet(nn.Module):
    def __init__(self, num_classes, h_image_size, w_image_size):
        super(SmoothNet, self).__init__()
        self.num_classes = num_classes
        self.H = h_image_size
        self.W = w_image_size

        # Pretrained model
        self.pre_model = resnet101(pretrained=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck_1 = nn.Conv2d(2048, 512, kernel_size=1)

        # Refinement Residual Block / Channel Attention Block
        self.rrb_1 = RefnResBlock(in_dim=256)
        self.rrb_2 = RefnResBlock(in_dim=512)
        self.rrb_3 = RefnResBlock(in_dim=1024)
        self.rrb_4 = RefnResBlock(in_dim=2048)
        self.rrb_last = RefnResBlock(in_dim=512) # For specification
        self.cab = ChAttnBlock(in_dim=512)
        self.bottleneck_2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1, h2, h3, h4 = self.pre_model(x) # 1/4, 1/8, 1/16, 1/32

        glob_pool = self.global_pool(h4)

        h1 = self.rrb_1(h1)
        h2 = self.rrb_2(h2)
        h3 = self.rrb_3(h3)
        h4 = self.rrb_4(h4)
        h4_up = F.upsample(glob_pool, size=(self.H // 32, self.W // 32), mode='bilinear') # 1/32
        h4_up = self.bottleneck_1(h4_up)

        h4 = self.rrb_last(self.cab(h4, h4_up)) # 1/32
        h3_up = F.upsample(h4, size=(self.H // 16, self.W // 16), mode='bilinear')
        h3 = self.rrb_last(self.cab(h3, h3_up)) # 1/16
        h2_up = F.upsample(h3, size=(self.H // 8, self.W // 8), mode='bilinear')
        h2 = self.rrb_last(self.cab(h2, h2_up)) # 1/8
        h1_up = F.upsample(h2, size=(self.H // 4, self.W // 4), mode='bilinear')
        h1 = self.rrb_last(self.cab(h1, h1_up)) # 1/4

        # 4x up
        out = F.upsample(h1, size=(self.H, self.W), mode='bilinear')
        out = self.bottleneck_2(out)
        out = self.sigmoid(out)

        return out


""" Border Network """
# TODO:

if __name__ == '__main__':
    model = resnet101()
    sample = torch.randn((2, 3, 128, 128))
    x, h1, h2, h3, h4 = model(sample)
    print(x.size(), h1.size(), h2.size(), h3.size(), h4.size())
