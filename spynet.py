"""
Source: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
"""

import torch
import torch.nn as nn
import math
from skimage import filters, color
import numpy as np


class MaskLayer(nn.Module):
    def __init__(self, use_thermal=False):
        super(MaskLayer, self).__init__()
        self.use_thermal = use_thermal

    @staticmethod
    def thershold_mask(X, mean=[0.5], std=[0.5]):
        device = X.device
        std = torch.Tensor(std).to(device)[:, None]
        mean = torch.Tensor(mean).to(device)[:, None]
        X = X.clone().mul_(std).add_(mean).clamp(0, 1)
        X_fil = np.copy(X.clone().cpu().numpy())

        blurred_image = filters.gaussian(X_fil, sigma=1.0)
        t = filters.threshold_otsu(blurred_image)
        soft_mask = np.where(blurred_image > t, 1, blurred_image)
        soft_mask = torch.Tensor(soft_mask).to(device)
        return soft_mask

    def forward(self, x):
        device = x.device
        
        x_rgb = []
        
        for x_i in x:
        
            mask = self.thershold_mask(x_i[0])

            x_rgba = torch.zeros_like(x_i)
            x_rgba[0:3] = x_i[-3::]
            x_rgba[3] = mask
            x_rgba = x_rgba.permute(1, 2, 0).cpu().numpy()
            x_rgb.append(torch.Tensor(color.rgba2rgb(x_rgba)).permute(2, 0, 1).to(device))
        
        x_rgb = torch.stack(x_rgb)

        if self.use_thermal:
            x[:, -3::] = x_rgb
            return x
        else:
            return x_rgb


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
   

class SpyNet(nn.Module):
    """
    Adapted from MobileNetV2
    Takes 2 streams of input (normal and thermal images)
    """
    def __init__(self, n_class=2, input_size=224, width_mult=1., use_ori=True, use_thermal=True):
        super(SpyNet, self).__init__()
        assert use_ori or use_thermal is True
        self.use_ori = use_ori
        self.use_thermal = use_thermal
        
        block = InvertedResidual
        input_channel = 32
        if self.use_ori and self.use_thermal:
            last_channel = 640 # modified from 1280
        else:
            last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        if self.use_ori and self.use_thermal:
            self.features_1 = [conv_bn(3, input_channel, 2)] # for normal image input
            self.features_2 = [conv_bn(1, input_channel, 2)] # for thermal image input
        elif self.use_ori:
            self.features = [conv_bn(3, input_channel, 2)]
        else:
            self.features = [conv_bn(1, input_channel, 2)]
        
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    if self.use_ori and self.use_thermal:
                        self.features_1.append(block(input_channel, output_channel, s, expand_ratio=t))
                        self.features_2.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    if self.use_ori and self.use_thermal:
                        self.features_1.append(block(input_channel, output_channel, 1, expand_ratio=t))
                        self.features_2.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        if self.use_ori and self.use_thermal:
            # building last several layers
            self.features_1.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features_2.append(conv_1x1_bn(input_channel, self.last_channel))
            # make it nn.Sequential
            self.features_1 = nn.Sequential(*self.features_1)
            self.features_2 = nn.Sequential(*self.features_2)
        else:
            self.features.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features = nn.Sequential(*self.features)

        # building classifier
        if self.use_ori and self.use_thermal:
            self.classifier = nn.Linear(2*self.last_channel, n_class)
        else:
            self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        if self.use_ori and self.use_thermal:
            x_ori = x[:, -3::]
            x_thermal = x[:, 0:1]
            x_ori = self.features_1(x_ori)
            x_ori = x_ori.mean(3).mean(2)

            x_thermal = self.features_2(x_thermal)
            x_thermal = x_thermal.mean(3).mean(2)
            
            x = torch.cat((x_ori, x_thermal), -1)
        else:
            x = self.features(x)
            x = x.mean(3).mean(2)
        
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SpyNet_Mask(nn.Module):
    """
    Adapted from MobileNetV2
    Takes 2 streams of input (normal and thermal images)
    """
    def __init__(self, n_class=2, input_size=224, width_mult=1., use_ori=True, use_thermal=True):
        super(SpyNet_Mask, self).__init__()
        assert use_ori or use_thermal is True
        self.use_ori = use_ori
        self.use_thermal = use_thermal
        
        block = InvertedResidual
        input_channel = 32
        if self.use_ori and self.use_thermal:
            last_channel = 640 # modified from 1280
        else:
            last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        if self.use_ori and self.use_thermal:
            self.features_1 = [conv_bn(3, input_channel, 2)] # for normal image input
            self.features_2 = [conv_bn(1, input_channel, 2)] # for thermal image input
        elif self.use_ori:
            self.features = [conv_bn(3, input_channel, 2)]
        else:
            self.features = [conv_bn(1, input_channel, 2)]
        
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    if self.use_ori and self.use_thermal:
                        self.features_1.append(block(input_channel, output_channel, s, expand_ratio=t))
                        self.features_2.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    if self.use_ori and self.use_thermal:
                        self.features_1.append(block(input_channel, output_channel, 1, expand_ratio=t))
                        self.features_2.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        if self.use_ori and self.use_thermal:
            # building last several layers
            self.features_1.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features_2.append(conv_1x1_bn(input_channel, self.last_channel))
            # make it nn.Sequential
            self.features_1 = nn.Sequential(*self.features_1)
            self.features_2 = nn.Sequential(*self.features_2)
        else:
            self.features.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features = nn.Sequential(*self.features)

        # building classifier
        if self.use_ori and self.use_thermal:
            self.classifier = nn.Linear(2*self.last_channel, n_class)
        else:
            self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        mask = MaskLayer(use_thermal=True)
        x = mask(x)
        if self.use_ori and self.use_thermal:
            x_ori = x[:, -3::]
            x_thermal = x[:, 0:1]
            x_ori = self.features_1(x_ori)
            x_ori = x_ori.mean(3).mean(2)

            x_thermal = self.features_2(x_thermal)
            x_thermal = x_thermal.mean(3).mean(2)
            
            x = torch.cat((x_ori, x_thermal), -1)
        else:
            x = self.features(x)
            x = x.mean(3).mean(2)
        
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = SpyNet(n_class=2, width_mult=1, use_ori=True, use_thermal=True)
    print(net)